//! Hardware Crossbar Protocols — Session Types for FPGA Communication
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! Research question: Can session types prevent communication mismatches
//! in FPGA crossbar interconnects the same way they prevent shuffle-from-
//! inactive-lane bugs in GPU warps?
//!
//! # The Mapping
//!
//! | GPU Warp                | FPGA Crossbar                         |
//! |-------------------------|---------------------------------------|
//! | Lane                    | Tile (processing element)             |
//! | Active set (mask)       | Tiles in compatible protocol phase    |
//! | `shuffle_xor`           | Crossbar butterfly exchange           |
//! | Inactive lane read = UB | Recv from non-sending tile = stale    |
//! | `Warp<All>`             | All tiles in same protocol phase      |
//! | `diverge`               | Tiles take different code paths       |
//! | `merge` + ComplementOf  | All tiles rejoin same phase           |
//!
//! # The Real Hardware
//!
//! Models the vcpu-d 16-tile pipelined crossbar (`crossbar_ntile.v`):
//! - N² pipeline registers (one per src×dst pair)
//! - valid/ready handshake per channel
//! - Channel index = destination tile ID (`ch[dst] = tile[dst]`)
//! - 64-bit messages, 1-cycle latency
//!
//! The crossbar provides *runtime* backpressure (valid/ready stalls if
//! receiver isn't ready). Session types provide *compile-time* guarantee
//! that communication partners are in compatible states — the type error
//! fires before synthesis, not as a hardware deadlock.
//!
//! # The Bug Class (Hardware)
//!
//! In vcpu-d, `SEND ch, reg` puts data on channel ch → tile ch's RX.
//! `RECV ch, reg` reads from channel ch → tile ch's TX to us.
//!
//! The bug: Tile 0 takes a branch where it doesn't SEND. Tile 3 still
//! does `RECV 0, a1`. The pipeline register from the *previous* cycle
//! is still valid — tile 3 reads stale data. No hardware error.
//! This is *exactly* shuffle-from-inactive-lane, but in silicon.

use std::marker::PhantomData;

// ============================================================================
// TILE SET TYPES (analogous to ActiveSet)
// ============================================================================

/// Marker trait for tile subsets — analogous to `ActiveSet` for lane subsets.
///
/// Each tile set is a zero-sized type encoding which tiles are participating
/// in the current communication phase. 16-bit mask for up to 16 tiles
/// (matching the vcpu-d crossbar).
pub trait TileSet: Copy + 'static {
    const MASK: u16;
    const NAME: &'static str;
}

/// Proof that tile sets A and B are complements: disjoint and covering all tiles.
///
/// Same role as `ComplementOf` in the warp type system. Required by `merge_tiles`
/// to prove all tiles are accounted for before collective operations resume.
pub trait TileComplement<Other: TileSet>: TileSet {}

/// All 16 tiles active.
#[derive(Copy, Clone, Debug)]
pub struct AllTiles;
impl TileSet for AllTiles { const MASK: u16 = 0xFFFF; const NAME: &'static str = "AllTiles"; }

/// No tiles active.
#[derive(Copy, Clone, Debug)]
pub struct NoTiles;
impl TileSet for NoTiles { const MASK: u16 = 0x0000; const NAME: &'static str = "NoTiles"; }

/// Lower half: tiles 0–7.
#[derive(Copy, Clone, Debug)]
pub struct LowerHalf;
impl TileSet for LowerHalf { const MASK: u16 = 0x00FF; const NAME: &'static str = "LowerHalf"; }

/// Upper half: tiles 8–15.
#[derive(Copy, Clone, Debug)]
pub struct UpperHalf;
impl TileSet for UpperHalf { const MASK: u16 = 0xFF00; const NAME: &'static str = "UpperHalf"; }

/// Even tiles: 0, 2, 4, ..., 14.
#[derive(Copy, Clone, Debug)]
pub struct EvenTiles;
impl TileSet for EvenTiles { const MASK: u16 = 0x5555; const NAME: &'static str = "EvenTiles"; }

/// Odd tiles: 1, 3, 5, ..., 15.
#[derive(Copy, Clone, Debug)]
pub struct OddTiles;
impl TileSet for OddTiles { const MASK: u16 = 0xAAAA; const NAME: &'static str = "OddTiles"; }

// Complement pairs — same pattern as warp active sets
impl TileComplement<OddTiles> for EvenTiles {}
impl TileComplement<EvenTiles> for OddTiles {}
impl TileComplement<UpperHalf> for LowerHalf {}
impl TileComplement<LowerHalf> for UpperHalf {}
impl TileComplement<NoTiles> for AllTiles {}
impl TileComplement<AllTiles> for NoTiles {}

// ============================================================================
// TILE GROUP (analogous to Warp<S>)
// ============================================================================

/// A group of tiles parameterized by which tiles are participating.
///
/// Zero-sized — `PhantomData<S>` only. Same erasure as `Warp<S>`.
/// Crossbar collective operations (ring_pass, butterfly, scatter, gather)
/// are only available on `TileGroup<AllTiles>`. After `diverge`, they
/// vanish from the type — the method literally does not exist.
pub struct TileGroup<S: TileSet> {
    _set: PhantomData<S>,
}

impl<S: TileSet> TileGroup<S> {
    pub fn new() -> Self { TileGroup { _set: PhantomData } }
    pub fn active_mask(&self) -> u16 { S::MASK }
}

// ============================================================================
// CROSSBAR COLLECTIVE OPERATIONS (analogous to shuffle/ballot)
// ============================================================================
//
// Only implemented for TileGroup<AllTiles>. After diverge, these methods
// vanish — calling ring_pass on TileGroup<LowerHalf> is a compile error.
// This is the same mechanism as shuffle on Warp<Active>.

impl TileGroup<AllTiles> {
    /// Ring pass: tile i sends to tile (i+1) % 16, receives from (i-1) % 16.
    ///
    /// Models the common FPGA communication pattern where each tile passes
    /// its result to its neighbor. Requires all tiles active — if any tile
    /// diverges, the neighbor reads stale pipeline register data.
    pub fn ring_pass(&self, data: &[u64; 16]) -> [u64; 16] {
        let mut result = [0u64; 16];
        for i in 0..16 {
            result[(i + 1) % 16] = data[i];
        }
        result
    }

    /// Butterfly exchange: tiles at distance `stride` swap data.
    ///
    /// Direct analog of `shuffle_xor(data, stride)` in GPU warps.
    /// tile[i] gets data from tile[i ^ stride].
    /// Foundation for butterfly reductions across the crossbar.
    pub fn butterfly(&self, data: &[u64; 16], stride: usize) -> [u64; 16] {
        assert!(stride > 0 && stride < 16, "stride must be 1..15");
        let mut result = [0u64; 16];
        for i in 0..16 {
            result[i] = data[i ^ stride];
        }
        result
    }

    /// Scatter: tile `src` distributes data[i] to tile i via crossbar channels.
    ///
    /// In vcpu-d terms: tile `src` does `SEND i, data[i]` for each channel i.
    /// All other tiles do `RECV src, reg`. Requires all tiles active to receive.
    pub fn scatter(&self, _src: usize, data: &[u64; 16]) -> [u64; 16] {
        *data
    }

    /// Gather: each tile sends its value; tile `dst` collects all.
    ///
    /// In vcpu-d terms: each tile i does `SEND dst, value`.
    /// Tile `dst` does `RECV i, reg` for each channel i.
    pub fn gather(&self, data: &[u64; 16], _dst: usize) -> [u64; 16] {
        *data
    }

    /// Butterfly reduction across all 16 tiles.
    ///
    /// Same algorithm as warp shuffle reduction:
    /// ```text
    /// stride = 8: pairs (0,8), (1,9), ..., (7,15)
    /// stride = 4: pairs (0,4), (1,5), ..., (3,7), (8,12), ...
    /// stride = 2: pairs (0,2), (1,3), ...
    /// stride = 1: pairs (0,1), (2,3), ...
    /// ```
    /// After 4 rounds, tile 0 has the sum. All tiles must participate
    /// in each butterfly step — diverged tiles would contribute stale data.
    pub fn reduce_sum(&self, data: &[u64; 16]) -> u64 {
        let mut work = *data;
        let mut stride = 8;
        while stride >= 1 {
            let exchanged = self.butterfly(&work, stride);
            for i in 0..16 {
                work[i] = work[i].wrapping_add(exchanged[i]);
            }
            stride >>= 1;
        }
        work[0]
    }
}

// ============================================================================
// DIVERGE / MERGE
// ============================================================================

impl TileGroup<AllTiles> {
    /// Split tiles by half — tiles 0-7 vs tiles 8-15.
    ///
    /// After this, neither half can perform crossbar collectives.
    /// Each half can do independent computation. Must merge before
    /// any crossbar operation that spans the full tile set.
    pub fn diverge_halves(self) -> (TileGroup<LowerHalf>, TileGroup<UpperHalf>) {
        (TileGroup::new(), TileGroup::new())
    }

    /// Split tiles by parity — even tiles vs odd tiles.
    pub fn diverge_parity(self) -> (TileGroup<EvenTiles>, TileGroup<OddTiles>) {
        (TileGroup::new(), TileGroup::new())
    }
}

/// Merge complementary tile groups back to AllTiles.
///
/// Same `ComplementOf` proof as warp `merge`. The compiler rejects merges
/// of non-complementary tile groups — you can't claim all tiles are present
/// when some are missing.
pub fn merge_tiles<A, B>(
    _a: TileGroup<A>,
    _b: TileGroup<B>,
) -> TileGroup<AllTiles>
where
    A: TileSet + TileComplement<B>,
    B: TileSet,
{
    TileGroup::new()
}

// ============================================================================
// VCPU-D ISA MAPPING
// ============================================================================

/// Models the vcpu-d SEND/RECV instructions through the crossbar.
///
/// In vcpu-d:
///   SEND ch, rs  →  tile_tx_valid[self * 16 + ch] = 1, data = regs[rs]
///   RECV ch, rd  →  rd = tile_rx_data[self * 16 + ch] (if tile_rx_valid)
///
/// The session-typed version: SEND/RECV require the tile group to be in
/// AllTiles state — ensuring the partner tile is actually participating.
pub struct CrossbarPort {
    tile_id: usize,
    channels: [Option<u64>; 16],
}

impl CrossbarPort {
    pub fn new(tile_id: usize) -> Self {
        CrossbarPort { tile_id, channels: [Option::None; 16] }
    }

    /// SEND ch, value — stage a message on channel ch.
    /// In hardware: sets tile_tx_valid[tile_id * 16 + ch] = 1.
    pub fn send(&mut self, channel: usize, value: u64) {
        assert!(channel < 16);
        self.channels[channel] = Some(value);
    }

    /// Check if a send is staged on a channel.
    pub fn is_sending(&self, channel: usize) -> bool {
        self.channels[channel].is_some()
    }

    pub fn tile_id(&self) -> usize { self.tile_id }
}

/// A full crossbar connecting N tiles — runtime model of crossbar_ntile.v.
///
/// The pipeline registers are modeled as Option<u64>: None = not valid,
/// Some(v) = valid with data v. This mirrors the hardware's pipe_valid/pipe_data.
pub struct Crossbar {
    pipe: [[Option<u64>; 16]; 16],  // pipe[src][dst]
}

impl Crossbar {
    pub fn new() -> Self {
        Crossbar { pipe: [[Option::None; 16]; 16] }
    }

    /// Clock the crossbar: accept sends from ports, deliver to receivers.
    /// Models one cycle of crossbar_ntile.v's always @(posedge clk) block.
    pub fn clock(&mut self, ports: &[CrossbarPort]) {
        for port in ports {
            let src = port.tile_id();
            for dst in 0..16 {
                if let Some(data) = port.channels[dst] {
                    self.pipe[src][dst] = Some(data);
                }
                // Note: if port did NOT send on this channel,
                // the old pipe data stays — this is the stale data bug!
            }
        }
    }

    /// Receive: tile `dst` reads from tile `src`'s pipeline register.
    /// Returns whatever is in the register — could be stale if src didn't send.
    pub fn recv(&self, dst: usize, src: usize) -> Option<u64> {
        self.pipe[src][dst]
    }

    /// Safe receive: only reads if the source tile is in the active set.
    /// This is what session types enforce at compile time.
    pub fn recv_checked<S: TileSet>(
        &self,
        _proof: &TileGroup<S>,
        dst: usize,
        src: usize,
    ) -> Option<u64> {
        if S::MASK & (1 << src) != 0 {
            self.pipe[src][dst]
        } else {
            Option::None  // Source not in active set — would be stale
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tile set properties ---

    #[test]
    fn tile_sets_are_correct() {
        assert_eq!(AllTiles::MASK, 0xFFFF);
        assert_eq!(NoTiles::MASK, 0x0000);
        assert_eq!(LowerHalf::MASK, 0x00FF);
        assert_eq!(UpperHalf::MASK, 0xFF00);
        assert_eq!(EvenTiles::MASK, 0x5555);
        assert_eq!(OddTiles::MASK, 0xAAAA);
    }

    #[test]
    fn tile_complements_cover_all() {
        assert_eq!(LowerHalf::MASK | UpperHalf::MASK, AllTiles::MASK);
        assert_eq!(LowerHalf::MASK & UpperHalf::MASK, 0);
        assert_eq!(EvenTiles::MASK | OddTiles::MASK, AllTiles::MASK);
        assert_eq!(EvenTiles::MASK & OddTiles::MASK, 0);
    }

    // --- Crossbar collectives ---

    #[test]
    fn ring_pass_rotates() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();
        let data: [u64; 16] = core::array::from_fn(|i| i as u64);

        let result = tiles.ring_pass(&data);

        // tile[1] should have tile[0]'s data, tile[2] has tile[1]'s, etc.
        for i in 0..16 {
            assert_eq!(result[(i + 1) % 16], data[i]);
        }
    }

    #[test]
    fn butterfly_exchange_swaps_pairs() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();
        let data: [u64; 16] = core::array::from_fn(|i| i as u64 * 10);

        // stride=1: swap adjacent pairs
        let result = tiles.butterfly(&data, 1);
        assert_eq!(result[0], data[1]);  // tile 0 gets tile 1's data
        assert_eq!(result[1], data[0]);  // tile 1 gets tile 0's data
        assert_eq!(result[2], data[3]);
        assert_eq!(result[3], data[2]);

        // stride=8: swap tiles 8 apart
        let result = tiles.butterfly(&data, 8);
        assert_eq!(result[0], data[8]);
        assert_eq!(result[8], data[0]);
        assert_eq!(result[7], data[15]);
        assert_eq!(result[15], data[7]);
    }

    #[test]
    fn butterfly_reduction_sums_all() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();
        let data: [u64; 16] = [1; 16];

        let sum = tiles.reduce_sum(&data);
        // Each tile sees its value doubled 4 times: 1→2→4→8→16
        assert_eq!(sum, 16);
    }

    #[test]
    fn butterfly_reduction_distinct_values() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();
        let data: [u64; 16] = core::array::from_fn(|i| (i + 1) as u64);
        // Sum = 1+2+...+16 = 136

        let sum = tiles.reduce_sum(&data);
        assert_eq!(sum, 136);
    }

    // --- Diverge / merge cycle ---

    #[test]
    fn diverge_merge_ring_pass() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();
        let data: [u64; 16] = core::array::from_fn(|i| i as u64);

        // Diverge — each half does independent work
        let (lower, upper) = tiles.diverge_halves();
        assert_eq!(lower.active_mask(), 0x00FF);
        assert_eq!(upper.active_mask(), 0xFF00);

        // Cannot call ring_pass on lower or upper — method doesn't exist.
        // lower.ring_pass(&data);  // COMPILE ERROR

        // Merge back — now collectives work again
        let all = merge_tiles(lower, upper);
        let result = all.ring_pass(&data);
        assert_eq!(result[1], 0);  // tile 0's data moved to tile 1
    }

    #[test]
    fn diverge_parity_merge() {
        let tiles: TileGroup<AllTiles> = TileGroup::new();

        let (evens, odds) = tiles.diverge_parity();
        assert_eq!(evens.active_mask(), 0x5555);
        assert_eq!(odds.active_mask(), 0xAAAA);

        // Merge back
        let all = merge_tiles(evens, odds);
        let data = [42u64; 16];
        let sum = all.reduce_sum(&data);
        assert_eq!(sum, 42 * 16);
    }

    // --- Stale data bug demonstration ---

    #[test]
    fn stale_data_bug_demonstration() {
        // This test demonstrates the hardware bug that session types prevent.
        //
        // Scenario: tile 0 sends to tile 1, then tile 0 diverges (takes a
        // branch where it doesn't send). Tile 1 reads from tile 0 again
        // and gets the OLD data — the pipeline register wasn't cleared.

        let mut xbar = Crossbar::new();

        // Cycle 1: tile 0 sends 0xDEAD to tile 1
        let mut port0 = CrossbarPort::new(0);
        port0.send(1, 0xDEAD);
        xbar.clock(&[port0]);

        assert_eq!(xbar.recv(1, 0), Some(0xDEAD));  // tile 1 gets correct data

        // Cycle 2: tile 0 diverges — does NOT send.
        // But the pipeline register still holds 0xDEAD!
        let port0_idle = CrossbarPort::new(0);  // no sends staged
        assert!(!port0_idle.is_sending(1));
        // Note: we don't clock with the idle port — the old data persists.

        // tile 1 reads again — gets STALE data
        let stale = xbar.recv(1, 0);
        assert_eq!(stale, Some(0xDEAD));  // BUG: still sees old value

        // With session types, this would be caught:
        // After diverge, tile 1 cannot call recv from tile 0's group
        // because the method doesn't exist on the diverged TileGroup.
    }

    #[test]
    fn checked_recv_prevents_stale_read() {
        let mut xbar = Crossbar::new();

        // tile 0 sends
        let mut port0 = CrossbarPort::new(0);
        port0.send(1, 0xBEEF);
        xbar.clock(&[port0]);

        // With AllTiles proof, recv works
        let all: TileGroup<AllTiles> = TileGroup::new();
        assert_eq!(xbar.recv_checked(&all, 1, 0), Some(0xBEEF));

        // After diverge, only upper half (tiles 8-15) is active.
        // Tile 0 is in lower half — recv_checked returns None.
        let upper: TileGroup<UpperHalf> = TileGroup::new();
        assert_eq!(xbar.recv_checked(&upper, 1, 0), Option::None);
        // Tile 0 (in LowerHalf) is not in UpperHalf — read blocked.
    }

    // --- vcpu-d communication pattern ---

    #[test]
    fn vcpud_ring_send_recv_pattern() {
        // Models the vcpu-d pattern: each tile sends to (id+1) % N
        // This is a ring topology over the crossbar.
        let mut xbar = Crossbar::new();

        // Each tile stages SEND to its right neighbor
        let ports: Vec<CrossbarPort> = (0..16).map(|id| {
            let mut port = CrossbarPort::new(id);
            port.send((id + 1) % 16, id as u64 * 100);
            port
        }).collect();

        xbar.clock(&ports);

        // Each tile receives from its left neighbor
        for id in 0..16 {
            let src = (id + 16 - 1) % 16;  // left neighbor
            let data = xbar.recv(id, src);
            assert_eq!(data, Some(src as u64 * 100));
        }
    }

    #[test]
    fn zero_overhead_tile_group() {
        // TileGroup<S> is PhantomData<S> — zero-sized, same as Warp<S>
        assert_eq!(std::mem::size_of::<TileGroup<AllTiles>>(), 0);
        assert_eq!(std::mem::size_of::<TileGroup<LowerHalf>>(), 0);
        assert_eq!(std::mem::size_of::<TileGroup<EvenTiles>>(), 0);
    }
}
