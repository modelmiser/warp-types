//! Affine clause tokens — compile-time prevention of double-assignment.
//!
//! The #1 bug class in parallel SAT solvers: two threads propagating the same
//! clause, producing contradictory learned clauses. Every existing solver
//! uses runtime checks. `ClauseToken` makes it a compile error.
//!
//! `ClauseToken` is non-Copy, non-Clone. Moving it transfers ownership.
//! Dropping it releases the clause back to the pool.
//!
//! # Example
//!
//! ```
//! use warp_types_sat::clause::*;
//!
//! let mut pool = ClausePool::new(100); // 100 clauses
//!
//! // Take ownership of clause 42
//! let token = pool.acquire(42).unwrap();
//! assert_eq!(token.index(), 42);
//!
//! // Can't acquire the same clause again
//! assert!(pool.acquire(42).is_none());
//!
//! // Release returns it to the pool
//! pool.release(token);
//! assert!(pool.acquire(42).is_some());
//! ```

use core::sync::atomic::{AtomicU32, Ordering};

/// Global pool-identity counter. Each `ClausePool` draws a unique nonce at
/// construction; tokens are stamped with it at acquire so `release` can verify
/// a token is returned to the pool that issued it (cross-pool release would
/// free a slot the pool never gave out, defeating the exclusivity invariant).
///
/// Drawn through [`next_nonce`], never `fetch_add` — uniqueness is the whole
/// point of this counter, and a wrapping `fetch_add` would quietly reissue
/// nonce 0 and make `release`'s cross-pool check pass on a foreign token.
static POOL_NONCE: AtomicU32 = AtomicU32::new(0);

/// Draw the next pool identity, or panic rather than repeat one.
///
/// Returns the pre-increment value, so identities run `0 ..= u32::MAX - 1`.
/// On exhaustion the counter saturates at `u32::MAX` and every later call
/// panics too — a wrapping counter would hand the 2^32-nd pool the identity
/// the first pool is still using, and `release` would then accept that pool's
/// token and free a slot whose owner is still live. Silent exclusivity loss
/// is exactly what `ClauseToken` exists to prevent, so this fails loudly.
///
/// Split out from `ClausePool::new` so the exhaustion path is reachable in a
/// test without allocating 2^32 pools.
fn next_nonce(counter: &AtomicU32) -> u32 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect(
            "ClausePool: pool-identity counter exhausted (2^32 pools created) — \
             the next identity would repeat one already in use and release() \
             would accept a foreign token",
        )
}

// ============================================================================
// Clause token (affine — non-Copy, non-Clone)
// ============================================================================

/// An affine ownership token for a clause.
///
/// Holding a `ClauseToken` proves exclusive access to that clause index.
/// No other thread/lane can acquire the same clause until this token is
/// released. The compiler enforces this — `ClauseToken` is neither `Copy`
/// nor `Clone`.
///
/// This is the clause-level analog of `Warp<S>`: you can't clone a warp
/// token and use it in two places. You can't clone a clause token either.
#[must_use = "dropping a ClauseToken without releasing it leaks the clause reservation"]
pub struct ClauseToken {
    index: usize,
    /// Identity of the pool that issued this token (see `POOL_NONCE`).
    pool_nonce: u32,
}

// Deliberately NOT deriving Copy or Clone.

impl ClauseToken {
    /// Which clause this token owns.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl core::fmt::Debug for ClauseToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ClauseToken({})", self.index)
    }
}

// ============================================================================
// Clause pool
// ============================================================================

/// A pool of clause tokens enforcing exclusive ownership.
///
/// Each clause index can be acquired at most once. Acquiring returns
/// `None` if already held. Releasing returns the token to the pool.
///
/// In a real solver, the pool tracks which clauses are assigned to
/// which tiles/lanes for BCP. The affine discipline ensures no clause
/// is processed by two tiles simultaneously.
pub struct ClausePool {
    /// Bitset tracking which clauses are currently acquired.
    /// True = acquired (unavailable), False = available.
    acquired: Vec<bool>,
    /// Unique pool identity, stamped into every token this pool issues.
    nonce: u32,
}

impl ClausePool {
    /// Create a pool for `num_clauses` clauses, all initially available.
    pub fn new(num_clauses: usize) -> Self {
        ClausePool {
            acquired: vec![false; num_clauses],
            nonce: next_nonce(&POOL_NONCE),
        }
    }

    /// Number of clauses in the pool.
    pub fn len(&self) -> usize {
        self.acquired.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.acquired.is_empty()
    }

    /// Number of currently available (unacquired) clauses.
    pub fn available(&self) -> usize {
        self.acquired.iter().filter(|&&a| !a).count()
    }

    /// Acquire exclusive ownership of a clause. Returns `None` if already acquired
    /// or out of range.
    pub fn acquire(&mut self, index: usize) -> Option<ClauseToken> {
        if index >= self.acquired.len() || self.acquired[index] {
            return None;
        }
        self.acquired[index] = true;
        Some(ClauseToken {
            index,
            pool_nonce: self.nonce,
        })
    }

    /// Acquire the next available clause (lowest index). Returns `None` if
    /// all clauses are currently acquired.
    pub fn acquire_next(&mut self) -> Option<ClauseToken> {
        let index = self.acquired.iter().position(|&a| !a)?;
        self.acquired[index] = true;
        Some(ClauseToken {
            index,
            pool_nonce: self.nonce,
        })
    }

    /// Release a clause token back to the pool.
    ///
    /// Consumes the token — the clause becomes available for re-acquisition.
    ///
    /// # Panics
    /// Panics if the token was issued by a different pool. Releasing pool B's
    /// token into pool A would free A's slot while A's own token is still
    /// live, silently breaking the exclusivity invariant.
    pub fn release(&mut self, token: ClauseToken) {
        assert_eq!(
            token.pool_nonce, self.nonce,
            "releasing clause token {} into a different pool than issued it",
            token.index
        );
        debug_assert!(
            self.acquired[token.index],
            "releasing clause {} that wasn't acquired",
            token.index
        );
        self.acquired[token.index] = false;
        // Token is consumed (moved into this function), so it can't be used again.
    }

    /// Release multiple tokens at once.
    pub fn release_batch(&mut self, tokens: Vec<ClauseToken>) {
        for token in tokens {
            self.release(token);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_release() {
        let mut pool = ClausePool::new(10);
        assert_eq!(pool.available(), 10);

        let t0 = pool.acquire(0).unwrap();
        assert_eq!(pool.available(), 9);

        // Can't acquire same clause twice
        assert!(pool.acquire(0).is_none());

        // Release makes it available again
        pool.release(t0);
        assert_eq!(pool.available(), 10);
        assert!(pool.acquire(0).is_some());
    }

    #[test]
    fn acquire_next() {
        let mut pool = ClausePool::new(3);
        let _t0 = pool.acquire_next().unwrap(); // gets 0
        let t1 = pool.acquire_next().unwrap(); // gets 1
        let _t2 = pool.acquire_next().unwrap(); // gets 2
        assert!(pool.acquire_next().is_none()); // exhausted
        assert_eq!(t1.index(), 1);
    }

    #[test]
    fn out_of_range() {
        let mut pool = ClausePool::new(5);
        assert!(pool.acquire(5).is_none());
        assert!(pool.acquire(100).is_none());
    }

    #[test]
    fn batch_release() {
        let mut pool = ClausePool::new(5);
        let tokens: Vec<_> = (0..5).map(|_| pool.acquire_next().unwrap()).collect();
        assert_eq!(pool.available(), 0);
        pool.release_batch(tokens);
        assert_eq!(pool.available(), 5);
    }

    #[test]
    #[should_panic(expected = "different pool")]
    fn cross_pool_release_panics() {
        // Releasing pool B's token into pool A would free A's slot 1 while
        // A's own token for slot 1 is still live — exclusivity broken.
        let mut a = ClausePool::new(4);
        let mut b = ClausePool::new(4);
        let _ta = a.acquire(1).unwrap();
        let tb = b.acquire(1).unwrap();
        a.release(tb);
    }

    #[test]
    fn nonces_are_distinct_and_monotonic() {
        let c = AtomicU32::new(0);
        assert_eq!(next_nonce(&c), 0);
        assert_eq!(next_nonce(&c), 1);
        assert_eq!(next_nonce(&c), 2);
    }

    #[test]
    fn last_nonce_before_exhaustion_is_still_issued() {
        let c = AtomicU32::new(u32::MAX - 1);
        assert_eq!(next_nonce(&c), u32::MAX - 1);
        assert_eq!(c.load(Ordering::Relaxed), u32::MAX, "counter must saturate");
    }

    #[test]
    #[should_panic(expected = "counter exhausted")]
    fn exhausted_nonce_counter_panics_rather_than_wrapping() {
        // With `fetch_add` this returned u32::MAX and wrapped the counter to
        // 0, so the next pool created would share pool 0's identity and
        // release() would accept its tokens.
        let c = AtomicU32::new(u32::MAX);
        let _ = next_nonce(&c);
    }

    #[test]
    fn exhaustion_stays_exhausted() {
        let c = AtomicU32::new(u32::MAX);
        assert!(std::panic::catch_unwind(|| next_nonce(&c)).is_err());
        assert_eq!(
            c.load(Ordering::Relaxed),
            u32::MAX,
            "a failed draw must not advance the counter — otherwise the panic \
             just delays the collision by one pool"
        );
        assert!(std::panic::catch_unwind(|| next_nonce(&c)).is_err());
    }

    #[test]
    fn affine_discipline() {
        // This test verifies the design — ClauseToken is !Copy and !Clone.
        // The following would fail to compile if uncommented:
        //
        //   let token = pool.acquire(0).unwrap();
        //   let copy = token;       // moves token
        //   let _ = token.index();  // ERROR: use of moved value
        //
        // We can't write a compile-fail test in regular #[test], but the
        // type system enforces this statically.
        let mut pool = ClausePool::new(1);
        let token = pool.acquire(0).unwrap();
        // Move token into release — can't use it afterward
        pool.release(token);
        // token.index() would be a compile error here
    }
}
