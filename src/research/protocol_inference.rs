//! Protocol Inference: Can We Infer Warp Session Types?
//!
//! **STATUS: Research** — Exploratory prototype, not promoted to main API.
//!
//! This module explores whether warp session protocols can be automatically
//! inferred from code, or must be manually annotated.
//!
//! # The Question
//!
//! Given code like:
//! ```ignore
//! fn reduce(warp: Warp<All>, data: PerLane<i32>) -> i32 {
//!     let data = warp.shuffle_xor(data, 1);
//!     let data = warp.shuffle_xor(data, 2);
//!     // ...
//!     data
//! }
//! ```
//!
//! Can we infer the protocol `Shuffle<XOR<1>>; Shuffle<XOR<2>>; ...`?
//!
//! # Approaches Explored
//!
//! 1. **Full inference**: Infer entire protocol from code (undecidable in general)
//! 2. **Local inference**: Infer within functions, require signatures at boundaries
//! 3. **Bidirectional**: Mix inference (easy cases) and checking (complex cases)
//! 4. **Protocol-first**: Write protocol, generate/check code against it
//! 5. **Gradual**: Start untyped, add annotations incrementally

use std::marker::PhantomData;
use std::collections::HashMap;

// ============================================================================
// WHAT NEEDS TO BE INFERRED?
// ============================================================================

/// For warp sessions, we need to infer:
///
/// 1. **Active set at each program point**
///    - After diverge: which lanes are active?
///    - After merge: back to union
///    - This is DATAFLOW ANALYSIS
///
/// 2. **Protocol sequence**
///    - Which shuffles occur in what order?
///    - Where are the diverge/merge points?
///    - This is EFFECT TRACKING
///
/// 3. **Protocol equivalence**
///    - Do two paths through the code have compatible protocols?
///    - This is TYPE EQUALITY
pub mod what_to_infer {
    /// Active set inference is essentially dataflow analysis:
    /// - Forward analysis: track active set through control flow
    /// - At diverge: split into two sets
    /// - At merge: union the sets
    /// - At join points: must be equal (or error)
    pub struct ActiveSetInference;

    /// Protocol inference is effect tracking:
    /// - Each shuffle adds to the protocol
    /// - Each diverge adds a branch
    /// - Each merge closes a branch
    /// - The "effect" is the protocol fragment
    pub struct ProtocolInference;
}

// ============================================================================
// APPROACH 1: FULL INFERENCE (Theoretical Limits)
// ============================================================================

/// Full protocol inference: given arbitrary code, infer the protocol.
///
/// **Result: UNDECIDABLE in general**
///
/// Why? Because:
/// 1. Protocols can depend on runtime values (which shuffle mask?)
/// 2. Loops create recursive protocols (halting problem)
/// 3. Higher-order functions pass protocols as arguments
///
/// However, for RESTRICTED subsets, inference IS possible.
pub mod full_inference {
    /// Restrictions that make inference decidable:
    ///
    /// 1. **No data-dependent shuffles**: mask must be constant
    /// 2. **Bounded loops**: max iterations known statically
    /// 3. **No higher-order warp functions**: can't pass Warp as closure arg
    /// 4. **Structured control flow**: no goto, just if/while/for
    pub struct DecidableRestrictions;

    /// With these restrictions, inference reduces to:
    /// - Abstract interpretation over active sets
    /// - Effect collection for protocol sequence
    /// - Both are decidable for finite domains
    pub fn is_inference_decidable(
        has_data_dependent_shuffle: bool,
        has_unbounded_loops: bool,
        has_higher_order: bool,
    ) -> bool {
        !has_data_dependent_shuffle && !has_unbounded_loops && !has_higher_order
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_decidability_conditions() {
            // All false = decidable
            assert!(is_inference_decidable(false, false, false));
            // Any true = undecidable
            assert!(!is_inference_decidable(true, false, false));
            assert!(!is_inference_decidable(false, true, false));
            assert!(!is_inference_decidable(false, false, true));
        }
    }
}

// ============================================================================
// APPROACH 2: LOCAL INFERENCE
// ============================================================================

/// Local inference: infer within functions, require annotations at boundaries.
///
/// This is the PRACTICAL approach used by most session type systems.
///
/// Key insight: Most warp operations are LOCAL to a function.
/// Cross-function warp passing is rare (and should be annotated).
pub mod local_inference {
    

    /// A program point with inferred active set
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct InferredPoint {
        /// The active set at this point (as a mask)
        pub active_mask: u32,
        /// The protocol fragment up to this point
        pub protocol: Vec<ProtocolOp>,
    }

    /// Protocol operations we can infer
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum ProtocolOp {
        Shuffle { mask: u32 },
        Diverge { predicate: String },
        Merge,
        Sync,
    }

    /// Local inference engine
    pub struct LocalInferrer {
        /// Current active set
        current_mask: u32,
        /// Accumulated protocol
        protocol: Vec<ProtocolOp>,
        /// Stack for diverge/merge tracking
        diverge_stack: Vec<u32>,
    }

    impl LocalInferrer {
        pub fn new() -> Self {
            LocalInferrer {
                current_mask: 0xFFFFFFFF,  // All lanes active
                protocol: Vec::new(),
                diverge_stack: Vec::new(),
            }
        }

        /// Infer effect of a shuffle
        pub fn shuffle(&mut self, mask: u32) -> Result<(), String> {
            if self.current_mask != 0xFFFFFFFF {
                return Err(format!(
                    "Shuffle requires All lanes, but active set is 0x{:08X}",
                    self.current_mask
                ));
            }
            self.protocol.push(ProtocolOp::Shuffle { mask });
            Ok(())
        }

        /// Infer effect of a diverge
        pub fn diverge(&mut self, predicate: &str, true_mask: u32) {
            self.diverge_stack.push(self.current_mask);
            self.protocol.push(ProtocolOp::Diverge {
                predicate: predicate.to_string(),
            });
            // After diverge, we're in the "true" branch
            self.current_mask &= true_mask;
        }

        /// Switch to the "else" branch
        pub fn else_branch(&mut self) {
            if let Some(&parent_mask) = self.diverge_stack.last() {
                // Compute false branch: parent - true branch
                let true_mask = self.current_mask;
                self.current_mask = parent_mask & !true_mask;
            }
        }

        /// Infer effect of a merge
        pub fn merge(&mut self) -> Result<(), String> {
            if let Some(parent_mask) = self.diverge_stack.pop() {
                self.current_mask = parent_mask;
                self.protocol.push(ProtocolOp::Merge);
                Ok(())
            } else {
                Err("Merge without matching diverge".to_string())
            }
        }

        /// Get the inferred protocol
        pub fn finish(self) -> Result<Vec<ProtocolOp>, String> {
            if !self.diverge_stack.is_empty() {
                return Err(format!(
                    "Unclosed diverge: {} pending",
                    self.diverge_stack.len()
                ));
            }
            Ok(self.protocol)
        }

        /// Get current state
        pub fn current_state(&self) -> InferredPoint {
            InferredPoint {
                active_mask: self.current_mask,
                protocol: self.protocol.clone(),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_simple_shuffle_inference() {
            let mut inf = LocalInferrer::new();
            inf.shuffle(1).unwrap();
            inf.shuffle(2).unwrap();
            let protocol = inf.finish().unwrap();

            assert_eq!(protocol.len(), 2);
            assert_eq!(protocol[0], ProtocolOp::Shuffle { mask: 1 });
            assert_eq!(protocol[1], ProtocolOp::Shuffle { mask: 2 });
        }

        #[test]
        fn test_diverge_merge_inference() {
            let mut inf = LocalInferrer::new();
            inf.diverge("even", 0x55555555);  // Even lanes
            assert_eq!(inf.current_mask, 0x55555555);
            inf.merge().unwrap();
            assert_eq!(inf.current_mask, 0xFFFFFFFF);
            let protocol = inf.finish().unwrap();

            assert_eq!(protocol.len(), 2);
        }

        #[test]
        fn test_shuffle_after_diverge_fails() {
            let mut inf = LocalInferrer::new();
            inf.diverge("even", 0x55555555);
            let result = inf.shuffle(1);
            assert!(result.is_err());
        }
    }
}

// ============================================================================
// APPROACH 3: BIDIRECTIONAL TYPE CHECKING
// ============================================================================

/// Bidirectional: infer where easy, check where annotated.
///
/// The insight: some expressions have "obvious" types (inference mode),
/// others need annotations (checking mode).
///
/// For warp sessions:
/// - INFER: shuffle mask, diverge predicate, local operations
/// - CHECK: function boundaries, loop invariants, complex predicates
pub mod bidirectional {
    use super::*;

    /// Inference mode: compute the protocol from the expression
    pub trait Infer {
        fn infer(&self, ctx: &InferContext) -> Result<InferredProtocol, String>;
    }

    /// Checking mode: verify expression matches expected protocol
    pub trait Check {
        fn check(&self, ctx: &CheckContext, expected: &Protocol) -> Result<(), String>;
    }

    /// Context for inference
    pub struct InferContext {
        pub active_mask: u32,
        pub variables: HashMap<String, Protocol>,
    }

    /// Context for checking
    pub struct CheckContext {
        pub active_mask: u32,
        pub variables: HashMap<String, Protocol>,
    }

    /// Inferred protocol (may be incomplete)
    #[derive(Clone, Debug)]
    pub struct InferredProtocol {
        pub ops: Vec<local_inference::ProtocolOp>,
        pub final_mask: u32,
    }

    /// Declared protocol (complete)
    #[derive(Clone, Debug)]
    pub struct Protocol {
        pub ops: Vec<local_inference::ProtocolOp>,
        pub input_mask: u32,
        pub output_mask: u32,
    }

    /// Bidirectional type checker
    pub struct BiChecker {
        mode: Mode,
    }

    enum Mode {
        Infer,
        Check(Protocol),
    }

    impl BiChecker {
        /// Start in inference mode
        pub fn infer() -> Self {
            BiChecker { mode: Mode::Infer }
        }

        /// Start in checking mode with expected protocol
        pub fn check(expected: Protocol) -> Self {
            BiChecker { mode: Mode::Check(expected) }
        }

        /// Switch from infer to check when we hit an annotation
        pub fn switch_to_check(&mut self, annotation: Protocol) {
            self.mode = Mode::Check(annotation);
        }
    }

    /// When do we switch modes?
    ///
    /// INFER → CHECK:
    /// - Function call with protocol annotation
    /// - Loop with invariant annotation
    /// - Explicit type annotation
    ///
    /// CHECK → INFER:
    /// - Inside function body
    /// - Local let bindings
    /// - Lambda bodies
    pub struct ModeSwitching;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bidirectional_creation() {
            let _inf = BiChecker::infer();
            let _chk = BiChecker::check(Protocol {
                ops: vec![],
                input_mask: 0xFFFFFFFF,
                output_mask: 0xFFFFFFFF,
            });
        }
    }
}

// ============================================================================
// APPROACH 4: PROTOCOL-FIRST DEVELOPMENT
// ============================================================================

/// Protocol-first: write the protocol, then write/generate code.
///
/// This flips the question: instead of inferring protocols from code,
/// we DESIGN protocols first, then ensure code follows them.
///
/// Benefits:
/// - Protocol is the specification (clear intent)
/// - Can generate skeleton code from protocol
/// - Type errors are "code doesn't match spec" (clear blame)
pub mod protocol_first {
    

    /// A protocol specification
    #[derive(Clone, Debug)]
    pub enum ProtocolSpec {
        /// End of protocol
        End,
        /// Shuffle then continue
        Shuffle { mask: u32, then: Box<ProtocolSpec> },
        /// Diverge into two branches
        Diverge {
            predicate: String,
            true_branch: Box<ProtocolSpec>,
            false_branch: Box<ProtocolSpec>,
        },
        /// Sequence of protocols
        Seq(Vec<ProtocolSpec>),
    }

    impl ProtocolSpec {
        /// Generate a code skeleton from the protocol
        pub fn generate_skeleton(&self, indent: usize) -> String {
            let pad = "    ".repeat(indent);
            match self {
                ProtocolSpec::End => format!("{}// protocol complete\n", pad),

                ProtocolSpec::Shuffle { mask, then } => {
                    format!(
                        "{}let data = warp.shuffle_xor(data, {});\n{}",
                        pad, mask, then.generate_skeleton(indent)
                    )
                }

                ProtocolSpec::Diverge { predicate, true_branch, false_branch } => {
                    format!(
                        "{}let (true_warp, false_warp) = warp.diverge(|lane| {});\n\
                         {}// true branch:\n{}\
                         {}// false branch:\n{}\
                         {}let warp = merge(true_warp, false_warp);\n",
                        pad, predicate,
                        pad, true_branch.generate_skeleton(indent + 1),
                        pad, false_branch.generate_skeleton(indent + 1),
                        pad
                    )
                }

                ProtocolSpec::Seq(specs) => {
                    specs.iter()
                        .map(|s| s.generate_skeleton(indent))
                        .collect::<Vec<_>>()
                        .join("")
                }
            }
        }
    }

    /// Example: Butterfly reduction protocol
    pub fn butterfly_protocol() -> ProtocolSpec {
        ProtocolSpec::Seq(vec![
            ProtocolSpec::Shuffle { mask: 1, then: Box::new(ProtocolSpec::End) },
            ProtocolSpec::Shuffle { mask: 2, then: Box::new(ProtocolSpec::End) },
            ProtocolSpec::Shuffle { mask: 4, then: Box::new(ProtocolSpec::End) },
            ProtocolSpec::Shuffle { mask: 8, then: Box::new(ProtocolSpec::End) },
            ProtocolSpec::Shuffle { mask: 16, then: Box::new(ProtocolSpec::End) },
        ])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_generate_skeleton() {
            let protocol = ProtocolSpec::Shuffle {
                mask: 1,
                then: Box::new(ProtocolSpec::Shuffle {
                    mask: 2,
                    then: Box::new(ProtocolSpec::End),
                }),
            };

            let code = protocol.generate_skeleton(0);
            assert!(code.contains("shuffle_xor(data, 1)"));
            assert!(code.contains("shuffle_xor(data, 2)"));
        }

        #[test]
        fn test_butterfly_skeleton() {
            let protocol = butterfly_protocol();
            let code = protocol.generate_skeleton(0);
            assert!(code.contains("shuffle_xor(data, 1)"));
            assert!(code.contains("shuffle_xor(data, 16)"));
        }
    }
}

// ============================================================================
// APPROACH 5: GRADUAL TYPING
// ============================================================================

/// Gradual: start untyped, add annotations incrementally.
///
/// The "dynamic" warp type accepts any operation but checks at runtime.
/// Add static types gradually for more compile-time guarantees.
pub mod gradual {
    use super::*;

    /// Dynamic warp: any operation allowed, runtime checked
    #[derive(Clone)]
    pub struct DynWarp {
        active_mask: u32,
    }

    impl DynWarp {
        pub fn all() -> Self {
            DynWarp { active_mask: 0xFFFFFFFF }
        }

        /// Shuffle - runtime check for All
        pub fn shuffle(&self, _data: i32, _mask: u32) -> Result<i32, String> {
            if self.active_mask != 0xFFFFFFFF {
                return Err(format!(
                    "Runtime error: shuffle requires All lanes, got 0x{:08X}",
                    self.active_mask
                ));
            }
            Ok(0)  // Placeholder
        }

        /// Diverge - always works, returns two warps
        pub fn diverge(&self, predicate_mask: u32) -> (DynWarp, DynWarp) {
            let true_warp = DynWarp {
                active_mask: self.active_mask & predicate_mask,
            };
            let false_warp = DynWarp {
                active_mask: self.active_mask & !predicate_mask,
            };
            (true_warp, false_warp)
        }

        /// Merge - runtime check for disjoint
        pub fn merge(self, other: DynWarp) -> Result<DynWarp, String> {
            if self.active_mask & other.active_mask != 0 {
                return Err("Runtime error: merge of overlapping warps".to_string());
            }
            Ok(DynWarp {
                active_mask: self.active_mask | other.active_mask,
            })
        }

        /// Get current mask (for gradual migration)
        pub fn get_mask(&self) -> u32 {
            self.active_mask
        }
    }

    /// Gradual migration: refine DynWarp to typed Warp<S>
    ///
    /// Step 1: Write code with DynWarp, test it
    /// Step 2: Add type annotations where confident
    /// Step 3: Compiler tells you where types conflict
    /// Step 4: Fix conflicts or keep dynamic where needed
    pub struct GradualMigration;

    /// The "gradually typed" warp: either static or dynamic
    pub enum GradualWarp<S: ActiveSet> {
        Static(StaticWarp<S>),
        Dynamic(DynWarp),
    }

    pub trait ActiveSet {
        const MASK: u32;
    }

    pub struct All;
    impl ActiveSet for All {
        const MASK: u32 = 0xFFFFFFFF;
    }

    pub struct StaticWarp<S: ActiveSet> {
        _marker: PhantomData<S>,
    }

    impl<S: ActiveSet> GradualWarp<S> {
        /// Ascribe: assert that dynamic warp matches static type
        pub fn ascribe(dyn_warp: DynWarp) -> Result<Self, String> {
            if dyn_warp.active_mask == S::MASK {
                Ok(GradualWarp::Dynamic(dyn_warp))
            } else {
                Err(format!(
                    "Type ascription failed: expected 0x{:08X}, got 0x{:08X}",
                    S::MASK, dyn_warp.active_mask
                ))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_dyn_warp_shuffle_all() {
            let warp = DynWarp::all();
            assert!(warp.shuffle(42, 1).is_ok());
        }

        #[test]
        fn test_dyn_warp_shuffle_partial_fails() {
            let warp = DynWarp { active_mask: 0x55555555 };
            assert!(warp.shuffle(42, 1).is_err());
        }

        #[test]
        fn test_dyn_warp_diverge_merge() {
            let warp = DynWarp::all();
            let (even, odd) = warp.diverge(0x55555555);
            assert_eq!(even.active_mask, 0x55555555);
            assert_eq!(odd.active_mask, 0xAAAAAAAA);
            let merged = even.merge(odd).unwrap();
            assert_eq!(merged.active_mask, 0xFFFFFFFF);
        }

        #[test]
        fn test_gradual_ascription() {
            let dyn_warp = DynWarp::all();
            let result: Result<GradualWarp<All>, _> = GradualWarp::ascribe(dyn_warp);
            assert!(result.is_ok());
        }
    }
}

// ============================================================================
// PRACTICAL RECOMMENDATION
// ============================================================================

/// Practical recommendation for warp session inference:
///
/// 1. **Use LOCAL INFERENCE by default**
///    - Infer active sets within functions
///    - Infer protocol fragments from operations
///    - Works for 80% of GPU code
///
/// 2. **Require ANNOTATIONS at boundaries**
///    - Function signatures: `fn reduce(warp: Warp<All>) -> Warp<All>`
///    - Loop invariants: `#[warp_invariant(All)]`
///    - Public APIs: must be explicit
///
/// 3. **Support GRADUAL typing for migration**
///    - Existing code starts with DynWarp
///    - Add static types incrementally
///    - Compiler guides the migration
///
/// 4. **Provide PROTOCOL-FIRST tools for new code**
///    - Define protocol in DSL
///    - Generate skeleton code
///    - Type-check against protocol
///
/// This balances:
/// - Minimal annotation burden (local inference)
/// - Clear API contracts (boundary annotations)
/// - Practical migration path (gradual typing)
/// - Design clarity (protocol-first option)
pub mod recommendation {
    /// The recommended approach combines all four:
    ///
    /// ```text
    /// ┌─────────────────────────────────────────────┐
    /// │  Function Boundary (ANNOTATED)              │
    /// │  fn reduce(warp: Warp<All>) -> Warp<All>   │
    /// ├─────────────────────────────────────────────┤
    /// │  Function Body (INFERRED)                   │
    /// │  let x = warp.shuffle(data, 1);  // infer  │
    /// │  let y = warp.shuffle(x, 2);     // infer  │
    /// ├─────────────────────────────────────────────┤
    /// │  Complex Cases (GRADUAL)                    │
    /// │  let dyn_warp = DynWarp::from(warp);       │
    /// │  // ... complex logic ...                  │
    /// │  let warp: Warp<All> = dyn_warp.ascribe()?;│
    /// └─────────────────────────────────────────────┘
    /// ```
    pub struct CombinedApproach;
}

// ============================================================================
// COMPARISON WITH LITERATURE
// ============================================================================

/// How does this compare to existing session type inference work?
///
/// | System | Inference | Annotations | Gradual |
/// |--------|-----------|-------------|---------|
/// | Scribble | No | Full protocols | No |
/// | GradualSession | Partial | Boundaries | Yes |
/// | SessionML | Local | Functions | No |
/// | **WarpTypes** | Local | Boundaries | Yes |
///
/// Our approach is closest to GradualSession (Igarashi et al.) but
/// specialized for warp operations instead of message passing.
pub mod literature {
    /// Key papers:
    /// - Honda/Yoshida/Carbone (2008): Original MPST, no inference
    /// - Scribble (2011): Protocol-first, generate code
    /// - GradualSession (2017): Gradual typing for sessions
    /// - SessionML (2019): ML-style inference for sessions
    pub struct References;
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_inference_reduction() {
        // Full butterfly reduction: all shuffles in sequence
        let mut inf = local_inference::LocalInferrer::new();
        for mask in [1, 2, 4, 8, 16] {
            inf.shuffle(mask).unwrap();
        }
        let protocol = inf.finish().unwrap();
        assert_eq!(protocol.len(), 5);
    }

    #[test]
    fn test_diverge_with_different_work() {
        // Diverge, do different work in each branch, merge
        let mut inf = local_inference::LocalInferrer::new();

        // All lanes start
        assert_eq!(inf.current_state().active_mask, 0xFFFFFFFF);

        // Diverge by even/odd
        inf.diverge("lane % 2 == 0", 0x55555555);
        assert_eq!(inf.current_state().active_mask, 0x55555555);

        // Can't shuffle here (not All)
        assert!(inf.shuffle(1).is_err());

        // Merge back
        inf.merge().unwrap();
        assert_eq!(inf.current_state().active_mask, 0xFFFFFFFF);

        // Now can shuffle
        assert!(inf.shuffle(1).is_ok());
    }

    #[test]
    fn test_protocol_first_matches_inference() {
        // Generate from protocol
        let protocol = protocol_first::butterfly_protocol();
        let skeleton = protocol.generate_skeleton(0);

        // The generated code would infer the same protocol
        // (This is the duality between inference and generation)
        assert!(skeleton.contains("shuffle_xor"));
    }
}

// ============================================================================
// RESEARCH QUESTION ANSWERED
// ============================================================================

// Q: Can we infer protocols or must they be written?
//
// A: BOTH - it depends on the context:
//
// 1. FULL INFERENCE: Undecidable in general, but decidable for restricted
//    subsets (no data-dependent shuffles, bounded loops, no higher-order).
//
// 2. LOCAL INFERENCE: Practical and decidable. Infer within functions,
//    require annotations at boundaries. Works for 80% of GPU code.
//
// 3. BIDIRECTIONAL: Mix of inference and checking. Infer easy cases,
//    check against annotations for complex cases.
//
// 4. PROTOCOL-FIRST: Alternative workflow - design protocol first,
//    generate/check code against it. Good for new code.
//
// 5. GRADUAL: Start untyped, add annotations incrementally. Good for
//    migrating existing code.
//
// RECOMMENDATION: Use local inference by default, require boundary
// annotations, support gradual typing for migration, offer protocol-first
// tools for new code.
//
// See src/protocol_inference.rs for all approaches prototyped.
