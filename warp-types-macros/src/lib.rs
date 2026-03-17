//! Proc macro for generating active set type hierarchies.
//!
//! Replaces ~180 lines of boilerplate with a compact DSL:
//! - Validates disjoint + covering masks at compile time
//! - Generates structs, ActiveSet impls, complement impls, diverge impls
//! - Deduplicates shared types (e.g., EvenLow under both Even and LowHalf)

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::{braced, Ident, LitInt, Result, Token};

// ============================================================================
// DSL Parsing
// ============================================================================

/// A single set definition: `Name = 0xMASK` or just `Name` (reference)
struct SetDef {
    name: Ident,
    mask: Option<u64>,
}

impl Parse for SetDef {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let mask = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            let lit: LitInt = input.parse()?;
            Some(lit.base10_parse::<u64>()?)
        } else {
            None
        };
        Ok(SetDef { name, mask })
    }
}

/// A diverge pair: `TrueBranch / FalseBranch`
struct DivergePair {
    true_branch: SetDef,
    false_branch: SetDef,
}

impl Parse for DivergePair {
    fn parse(input: ParseStream) -> Result<Self> {
        let true_branch: SetDef = input.parse()?;
        input.parse::<Token![/]>()?;
        let false_branch: SetDef = input.parse()?;
        Ok(DivergePair {
            true_branch,
            false_branch,
        })
    }
}

/// A parent block: `Parent = 0xMASK { pair, pair, ... }` or `Parent { pair, ... }`
struct ParentBlock {
    name: Ident,
    mask: Option<u64>,
    pairs: Vec<DivergePair>,
}

impl Parse for ParentBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let mask = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            let lit: LitInt = input.parse()?;
            Some(lit.base10_parse::<u64>()?)
        } else {
            None
        };
        let content;
        braced!(content in input);
        let mut pairs = Vec::new();
        while !content.is_empty() {
            pairs.push(content.parse()?);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        Ok(ParentBlock { name, mask, pairs })
    }
}

/// The full `warp_sets!` input
struct WarpSetsInput {
    blocks: Vec<ParentBlock>,
}

impl Parse for WarpSetsInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut blocks = Vec::new();
        while !input.is_empty() {
            blocks.push(input.parse()?);
        }
        Ok(WarpSetsInput { blocks })
    }
}

// ============================================================================
// Code Generation
// ============================================================================

#[proc_macro]
pub fn warp_sets(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as WarpSetsInput);

    // Phase 1: Collect all set names → masks
    let mut masks: HashMap<String, u64> = HashMap::new();
    let mut emitted: HashSet<String> = HashSet::new();
    let mut output = TokenStream2::new();

    // Collect masks from all blocks, checking for conflicts
    let insert_mask = |name: &Ident, m: u64, masks: &mut HashMap<String, u64>| -> std::result::Result<(), TokenStream> {
        let key = name.to_string();
        if let Some(&existing) = masks.get(&key) {
            if existing != m {
                let msg = format!(
                    "{} defined with conflicting masks: 0x{:08X} vs 0x{:08X}",
                    key, existing, m,
                );
                return Err(syn::Error::new_spanned(name, msg)
                    .to_compile_error()
                    .into());
            }
        }
        masks.insert(key, m);
        Ok(())
    };
    for block in &input.blocks {
        if let Some(m) = block.mask {
            if let Err(e) = insert_mask(&block.name, m, &mut masks) {
                return e;
            }
        }
        for pair in &block.pairs {
            if let Some(m) = pair.true_branch.mask {
                if let Err(e) = insert_mask(&pair.true_branch.name, m, &mut masks) {
                    return e;
                }
            }
            if let Some(m) = pair.false_branch.mask {
                if let Err(e) = insert_mask(&pair.false_branch.name, m, &mut masks) {
                    return e;
                }
            }
        }
    }

    // Phase 2: Generate code for each block
    for block in &input.blocks {
        let parent_name = &block.name;
        let parent_str = parent_name.to_string();
        let parent_mask = match masks.get(&parent_str) {
            Some(m) => *m,
            None => {
                return syn::Error::new_spanned(parent_name, "unknown mask for parent")
                    .to_compile_error()
                    .into();
            }
        };

        // Emit parent struct + Sealed + ActiveSet impl (if not already emitted)
        if emitted.insert(parent_str.clone()) {
            let mask_lit = parent_mask;
            let name_str = &parent_str;
            output.extend(quote! {
                #[derive(Copy, Clone, Debug, Default)]
                pub struct #parent_name;
                #[allow(private_interfaces)]
                impl sealed::Sealed for #parent_name {
                    fn _sealed() -> sealed::SealToken { sealed::SealToken }
                }
                impl ActiveSet for #parent_name {
                    const MASK: u64 = #mask_lit;
                    const NAME: &'static str = #name_str;
                }
            });
        }

        for pair in &block.pairs {
            let true_name = &pair.true_branch.name;
            let false_name = &pair.false_branch.name;
            let true_str = true_name.to_string();
            let false_str = false_name.to_string();

            let true_mask = match masks.get(&true_str) {
                Some(m) => *m,
                None => {
                    return syn::Error::new_spanned(true_name, "unknown mask")
                        .to_compile_error()
                        .into();
                }
            };
            let false_mask = match masks.get(&false_str) {
                Some(m) => *m,
                None => {
                    return syn::Error::new_spanned(false_name, "unknown mask")
                        .to_compile_error()
                        .into();
                }
            };

            // Validate: disjoint
            if true_mask & false_mask != 0 {
                let msg = format!(
                    "{} (0x{:08X}) and {} (0x{:08X}) overlap: 0x{:08X}",
                    true_str,
                    true_mask,
                    false_str,
                    false_mask,
                    true_mask & false_mask,
                );
                return syn::Error::new_spanned(true_name, msg)
                    .to_compile_error()
                    .into();
            }

            // Validate: covering
            if true_mask | false_mask != parent_mask {
                let msg = format!(
                    "{} | {} = 0x{:08X}, expected parent {} = 0x{:08X}",
                    true_str,
                    false_str,
                    true_mask | false_mask,
                    parent_str,
                    parent_mask,
                );
                return syn::Error::new_spanned(true_name, msg)
                    .to_compile_error()
                    .into();
            }

            // Validate: subset of parent
            if true_mask & !parent_mask != 0 {
                let msg = format!(
                    "{} (0x{:08X}) has bits outside parent {} (0x{:08X})",
                    true_str, true_mask, parent_str, parent_mask,
                );
                return syn::Error::new_spanned(true_name, msg)
                    .to_compile_error()
                    .into();
            }

            // Emit child structs (deduplicated)
            if emitted.insert(true_str.clone()) {
                let name_str_t = &true_str;
                output.extend(quote! {
                    #[derive(Copy, Clone, Debug, Default)]
                    pub struct #true_name;
                    #[allow(private_interfaces)]
                    impl sealed::Sealed for #true_name {
                        fn _sealed() -> sealed::SealToken { sealed::SealToken }
                    }
                    impl ActiveSet for #true_name {
                        const MASK: u64 = #true_mask;
                        const NAME: &'static str = #name_str_t;
                    }
                });
            }
            if emitted.insert(false_str.clone()) {
                let name_str_f = &false_str;
                output.extend(quote! {
                    #[derive(Copy, Clone, Debug, Default)]
                    pub struct #false_name;
                    #[allow(private_interfaces)]
                    impl sealed::Sealed for #false_name {
                        fn _sealed() -> sealed::SealToken { sealed::SealToken }
                    }
                    impl ActiveSet for #false_name {
                        const MASK: u64 = #false_mask;
                        const NAME: &'static str = #name_str_f;
                    }
                });
            }

            // Emit ComplementOf (symmetric)
            if parent_str == "All" {
                // Top-level: ComplementOf
                output.extend(quote! {
                    impl ComplementOf<#false_name> for #true_name {}
                    impl ComplementOf<#true_name> for #false_name {}
                });
            }

            // Emit ComplementWithin (always)
            output.extend(quote! {
                impl ComplementWithin<#false_name, #parent_name> for #true_name {}
                impl ComplementWithin<#true_name, #parent_name> for #false_name {}
            });

            // Emit CanDiverge
            output.extend(quote! {
                impl CanDiverge<#true_name, #false_name> for #parent_name {
                    fn diverge(
                        _warp: crate::warp::Warp<Self>,
                    ) -> (crate::warp::Warp<#true_name>, crate::warp::Warp<#false_name>) {
                        (crate::warp::Warp::new(), crate::warp::Warp::new())
                    }
                }
            });
        }
    }

    output.into()
}
