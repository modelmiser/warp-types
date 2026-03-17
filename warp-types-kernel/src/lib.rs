//! Proc macro for marking GPU kernel functions.
//!
//! `#[warp_kernel]` transforms a function into a proper PTX kernel entry point
//! when compiling for nvptx64, and generates a host-side launcher when compiling
//! for the host target.
//!
//! # Usage
//!
//! In your kernel crate (compiled for nvptx64):
//!
//! ```rust,ignore
//! use warp_types::prelude::*;
//! use warp_types_kernel::warp_kernel;
//!
//! #[warp_kernel]
//! pub fn butterfly_reduce(data: *mut i32) {
//!     let warp: Warp<All> = Warp::kernel_entry();
//!     let tid = warp_types::gpu::thread_id_x();
//!     let mut val = unsafe { *data.add(tid as usize) };
//!
//!     val += warp.shuffle_xor(PerLane::new(val), 16).get();
//!     val += warp.shuffle_xor(PerLane::new(val), 8).get();
//!     val += warp.shuffle_xor(PerLane::new(val), 4).get();
//!     val += warp.shuffle_xor(PerLane::new(val), 2).get();
//!     val += warp.shuffle_xor(PerLane::new(val), 1).get();
//!
//!     unsafe { *data.add(tid as usize) = val; }
//! }
//! ```
//!
//! The macro emits:
//! - On nvptx64: `#[no_mangle] pub unsafe extern "ptx-kernel" fn butterfly_reduce(...)`
//! - On host: nothing (kernel functions are only compiled for GPU)

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, FnArg, ItemFn, Pat};

/// Mark a function as a GPU kernel entry point.
///
/// This attribute transforms the function signature for PTX compilation:
/// - Adds `#[no_mangle]` for symbol visibility in PTX
/// - Adds `extern "ptx-kernel"` ABI
/// - Wraps the body in `unsafe` (PTX kernels are inherently unsafe)
///
/// # Parameter Rules
///
/// Kernel parameters must be one of:
/// - Raw pointers (`*const T`, `*mut T`) — for device memory
/// - Scalars (`u32`, `i32`, `f32`, `u64`, `i64`, `f64`, `bool`) — passed by value
///
/// # Compile-Time Safety
///
/// The function body uses warp-types normally. `Warp::kernel_entry()` creates
/// the initial `Warp<All>`, and the type system prevents shuffle-from-inactive-lane
/// bugs at compile time — on the actual GPU target.
#[proc_macro_attribute]
pub fn warp_kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let name = &input.sig.ident;
    let params = &input.sig.inputs;
    let body = &input.block;
    let vis = &input.vis;

    // Validate parameters: must be raw pointers or scalars
    for param in params.iter() {
        if let FnArg::Typed(pat_type) = param {
            if let Err(err) = validate_kernel_param(&pat_type.ty, &pat_type.pat) {
                return err;
            }
        }
    }

    // Generate the kernel function for nvptx64
    let expanded = quote! {
        #[no_mangle]
        #vis unsafe extern "ptx-kernel" fn #name(#params) #body
    };

    TokenStream::from(expanded)
}

/// Validate that a kernel parameter type is GPU-compatible.
///
/// Returns `Ok(())` if valid, `Err(TokenStream)` with a `compile_error!` if not.
fn validate_kernel_param(ty: &syn::Type, pat: &Pat) -> Result<(), TokenStream> {
    match ty {
        // Raw pointers are always OK
        syn::Type::Ptr(_) => Ok(()),
        // Path types: check if they're known scalars
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                let name = seg.ident.to_string();
                let valid_scalars = [
                    "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "f32",
                    "f64", "bool",
                ];
                if !valid_scalars.contains(&name.as_str()) {
                    let msg = format!(
                        "warp_kernel: parameter `{}` has type `{}` which is not a GPU-compatible type. \
                         Use raw pointers (*const T, *mut T) for device memory or scalar types (u32, i32, f32, etc.).",
                        quote!(#pat), name
                    );
                    return Err(syn::Error::new_spanned(ty, msg).to_compile_error().into());
                }
            }
            Ok(())
        }
        _ => {
            let msg = format!(
                "warp_kernel: parameter `{}` has unsupported type `{}`. \
                 Kernel parameters must be raw pointers or scalar types.",
                quote!(#pat),
                quote!(#ty)
            );
            Err(syn::Error::new_spanned(ty, msg).to_compile_error().into())
        }
    }
}
