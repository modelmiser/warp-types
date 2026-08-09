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
//! The macro always emits `#[no_mangle] pub unsafe extern "ptx-kernel" fn ...`
//! regardless of target. Kernel crates should target nvptx64 exclusively —
//! the `extern "ptx-kernel"` ABI requires nightly `abi_ptx` and is only
//! meaningful on GPU targets.

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
/// - Thin raw pointers (`*const T`, `*mut T` with `T: Sized`) — for device
///   memory. Fat pointers (`*mut [T]`, `*const str`, `*const dyn Trait`) are
///   rejected: they are 16 bytes (data + length/vtable) where the kernel
///   parameter ABI expects 8, silently corrupting the parameter space.
/// - Scalars (`u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `bool`) — passed by value
///
/// Note: `usize`/`isize` are rejected because their width is platform-dependent.
/// On nvptx64 they are 64-bit, but the host launcher may assume a different size,
/// causing ABI mismatch. Use explicit-width types (`u32`, `u64`, etc.) instead.
///
/// # Compile-Time Safety
///
/// The function body uses warp-types normally. `Warp::kernel_entry()` creates
/// the initial `Warp<All>`, and the type system prevents shuffle-from-inactive-lane
/// bugs at compile time — on the actual GPU target.
#[proc_macro_attribute]
pub fn warp_kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    if let Err(err) = validate_kernel_signature(&input.sig) {
        return err.to_compile_error().into();
    }

    let name = &input.sig.ident;
    let params = &input.sig.inputs;
    let body = &input.block;
    let vis = &input.vis;
    let attrs = &input.attrs;

    // Generate the kernel function for nvptx64
    // Preserve outer attributes (doc comments, #[cfg], etc.)
    let expanded = quote! {
        #(#attrs)*
        #[no_mangle]
        #vis unsafe extern "ptx-kernel" fn #name(#params) #body
    };

    TokenStream::from(expanded)
}

/// Validate a full kernel signature: qualifiers, generics, return type, params.
///
/// Kept separate from the proc-macro entry point (and returning `syn::Error`
/// rather than `proc_macro::TokenStream`) so it can be unit-tested with
/// `syn::parse_str` outside a proc-macro invocation.
fn validate_kernel_signature(sig: &syn::Signature) -> Result<(), syn::Error> {
    // Qualifiers the macro would otherwise silently drop — reject instead.
    if let Some(constness) = &sig.constness {
        return Err(syn::Error::new_spanned(
            constness,
            "warp_kernel: GPU kernels cannot be `const fn`. \
             The macro emits an `extern \"ptx-kernel\"` entry point, which cannot be const.",
        ));
    }
    if let Some(asyncness) = &sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "warp_kernel: GPU kernels cannot be `async fn`. \
             PTX kernel entry points are synchronous; asyncness would be silently meaningless.",
        ));
    }
    if let Some(abi) = &sig.abi {
        return Err(syn::Error::new_spanned(
            abi,
            "warp_kernel: remove the explicit `extern` qualifier. \
             The macro sets the ABI to `extern \"ptx-kernel\"` itself; \
             a conflicting ABI here would be silently discarded.",
        ));
    }

    // PTX kernels must be void. `-> ()` is explicitly-written unit — allowed.
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        let is_unit = matches!(&**ty, syn::Type::Tuple(t) if t.elems.is_empty());
        if !is_unit {
            return Err(syn::Error::new_spanned(
                ty,
                "warp_kernel: GPU kernels must return `()`. \
                 PTX kernel entry points are always void.",
            ));
        }
    }

    // PTX kernels cannot be generic
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "warp_kernel: GPU kernels cannot be generic. \
             PTX entry points require concrete types.",
        ));
    }

    // Validate parameters: must be raw thin pointers or scalars
    for param in sig.inputs.iter() {
        match param {
            FnArg::Receiver(recv) => {
                return Err(syn::Error::new_spanned(
                    recv,
                    "warp_kernel: GPU kernels cannot take `self`. \
                     Kernel entry points are free functions, not methods.",
                ));
            }
            FnArg::Typed(pat_type) => {
                validate_kernel_param(&pat_type.ty, &pat_type.pat)?;
            }
        }
    }

    Ok(())
}

/// Is `ty` a pointee that makes a raw pointer to it *statically thin* (8 bytes)?
///
/// Pointers to slices, `str`, and trait objects are fat (16 bytes: pointer +
/// length/vtable). Passing one as a kernel parameter silently corrupts the
/// kernel parameter space: the host launcher and the PTX side disagree on the
/// parameter's size. Conservatively, only path types (excluding `str`),
/// arrays, and pointers-to-thin-pointees are accepted.
fn is_thin_pointee(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(tp) => {
            // `str` parses as a plain path type but is unsized → fat pointer.
            match tp.path.segments.last() {
                Some(seg) => seg.ident != "str",
                None => false,
            }
        }
        // Arrays [T; N] are sized → thin.
        syn::Type::Array(_) => true,
        // Pointer-to-pointer: thin iff the inner pointer is itself thin.
        syn::Type::Ptr(inner) => is_thin_pointee(&inner.elem),
        // Unwrap parentheses/groups.
        syn::Type::Paren(p) => is_thin_pointee(&p.elem),
        syn::Type::Group(g) => is_thin_pointee(&g.elem),
        // Slices, trait objects, impl Trait, bare fn, tuples, references,
        // macros, infer, never, ... — reject conservatively.
        _ => false,
    }
}

/// Validate that a kernel parameter type is GPU-compatible.
fn validate_kernel_param(ty: &syn::Type, pat: &Pat) -> Result<(), syn::Error> {
    match ty {
        // Raw pointers are OK only if statically thin (8 bytes).
        // Fat pointers (*mut [T], *const str, *const dyn Trait) are 16 bytes
        // on the host side — an ABI mismatch that silently corrupts the
        // kernel parameter space.
        syn::Type::Ptr(ptr) => {
            if !is_thin_pointee(&ptr.elem) {
                let msg = format!(
                    "warp_kernel: parameter `{}` has type `{}`, a pointer to an \
                     unsized (or non-thin) pointee. This is a fat pointer — 16 bytes \
                     (data + length/vtable) instead of the 8-byte thin pointer the \
                     kernel parameter ABI expects — which silently corrupts the \
                     kernel parameter space. Pass a thin pointer (`*const T`/`*mut T` \
                     with `T: Sized`) plus an explicit length parameter instead.",
                    quote!(#pat),
                    quote!(#ty)
                );
                return Err(syn::Error::new_spanned(ty, msg));
            }
            Ok(())
        }
        // Path types: check if they're known scalars
        syn::Type::Path(tp) => {
            // Reject qualified paths (e.g., my_crate::u32) — kernel params must be plain scalars
            if tp.path.segments.len() > 1 {
                let msg = format!(
                    "warp_kernel: parameter `{}` uses qualified type `{}`. \
                     Use unqualified scalar types (u32, i32, f32, etc.) for kernel parameters.",
                    quote!(#pat),
                    quote!(#ty)
                );
                return Err(syn::Error::new_spanned(ty, msg));
            }
            if let Some(seg) = tp.path.segments.last() {
                let name = seg.ident.to_string();
                let valid_scalars = [
                    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "bool",
                ];
                if !valid_scalars.contains(&name.as_str()) {
                    let msg = format!(
                        "warp_kernel: parameter `{}` has type `{}` which is not a GPU-compatible type. \
                         Use raw pointers (*const T, *mut T) for device memory or scalar types (u32, i32, f32, etc.).",
                        quote!(#pat), name
                    );
                    return Err(syn::Error::new_spanned(ty, msg));
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
            Err(syn::Error::new_spanned(ty, msg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_fn(src: &str) -> Result<(), syn::Error> {
        let item: syn::ItemFn = syn::parse_str(src).expect("test source must parse");
        validate_kernel_signature(&item.sig)
    }

    fn err_msg(src: &str) -> String {
        validate_fn(src)
            .expect_err("expected validation error")
            .to_string()
    }

    // ── Accepted signatures ──

    #[test]
    fn accepts_thin_pointers_and_scalars() {
        validate_fn("fn k(data: *mut i32, n: u32, s: f64, flag: bool) {}").unwrap();
    }

    #[test]
    fn accepts_pointer_to_pointer_and_array() {
        validate_fn("fn k(pp: *mut *mut i32, arr: *const [i32; 4]) {}").unwrap();
    }

    #[test]
    fn accepts_explicit_unit_return() {
        validate_fn("fn k(data: *mut i32) -> () {}").unwrap();
    }

    // ── Fat pointers rejected ──

    #[test]
    fn rejects_slice_pointer() {
        let msg = err_msg("fn k(data: *mut [i32]) {}");
        assert!(
            msg.contains("fat pointer"),
            "message must name the fat-pointer ABI issue: {msg}"
        );
    }

    #[test]
    fn rejects_str_pointer() {
        let msg = err_msg("fn k(s: *const str) {}");
        assert!(msg.contains("fat pointer"), "unexpected message: {msg}");
    }

    #[test]
    fn rejects_trait_object_pointer() {
        let msg = err_msg("fn k(obj: *const dyn core::fmt::Debug) {}");
        assert!(msg.contains("fat pointer"), "unexpected message: {msg}");
    }

    #[test]
    fn rejects_pointer_to_fat_pointer_conservatively() {
        // `*mut *mut [i32]` — the outer pointer is technically thin (its
        // pointee, a fat pointer, is Sized), but a fat-pointer pointee is not
        // in the conservative accept list (path/array/thin-pointer), so it
        // is rejected rather than risking a host/device layout mismatch for
        // the pointed-to fat pointer.
        let msg = err_msg("fn k(pp: *mut *mut [i32]) {}");
        assert!(msg.contains("fat pointer"), "unexpected message: {msg}");
    }

    // ── Return types ──

    #[test]
    fn rejects_non_unit_return() {
        let msg = err_msg("fn k(data: *mut i32) -> i32 { 0 }");
        assert!(
            msg.contains("must return `()`"),
            "unexpected message: {msg}"
        );
    }

    // ── Receivers ──

    #[test]
    fn rejects_self_receiver() {
        let msg = err_msg("fn k(&self, data: *mut i32) {}");
        assert!(
            msg.contains("cannot take `self`"),
            "unexpected message: {msg}"
        );
    }

    // ── Qualifiers ──

    #[test]
    fn rejects_const_fn() {
        let msg = err_msg("const fn k(data: *mut i32) {}");
        assert!(
            msg.contains("cannot be `const fn`"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn rejects_async_fn() {
        let msg = err_msg("async fn k(data: *mut i32) {}");
        assert!(
            msg.contains("cannot be `async fn`"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn rejects_explicit_extern_abi() {
        let msg = err_msg(r#"extern "C" fn k(data: *mut i32) {}"#);
        assert!(
            msg.contains("remove the explicit `extern`"),
            "unexpected message: {msg}"
        );
    }

    // ── Existing rules still enforced ──

    #[test]
    fn rejects_usize() {
        let msg = err_msg("fn k(n: usize) {}");
        assert!(
            msg.contains("not a GPU-compatible type"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn rejects_generics() {
        let msg = err_msg("fn k<T>(data: *mut i32) {}");
        assert!(
            msg.contains("cannot be generic"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn rejects_reference_param() {
        let msg = err_msg("fn k(data: &mut i32) {}");
        assert!(
            msg.contains("unsupported type"),
            "unexpected message: {msg}"
        );
    }
}
