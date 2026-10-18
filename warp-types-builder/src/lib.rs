//! Build-time PTX compilation for warp-types GPU kernels.
//!
//! Use in your `build.rs` to cross-compile a kernel crate to PTX,
//! then load the generated PTX at runtime via cudarc.
//!
//! # Example
//!
//! ```rust,no_run
//! // build.rs
//! warp_types_builder::WarpBuilder::new("my-kernels")
//!     .build()
//!     .expect("Failed to compile GPU kernels");
//! ```
//!
//! Then in your main crate:
//!
//! ```rust,ignore
//! // src/main.rs
//! mod kernels {
//!     include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
//! }
//!
//! fn main() {
//!     let ctx = cudarc::driver::CudaContext::new(0).unwrap();
//!     let k = kernels::Kernels::load(&ctx).unwrap();
//!     // k.butterfly_reduce — CudaFunction handle ready for launch
//! }
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// GPU target for cross-compilation.
#[derive(Clone, Debug)]
pub enum GpuTarget {
    /// NVIDIA nvptx64 (32-lane warps, PTX output)
    Nvidia,
    /// AMD amdgcn (64-lane wavefronts, AMDGPU output).
    ///
    /// **Not yet supported:** [`WarpBuilder::build`] returns
    /// [`BuildError::UnsupportedTarget`] for this target. Two gaps remain:
    /// the builder passes NVIDIA `sm_*` values as `-C target-cpu` (rejected
    /// by the amdgcn LLVM backend), and kernel entry parsing understands
    /// only PTX `.visible .entry` syntax, so an amdgcn build would yield a
    /// kernel-less module.
    Amd,
}

impl GpuTarget {
    fn triple(&self) -> &str {
        match self {
            GpuTarget::Nvidia => "nvptx64-nvidia-cuda",
            GpuTarget::Amd => "amdgcn-amd-amdhsa",
        }
    }

    #[allow(dead_code)]
    fn asm_extension(&self) -> &str {
        match self {
            GpuTarget::Nvidia => "s", // PTX assembly
            GpuTarget::Amd => "s",    // GCN assembly
        }
    }
}

/// Builder for cross-compiling kernel crates to GPU assembly.
pub struct WarpBuilder {
    /// Path to the kernel crate (relative to the manifest dir).
    kernel_crate: PathBuf,
    /// Rust toolchain to use. `None` (the default) means: discover the pin from
    /// the kernel crate's own `rust-toolchain.toml`, falling back to `"nightly"`.
    toolchain: Option<String>,
    /// Release mode (default: true).
    release: bool,
    /// Feature flags to pass to the kernel crate.
    features: Vec<String>,
    /// GPU target (default: NVIDIA).
    target: GpuTarget,
    /// NVIDIA SM architecture (default: "sm_70").
    ///
    /// Controls the PTX ISA version emitted by LLVM. Must be at least `sm_70`
    /// (Volta) because warp-types uses `shfl.sync.*` which requires PTX 6.0+.
    /// Higher values enable newer instructions and may produce better code.
    /// Common values: sm_70 (Volta), sm_80 (Ampere), sm_89 (Ada), sm_90 (Hopper).
    sm_arch: String,
}

impl WarpBuilder {
    /// Create a new builder pointing at a kernel crate directory.
    ///
    /// The path is relative to the directory containing the host crate's `Cargo.toml`
    /// (i.e., `CARGO_MANIFEST_DIR`).
    pub fn new(kernel_crate_path: impl Into<PathBuf>) -> Self {
        WarpBuilder {
            kernel_crate: kernel_crate_path.into(),
            toolchain: None,
            release: true,
            features: Vec::new(),
            target: GpuTarget::Nvidia,
            sm_arch: "sm_70".to_string(),
        }
    }

    /// Set the Rust toolchain explicitly, overriding pin discovery.
    pub fn toolchain(mut self, toolchain: impl Into<String>) -> Self {
        self.toolchain = Some(toolchain.into());
        self
    }

    /// The toolchain the inner cargo will run under.
    ///
    /// `RUSTUP_TOOLCHAIN` takes precedence over `rust-toolchain.toml`, so
    /// hardcoding a floating `"nightly"` here silently unpinned every kernel
    /// build — including in a repo whose toolchain file exists precisely to make
    /// GPU artifacts reproducible. Discover the pin instead: walk up from the
    /// kernel crate for a `rust-toolchain.toml` and use its channel.
    ///
    /// Only a NIGHTLY channel is adopted. `-Z build-std` needs nightly, so a
    /// crate pinned to stable must not drag the kernel build down with it —
    /// that would turn a working build into a confusing `-Z` error. In that case
    /// fall back to plain `"nightly"`, which is the old behaviour.
    fn resolve_toolchain(&self, kernel_dir: &Path) -> String {
        if let Some(explicit) = &self.toolchain {
            return explicit.clone();
        }
        let mut dir = Some(kernel_dir);
        while let Some(d) = dir {
            for name in ["rust-toolchain.toml", "rust-toolchain"] {
                if let Ok(text) = std::fs::read_to_string(d.join(name)) {
                    if let Some(chan) = text
                        .lines()
                        .find_map(|l| l.trim().strip_prefix("channel"))
                        .and_then(|r| r.trim().strip_prefix('='))
                        .map(|r| r.trim().trim_matches('"').trim_matches('\'').to_string())
                    {
                        if chan.starts_with("nightly") {
                            return chan;
                        }
                    }
                }
            }
            dir = d.parent();
        }
        "nightly".to_string()
    }

    /// Disable release mode (compile in debug mode).
    ///
    /// Caveat (2026-10-18): a dev-profile device build is not link-complete as a
    /// single PTX file. Debug builds share generic instantiations with the
    /// upstream crate (`-Zshare-generics`) and keep `core::panicking` live, so the
    /// PTX carries `.extern .func` declarations that `cuModuleLoad` rejects with
    /// `CUDA_ERROR_INVALID_PTX`. Use the default release profile for anything you
    /// intend to launch; `.debug()` is for reading the unoptimised PTX.
    pub fn debug(mut self) -> Self {
        self.release = false;
        self
    }

    /// Set the GPU target (default: NVIDIA).
    pub fn target(mut self, target: GpuTarget) -> Self {
        self.target = target;
        self
    }

    /// Set the NVIDIA SM architecture (default: "sm_70").
    ///
    /// Controls `-C target-cpu` passed to rustc, which determines the PTX ISA
    /// version. Must be at least `sm_70` for `shfl.sync` support.
    pub fn sm_arch(mut self, arch: impl Into<String>) -> Self {
        self.sm_arch = arch.into();
        self
    }

    /// Enable a feature flag on the kernel crate.
    pub fn feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }

    /// Build the kernel crate, producing PTX and generating a Rust module.
    ///
    /// On success, writes to `OUT_DIR`:
    /// - `kernels.ptx` — the raw PTX assembly
    /// - `kernels.rs` — a Rust module with `KERNEL_PTX` constant and a `Kernels` struct
    ///   that provides named `CudaFunction` handles for each kernel entry point
    ///
    /// Also prints `cargo:rerun-if-changed` for all kernel source files (recursive).
    pub fn build(self) -> Result<BuildResult, BuildError> {
        // AMD emission is not yet supported — fail loudly and immediately
        // rather than (1) passing an NVIDIA sm_* target-cpu to the amdgcn
        // backend (raw LLVM error) or (2) parsing amdgcn assembly with the
        // PTX-only entry parser (silently kernel-less module).
        if matches!(self.target, GpuTarget::Amd) {
            return Err(BuildError::UnsupportedTarget(
                "GpuTarget::Amd is not yet supported: the builder passes NVIDIA sm_* values \
                 as -C target-cpu (rejected by the amdgcn LLVM backend), and kernel entry \
                 parsing is PTX-only (an amdgcn build would produce a kernel-less module). \
                 Use GpuTarget::Nvidia."
                    .to_string(),
            ));
        }

        let manifest_dir =
            env::var("CARGO_MANIFEST_DIR").map_err(|_| BuildError::NotInBuildScript)?;
        let out_dir = env::var("OUT_DIR").map_err(|_| BuildError::NotInBuildScript)?;

        let kernel_dir = Path::new(&manifest_dir).join(&self.kernel_crate);
        if !kernel_dir.exists() {
            return Err(BuildError::KernelCrateNotFound(kernel_dir));
        }

        // Tell cargo to rerun if ANY kernel source file changes (recursive)
        emit_rerun_if_changed(&kernel_dir);

        // Invoke cargo rustc for nvptx64 with --emit=asm to get PTX output.
        // Use RUSTUP_TOOLCHAIN env var instead of +nightly syntax, because
        // the +toolchain syntax requires rustup's proxy and doesn't work
        // when cargo is invoked from within a build script.
        let mut cmd = Command::new("cargo");
        cmd.arg("rustc")
            .arg("--target")
            .arg(self.target.triple())
            .arg("-Z")
            .arg("build-std=core")
            .current_dir(&kernel_dir);

        // Pin the target dir explicitly. If the kernel crate is a member of
        // the calling workspace, the inner cargo would otherwise resolve the
        // workspace root and use the workspace target/ — deadlocking on the
        // build lock the outer cargo holds while running this build script,
        // and putting outputs where find_ptx_file never looks. An explicit
        // --target-dir also overrides any inherited CARGO_TARGET_DIR, so the
        // search path below is guaranteed to match where cargo wrote.
        let inner_target_dir = kernel_dir.join("target");
        cmd.arg("--target-dir").arg(&inner_target_dir);

        if self.release {
            cmd.arg("--release");
        }

        // Pass feature flags to the kernel crate
        for feat in &self.features {
            cmd.arg("--features").arg(feat);
        }

        // After `--`, pass rustc flags: emit assembly (PTX for nvptx64)
        // -C target-cpu sets the SM architecture, which determines the PTX ISA
        // version. Without this, LLVM defaults to sm_30 / PTX 3.2, which lacks
        // shfl.sync (requires PTX 6.0 / sm_70+).
        cmd.arg("--")
            .arg("--emit=asm")
            .arg("-C")
            .arg(format!("target-cpu={}", self.sm_arch));

        // Select the toolchain via env var (works inside build scripts). This is
        // the kernel crate's own pin when it has one — see `resolve_toolchain`;
        // RUSTUP_TOOLCHAIN outranks rust-toolchain.toml, so a hardcoded
        // "nightly" here would override the very pin that makes the artifact
        // reproducible.
        // CRITICAL: remove RUSTC — the parent cargo sets it to the absolute path
        // of its own rustc (e.g., stable's rustc), which the inner cargo would
        // inherit and use directly, bypassing toolchain selection entirely.
        // This is the same fix that spirv-builder uses.
        cmd.env("RUSTUP_TOOLCHAIN", self.resolve_toolchain(&kernel_dir));
        cmd.env_remove("RUSTC");
        // Same class of leak as RUSTC — cargo sets these in the build-script
        // environment (or the user's shell may), and each silently changes
        // what the inner cargo does:
        // - CARGO_ENCODED_RUSTFLAGS takes precedence over RUSTFLAGS, so an
        //   inherited value would drop `--cfg warp_kernel_build` set below.
        // - CARGO_TARGET_DIR / CARGO_BUILD_TARGET_DIR would redirect output
        //   away from the directory searched below (the explicit --target-dir
        //   above already wins, but keep the invocation self-contained).
        // - RUSTC_WRAPPER / RUSTC_WORKSPACE_WRAPPER wrap rustc with a binary
        //   from the OUTER toolchain (e.g. clippy-driver under `cargo clippy`),
        //   bypassing the nightly selection above.
        cmd.env_remove("CARGO_ENCODED_RUSTFLAGS");
        cmd.env_remove("CARGO_TARGET_DIR");
        cmd.env_remove("CARGO_BUILD_TARGET_DIR");
        cmd.env_remove("RUSTC_WRAPPER");
        cmd.env_remove("RUSTC_WORKSPACE_WRAPPER");
        cmd.env("RUSTFLAGS", "--cfg warp_kernel_build");

        let output = cmd
            .output()
            .map_err(|e| BuildError::CargoFailed(format!("Failed to run cargo: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BuildError::CompilationFailed(stderr.into_owned()));
        }

        // Find the generated PTX/assembly file
        let profile = if self.release { "release" } else { "debug" };
        let target_dir = inner_target_dir.join(self.target.triple()).join(profile);

        let ptx_path = find_ptx_file(&target_dir, &kernel_dir)?;

        // Read PTX content
        let ptx_content = std::fs::read_to_string(&ptx_path)
            .map_err(|e| BuildError::PtxReadFailed(format!("{}: {}", ptx_path.display(), e)))?;

        // Parse kernel entry points from PTX
        let kernels = parse_kernel_entries(&ptx_content);

        // Write PTX to OUT_DIR
        let out_ptx = Path::new(&out_dir).join("kernels.ptx");
        std::fs::write(&out_ptx, &ptx_content)
            .map_err(|e| BuildError::WriteFailed(format!("{}: {}", out_ptx.display(), e)))?;

        // Generate Rust module with Kernels struct
        let out_rs = Path::new(&out_dir).join("kernels.rs");
        let rs_content = generate_rust_module(&self.kernel_crate, &kernels);
        std::fs::write(&out_rs, &rs_content)
            .map_err(|e| BuildError::WriteFailed(format!("{}: {}", out_rs.display(), e)))?;

        Ok(BuildResult {
            ptx_path: out_ptx,
            module_path: out_rs,
            kernel_names: kernels,
        })
    }
}

/// Result of a successful build.
pub struct BuildResult {
    /// Path to the generated PTX file in OUT_DIR.
    pub ptx_path: PathBuf,
    /// Path to the generated Rust module in OUT_DIR.
    pub module_path: PathBuf,
    /// Names of kernel entry points found in the PTX.
    pub kernel_names: Vec<String>,
}

/// Errors that can occur during kernel compilation.
#[derive(Debug)]
pub enum BuildError {
    /// Not running inside a build script (CARGO_MANIFEST_DIR not set).
    NotInBuildScript,
    /// Kernel crate directory not found.
    KernelCrateNotFound(PathBuf),
    /// The selected GPU target is not supported yet.
    UnsupportedTarget(String),
    /// cargo build failed.
    CargoFailed(String),
    /// Kernel crate compilation failed.
    CompilationFailed(String),
    /// Could not find generated PTX file.
    PtxNotFound(String),
    /// Could not read PTX file.
    PtxReadFailed(String),
    /// Could not write output files.
    WriteFailed(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::NotInBuildScript => write!(
                f,
                "Not running in a build script (CARGO_MANIFEST_DIR not set)"
            ),
            BuildError::KernelCrateNotFound(p) => {
                write!(f, "Kernel crate not found: {}", p.display())
            }
            BuildError::UnsupportedTarget(s) => write!(f, "Unsupported GPU target: {}", s),
            BuildError::CargoFailed(s) => write!(f, "Cargo invocation failed: {}", s),
            BuildError::CompilationFailed(s) => write!(f, "Kernel compilation failed:\n{}", s),
            BuildError::PtxNotFound(s) => write!(f, "PTX file not found: {}", s),
            BuildError::PtxReadFailed(s) => write!(f, "Failed to read PTX: {}", s),
            BuildError::WriteFailed(s) => write!(f, "Failed to write output: {}", s),
        }
    }
}

impl std::error::Error for BuildError {}

// ============================================================================
// PTX parsing
// ============================================================================

/// Parse `.visible .entry <name>(` lines from PTX to find kernel entry points.
fn parse_kernel_entries(ptx: &str) -> Vec<String> {
    ptx.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with(".visible .entry ") {
                // Format: .visible .entry kernel_name(
                let rest = trimmed.strip_prefix(".visible .entry ")?;
                let name = rest.split('(').next()?.trim();
                if !name.is_empty() {
                    Some(name.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

// ============================================================================
// Code generation
// ============================================================================

/// Generate a Rust module with PTX constant and Kernels struct.
fn generate_rust_module(kernel_crate: &Path, kernels: &[String]) -> String {
    let mut code = String::new();

    // Header
    code.push_str(&format!(
        "// Auto-generated by warp-types-builder. Do not edit.\n\
         // Kernel crate: {}\n\
         // Kernel count: {}\n\n",
        kernel_crate.display(),
        kernels.len(),
    ));

    // PTX constant
    code.push_str(
        "/// Raw PTX assembly for all kernels in this module.\n\
         pub const KERNEL_PTX: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/kernels.ptx\"));\n\n"
    );

    // Kernels struct
    code.push_str(
        "/// Loaded GPU kernel functions.\n\
         ///\n\
         /// Created by [`Kernels::load`], which parses the PTX and extracts\n\
         /// each kernel entry point as a ready-to-launch [`CudaFunction`].\n\
         ///\n\
         /// # Available kernels\n",
    );
    for name in kernels {
        code.push_str(&format!("/// - `{}` \n", name));
    }
    code.push_str(
        "pub struct Kernels {\n\
         _module: ::std::sync::Arc<::cudarc::driver::CudaModule>,\n",
    );
    for name in kernels {
        code.push_str(&format!(
            "    /// Kernel: `{name}`\n\
                 pub {name}: ::cudarc::driver::CudaFunction,\n",
            name = name,
        ));
    }
    code.push_str("}\n\n");

    // Kernels::load impl
    code.push_str(
        "impl Kernels {\n\
         /// Load all kernels from the compiled PTX.\n\
         ///\n\
         /// Parses the embedded PTX assembly, loads it as a CUDA module,\n\
         /// and extracts each kernel entry point by name.\n\
         pub fn load(ctx: &::std::sync::Arc<::cudarc::driver::CudaContext>) -> \
             ::std::result::Result<Self, Box<dyn ::std::error::Error>> {\n\
             let ptx = ::cudarc::nvrtc::Ptx::from_src(KERNEL_PTX.to_string());\n\
             let module = ctx.load_module(ptx)?;\n",
    );
    for name in kernels {
        code.push_str(&format!(
            "        let {name} = module.load_function(\"{name}\")?;\n",
            name = name,
        ));
    }
    code.push_str("        let _module = module;\n");
    code.push_str("        Ok(Kernels {\n            _module,\n");
    for name in kernels {
        code.push_str(&format!("            {},\n", name));
    }
    code.push_str("        })\n    }\n}\n");

    code
}

// ============================================================================
// File watching
// ============================================================================

/// Emit `cargo:rerun-if-changed` for all files in the kernel crate recursively.
fn emit_rerun_if_changed(kernel_dir: &Path) {
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("Cargo.toml").display()
    );

    let src_dir = kernel_dir.join("src");
    if src_dir.exists() {
        emit_rerun_recursive(&src_dir);
    }

    // Path dependencies of the kernel crate (warp-types itself, in the worked
    // example) are compiled INTO the PTX, so a change there must rebuild it too.
    // Without this the example ran a stale kernels.ptx twice on 2026-10-18 after
    // src/gpu.rs changed: cargo saw nothing under my-kernels/ move and reused
    // the old artifact (CUDA_ERROR_INVALID_PTX once the ABI had shifted).
    for dep in path_dependencies(kernel_dir) {
        println!(
            "cargo:rerun-if-changed={}",
            dep.join("Cargo.toml").display()
        );
        let dep_src = dep.join("src");
        if dep_src.exists() {
            emit_rerun_recursive(&dep_src);
        }
    }
}

/// `path = "..."` dependencies of the kernel crate, TRANSITIVELY (a change in
/// `warp-types/src` reaches the PTX through `my-kernels`'s path dep on it; a
/// path dep's own path deps count too), each resolved against the manifest that
/// names it. A minimal TOML scan, no parser dependency: a `path` KEY (not a
/// substring — `xpath = …` and comment lines are ignored) with a single- or
/// double-quoted value, inside a `[dependencies]` / `[dev-dependencies]` /
/// `[build-dependencies]` / `[target.*.dependencies]` table or a
/// `[dependencies.<name>]` table, or inline on a dependency line. Not handled
/// (documented, not silent): `path.workspace = true` inheritance.
fn path_dependencies(kernel_dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut queue: Vec<PathBuf> = vec![kernel_dir.to_path_buf()];
    while let Some(dir) = queue.pop() {
        let canon = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if seen.contains(&canon) {
            continue;
        }
        seen.push(canon);
        for dep in direct_path_dependencies(&dir) {
            if !out.contains(&dep) {
                out.push(dep.clone());
                queue.push(dep);
            }
        }
    }
    out
}

/// One manifest's `path = …` dependencies (see [`path_dependencies`]).
fn direct_path_dependencies(dir: &Path) -> Vec<PathBuf> {
    let Ok(content) = std::fs::read_to_string(dir.join("Cargo.toml")) else {
        return Vec::new();
    };
    let mut in_deps = false;
    let mut out = Vec::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let t = line.trim_matches(|c| c == '[' || c == ']').trim();
            in_deps = t == "dependencies"
                || t == "dev-dependencies"
                || t == "build-dependencies"
                || t.starts_with("dependencies.")
                || t.starts_with("dev-dependencies.")
                || t.starts_with("build-dependencies.")
                || (t.starts_with("target.") && t.contains("dependencies"));
            continue;
        }
        if !in_deps {
            continue;
        }
        // every `path = "<v>"` / `path = '<v>'` whose `path` is a whole key
        let bytes = line.as_bytes();
        let mut from = 0;
        while let Some(rel) = line[from..].find("path") {
            let at = from + rel;
            from = at + 4;
            let key_start = at == 0 || matches!(bytes[at - 1], b' ' | b'\t' | b'{' | b',');
            if !key_start {
                continue;
            }
            let rest = line[at + 4..].trim_start();
            let Some(rest) = rest.strip_prefix('=') else {
                continue;
            };
            let rest = rest.trim_start();
            let Some(q) = rest.chars().next() else {
                continue;
            };
            if q != '"' && q != '\'' {
                continue;
            }
            let inner = &rest[1..];
            if let Some(end) = inner.find(q) {
                let p = dir.join(&inner[..end]);
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

fn emit_rerun_recursive(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                emit_rerun_recursive(&path);
            } else {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

// ============================================================================
// PTX file discovery
// ============================================================================

/// Search for the PTX (.s) file in the target directory.
fn find_ptx_file(target_dir: &Path, kernel_dir: &Path) -> Result<PathBuf, BuildError> {
    // Get the crate name from Cargo.toml
    let cargo_toml = kernel_dir.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).map_err(|e| {
        BuildError::PtxNotFound(format!("Can't read {}: {}", cargo_toml.display(), e))
    })?;

    // Simple TOML parsing — find name under [package] section
    let crate_name = {
        let mut in_package = false;
        content
            .lines()
            .find_map(|line| {
                let line = line.trim();
                if line.starts_with('[') {
                    in_package = line == "[package]";
                    return None;
                }
                if in_package && line.starts_with("name") {
                    let val = line.split('=').nth(1)?.trim().trim_matches('"');
                    return Some(val.replace('-', "_"));
                }
                None
            })
            .unwrap_or_else(|| "kernels".to_string())
    };

    // Unhashed locations some toolchains/flags produce (kept for the error message
    // and as candidates — they compete on mtime with the hashed ones below, they
    // are NOT returned first: the 2026-10-18 panel pointed out that an exact-name
    // file from an old build would otherwise shadow a fresh hashed one, the very
    // bug this function was just fixed for).
    let candidates = [
        target_dir.join(format!("{}.s", crate_name)),
        target_dir.join(format!("lib{}.s", crate_name)),
        target_dir.join(format!("{}.ptx", crate_name)),
        target_dir.join("deps").join(format!("{}.s", crate_name)),
        target_dir.join("deps").join(format!("lib{}.s", crate_name)),
    ];

    // Where rustc puts `--emit=asm` output depends on the toolchain: older
    // nightlies wrote `deps/<crate>-HASH.s`; nightly-2026-09-15 was observed to
    // write `build/<crate>/<HASH>/out/<crate>-HASH.s`. Both layouts are
    // searched. Multiple crate_name-HASH.s artifacts accumulate across rebuilds
    // with different flags, and read_dir order is arbitrary — collect all
    // matches, exact-name candidates included, and pick the newest (see
    // pick_newest) so a stale artifact is never silently selected. (2026-10-18:
    // a stale `deps/` file from the previous pin shadowed the fresh
    // `build/.../out/` file for one pin bump, and the worked example ran kernels
    // that were not in its source.) mtime is a heuristic; the exact answer would
    // be `cargo --message-format=json`'s artifact list, which is a larger change.
    let dirs = asm_search_dirs(target_dir);
    let is_ours = |fname: &str| {
        let lib = format!("lib{}", crate_name);
        fname == crate_name
            || fname == lib
            || fname.starts_with(&format!("{}-", crate_name))
            || fname.starts_with(&format!("{}-", lib))
    };
    let mut matches: Vec<PathBuf> = asm_files(&dirs)
        .into_iter()
        .filter(|p| {
            let fname = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            // `<crate>-HASH.s` exactly — not a sibling crate sharing the prefix
            is_ours(&fname)
        })
        .collect();
    matches.extend(candidates.iter().filter(|p| p.exists()).cloned());
    if let Some(p) = pick_newest(matches) {
        return Ok(p);
    }

    // Last resort: any non-core .s file in either layout, newest first (same rationale)
    let any_asm: Vec<PathBuf> = asm_files(&dirs)
        .into_iter()
        .filter(|p| {
            let fname = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            !fname.starts_with("core-")
                && !fname.starts_with("compiler_builtins-")
                && !fname.starts_with("warp_types-")
        })
        .collect();
    if let Some(p) = pick_newest(any_asm) {
        return Ok(p);
    }

    Err(BuildError::PtxNotFound(format!(
        "No .s/.ptx file found in {}. Crate name: '{}'. Checked: {:?}",
        target_dir.display(),
        crate_name,
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
    )))
}

/// The directories rustc may have written `--emit=asm` output to under a
/// `target/<triple>/<profile>` dir: `deps/` (older toolchains) and every
/// `build/<pkg>/<hash>/out/` or `build/<pkg-hash>/out/` (nightly-2026-09-15+).
fn asm_search_dirs(target_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![target_dir.join("deps")];
    let build = target_dir.join("build");
    if let Ok(entries) = std::fs::read_dir(&build) {
        for e in entries.flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            // `build/<pkg-hash>/out` (old layout)
            let out = p.join("out");
            if out.is_dir() {
                dirs.push(out);
            }
            // `build/<pkg>/<hash>/out` (new layout)
            if let Ok(inner) = std::fs::read_dir(&p) {
                for h in inner.flatten() {
                    let out = h.path().join("out");
                    if out.is_dir() {
                        dirs.push(out);
                    }
                }
            }
        }
    }
    dirs
}

/// Every `*.s` file directly inside any of `dirs`.
fn asm_files(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for d in dirs {
        if let Ok(entries) = std::fs::read_dir(d) {
            out.extend(
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "s")),
            );
        }
    }
    out
}

/// Pick the best artifact among several discovered candidates: the one with
/// the newest modification time, so a stale artifact from an earlier build
/// can never shadow the one just produced. Files whose mtime can't be read
/// lose to files with a known mtime; if no mtimes are available (or they tie),
/// fall back to the lexicographically first path so the choice is at least
/// deterministic across read_dir orderings.
fn pick_newest(candidates: Vec<PathBuf>) -> Option<PathBuf> {
    let stamped = candidates
        .into_iter()
        .map(|p| {
            let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
            (p, mtime)
        })
        .collect();
    pick_newest_stamped(stamped)
}

/// Pure selection core of [`pick_newest`]: newest mtime wins; `None` mtimes
/// lose to `Some`; ties (including all-`None`) resolve to sorted-name order.
fn pick_newest_stamped(
    mut stamped: Vec<(PathBuf, Option<std::time::SystemTime>)>,
) -> Option<PathBuf> {
    // Deterministic base order regardless of read_dir ordering.
    stamped.sort_by(|a, b| a.0.cmp(&b.0));
    let mut best: Option<(PathBuf, Option<std::time::SystemTime>)> = None;
    for (path, mtime) in stamped {
        match &best {
            // Strict `>` keeps the earlier (sorted-name-first) path on ties.
            Some((_, best_mtime)) if mtime <= *best_mtime => {}
            _ => best = Some((path, mtime)),
        }
    }
    best.map(|(p, _)| p)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn t(secs: u64) -> Option<SystemTime> {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
    }

    // --- pick_newest_stamped: stale-artifact selection (finding A) ---

    #[test]
    fn newest_mtime_wins_over_name_order() {
        // The stale artifact sorts FIRST by name — pure name-sort would pick it.
        let stale = PathBuf::from("deps/kernels-aaaa.s");
        let fresh = PathBuf::from("deps/kernels-zzzz.s");
        let picked = pick_newest_stamped(vec![(fresh.clone(), t(2_000)), (stale, t(1_000))]);
        assert_eq!(picked, Some(fresh));
    }

    #[test]
    fn result_is_independent_of_read_dir_order() {
        let a = (PathBuf::from("deps/kernels-aaaa.s"), t(3_000));
        let b = (PathBuf::from("deps/kernels-bbbb.s"), t(1_000));
        let c = (PathBuf::from("deps/kernels-cccc.s"), t(2_000));
        for perm in [
            vec![a.clone(), b.clone(), c.clone()],
            vec![c.clone(), a.clone(), b.clone()],
            vec![b.clone(), c.clone(), a.clone()],
        ] {
            assert_eq!(pick_newest_stamped(perm), Some(a.0.clone()));
        }
    }

    #[test]
    fn missing_mtimes_fall_back_to_sorted_name() {
        let picked = pick_newest_stamped(vec![
            (PathBuf::from("deps/kernels-zzzz.s"), None),
            (PathBuf::from("deps/kernels-aaaa.s"), None),
        ]);
        assert_eq!(picked, Some(PathBuf::from("deps/kernels-aaaa.s")));
    }

    #[test]
    fn known_mtime_beats_unknown() {
        let picked = pick_newest_stamped(vec![
            (PathBuf::from("deps/kernels-aaaa.s"), None),
            (PathBuf::from("deps/kernels-bbbb.s"), t(1)),
        ]);
        assert_eq!(picked, Some(PathBuf::from("deps/kernels-bbbb.s")));
    }

    #[test]
    fn equal_mtimes_resolve_to_sorted_name() {
        let picked = pick_newest_stamped(vec![
            (PathBuf::from("deps/kernels-bbbb.s"), t(5)),
            (PathBuf::from("deps/kernels-aaaa.s"), t(5)),
        ]);
        assert_eq!(picked, Some(PathBuf::from("deps/kernels-aaaa.s")));
    }

    #[test]
    fn empty_candidates_give_none() {
        assert_eq!(pick_newest_stamped(vec![]), None);
    }

    #[test]
    fn find_ptx_file_prefers_fresh_build_out_over_stale_deps() {
        // nightly-2026-09-15 writes --emit=asm output to build/<crate>/<hash>/out/,
        // not deps/. A stale deps/ artifact from an older pin must not win.
        let dir = std::env::temp_dir().join(format!(
            "warp-types-builder-layout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let kernel_dir = dir.join("my-kernels");
        let target_dir = dir
            .join("target")
            .join("nvptx64-nvidia-cuda")
            .join("release");
        std::fs::create_dir_all(&kernel_dir).unwrap();
        std::fs::write(
            kernel_dir.join("Cargo.toml"),
            "[package]\nname = \"my-kernels\"\nversion = \"0.0.0\"\n",
        )
        .unwrap();
        let deps = target_dir.join("deps");
        let out = target_dir
            .join("build")
            .join("my-kernels")
            .join("deadbeef")
            .join("out");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::create_dir_all(&out).unwrap();
        let stale = deps.join("my_kernels-aaaa.s");
        let fresh = out.join("my_kernels-deadbeef.s");
        std::fs::write(&stale, ".visible .entry old(").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&fresh, ".visible .entry new(").unwrap();
        let picked = find_ptx_file(&target_dir, &kernel_dir).unwrap();
        assert_eq!(picked, fresh);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_ptx_file_exact_name_competes_on_mtime_and_prefix_siblings_are_ignored() {
        let dir = std::env::temp_dir().join(format!(
            "warp-types-builder-exact-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let kernel_dir = dir.join("my-kernels");
        let target_dir = dir
            .join("target")
            .join("nvptx64-nvidia-cuda")
            .join("release");
        std::fs::create_dir_all(&kernel_dir).unwrap();
        std::fs::write(
            kernel_dir.join("Cargo.toml"),
            "[package]\nname = \"my-kernels\"\nversion = \"0.0.0\"\n",
        )
        .unwrap();
        let deps = target_dir.join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        // stale EXACT-name artifact (the pre-fix fast path returned this first)
        let stale_exact = deps.join("my_kernels.s");
        std::fs::write(&stale_exact, "old").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        // a sibling crate sharing the prefix, newer than everything: must be ignored
        let sibling = deps.join("my_kernels_extra-ffff.s");
        let fresh = deps.join("my_kernels-beef.s");
        std::fs::write(&fresh, "new").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&sibling, "sibling").unwrap();
        let picked = find_ptx_file(&target_dir, &kernel_dir).unwrap();
        assert_eq!(picked, fresh);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_dependencies_are_found_in_all_dependency_table_shapes() {
        let dir = std::env::temp_dir().join(format!(
            "warp-types-builder-pathdeps-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"k\"\nversion = \"0.0.0\"\npath = \"not-a-dep\"\n\n\
             [dependencies]\nwarp-types = { path = \"../../\" }\nother = \"1.0\"\n\
             # path = \"commented-out\"\nxpathy = { xpath = \"../xpath-crate\", version = \"1\" }\n\
             single = { path = '../single' }\n\n\
             [dependencies.local]\npath = \"../local\"\nfeatures = [\"x\"]\n\n\
             [profile.release]\nopt-level = 3\n",
        )
        .unwrap();
        // the first dep is itself a crate with a path dep: must be found transitively
        std::fs::create_dir_all(dir.join("../../")).ok();
        let nested = dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            nested.join("Cargo.toml"),
            "[package]\nname = \"n\"\nversion = \"0.0.0\"\n[dependencies]\ndeeper = { path = \"deeper\" }\n",
        )
        .unwrap();
        let mut toml = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        toml.push_str("[build-dependencies]\nnested = { path = \"nested\" }\n");
        std::fs::write(dir.join("Cargo.toml"), toml).unwrap();
        let deps = path_dependencies(&dir);
        assert!(deps.contains(&dir.join("../../")), "{deps:?}");
        assert!(
            deps.contains(&dir.join("../single")),
            "single-quoted path: {deps:?}"
        );
        assert!(deps.contains(&dir.join("../local")), "{deps:?}");
        assert!(deps.contains(&nested), "{deps:?}");
        assert!(
            deps.contains(&nested.join("deeper")),
            "transitive: {deps:?}"
        );
        assert!(
            !deps.iter().any(|d| d.ends_with("commented-out")),
            "comment: {deps:?}"
        );
        assert!(
            !deps.iter().any(|d| d.ends_with("xpath-crate")),
            "key boundary: {deps:?}"
        );
        assert!(!deps.iter().any(|d| d.ends_with("not-a-dep")), "{deps:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pick_newest_uses_real_mtimes() {
        // Filesystem-backed check: the file written LAST wins even though it
        // sorts last by name (old first-hit / name-sort behavior would be
        // order-dependent or pick the stale one).
        let dir = std::env::temp_dir().join(format!(
            "warp-types-builder-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("kernels-aaaa.s");
        let fresh = dir.join("kernels-zzzz.s");
        std::fs::write(&stale, "stale").unwrap();
        // Ensure a strictly newer mtime even on coarse-grained filesystems.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&fresh, "fresh").unwrap();
        let picked = pick_newest(vec![stale.clone(), fresh.clone()]);
        assert_eq!(picked, Some(fresh));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GpuTarget::Amd: loud typed error instead of raw/silent failure ---

    #[test]
    fn amd_target_errors_loudly() {
        // Must fail before any cargo invocation or env-var lookup: the
        // target is unsupported regardless of environment.
        let err = match WarpBuilder::new("kernels").target(GpuTarget::Amd).build() {
            Err(e) => e,
            Ok(_) => panic!("expected UnsupportedTarget error, got Ok"),
        };
        match err {
            BuildError::UnsupportedTarget(msg) => {
                assert!(msg.contains("GpuTarget::Amd"), "names the target: {msg}");
                assert!(msg.contains("target-cpu"), "names the sm_* gap: {msg}");
                assert!(msg.contains("PTX-only"), "names the parser gap: {msg}");
            }
            other => panic!("expected UnsupportedTarget, got: {other:?}"),
        }
    }

    // --- parse_kernel_entries: pure PTX parsing ---

    #[test]
    fn parses_visible_entries_only() {
        let ptx = "\
// Generated by LLVM
.visible .entry butterfly_reduce(
    .param .u64 p0
)
.visible .entry ballot_sync_kernel(
.entry hidden_helper(
.visible .entry (
";
        assert_eq!(
            parse_kernel_entries(ptx),
            vec!["butterfly_reduce", "ballot_sync_kernel"]
        );
    }

    #[test]
    fn parses_no_entries_from_empty_ptx() {
        assert!(parse_kernel_entries("").is_empty());
    }

    /// Scratch dir with a `rust-toolchain.toml` two levels up from the "crate",
    /// so the walk-up is exercised rather than a same-directory hit.
    fn scratch(tag: &str, channel: Option<&str>) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "wtb-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let crate_dir = root.join("workspace").join("kernels");
        std::fs::create_dir_all(&crate_dir).unwrap();
        if let Some(c) = channel {
            std::fs::write(
                root.join("workspace").join("rust-toolchain.toml"),
                format!("[toolchain]\nchannel = \"{c}\"\n"),
            )
            .unwrap();
        }
        crate_dir
    }

    #[test]
    fn adopts_a_nightly_pin_found_by_walking_up() {
        let dir = scratch("pin", Some("nightly-2026-07-01"));
        let b = WarpBuilder::new(".");
        assert_eq!(b.resolve_toolchain(&dir), "nightly-2026-07-01");
    }

    #[test]
    fn ignores_a_stable_pin_because_build_std_needs_nightly() {
        // A kernel crate under a stable-pinned project must not drag the GPU
        // build to stable: `-Z build-std` would fail with a confusing error.
        let dir = scratch("stable", Some("1.90.0"));
        let b = WarpBuilder::new(".");
        assert_eq!(b.resolve_toolchain(&dir), "nightly");
    }

    #[test]
    fn falls_back_to_nightly_when_no_pin_exists() {
        let dir = scratch("nopin", None);
        let b = WarpBuilder::new(".");
        assert_eq!(b.resolve_toolchain(&dir), "nightly");
    }

    #[test]
    fn explicit_toolchain_overrides_discovery() {
        let dir = scratch("explicit", Some("nightly-2026-07-01"));
        let b = WarpBuilder::new(".").toolchain("nightly-2026-01-01");
        assert_eq!(b.resolve_toolchain(&dir), "nightly-2026-01-01");
    }

    #[test]
    fn discovers_this_repos_own_pin_from_sat_kernels() {
        // The case that motivated the fix: an in-tree kernel crate must build
        // under the repo pin, not a floating nightly.
        let sat_kernels = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("warp-types-sat/sat-kernels");
        if !sat_kernels.is_dir() {
            return; // vendored/partial checkout
        }
        let resolved = WarpBuilder::new(".").resolve_toolchain(&sat_kernels);
        assert!(
            resolved.starts_with("nightly-"),
            "expected the repo's dated pin, got {resolved}"
        );
    }
}
