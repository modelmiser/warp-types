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
    /// AMD amdgcn (64-lane wavefronts, AMDGPU output)
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
    /// Rust toolchain to use (default: "nightly").
    toolchain: String,
    /// Release mode (default: true).
    release: bool,
    /// Feature flags to pass to the kernel crate.
    features: Vec<String>,
    /// GPU target (default: NVIDIA).
    target: GpuTarget,
}

impl WarpBuilder {
    /// Create a new builder pointing at a kernel crate directory.
    ///
    /// The path is relative to the directory containing the host crate's `Cargo.toml`
    /// (i.e., `CARGO_MANIFEST_DIR`).
    pub fn new(kernel_crate_path: impl Into<PathBuf>) -> Self {
        WarpBuilder {
            kernel_crate: kernel_crate_path.into(),
            toolchain: "nightly".to_string(),
            release: true,
            features: Vec::new(),
            target: GpuTarget::Nvidia,
        }
    }

    /// Set the Rust toolchain (default: "nightly").
    pub fn toolchain(mut self, toolchain: impl Into<String>) -> Self {
        self.toolchain = toolchain.into();
        self
    }

    /// Disable release mode (compile in debug mode).
    pub fn debug(mut self) -> Self {
        self.release = false;
        self
    }

    /// Set the GPU target (default: NVIDIA).
    pub fn target(mut self, target: GpuTarget) -> Self {
        self.target = target;
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

        if self.release {
            cmd.arg("--release");
        }

        // Pass feature flags to the kernel crate
        for feat in &self.features {
            cmd.arg("--features").arg(feat);
        }

        // After `--`, pass rustc flags: emit assembly (PTX for nvptx64)
        cmd.arg("--").arg("--emit=asm");

        // Select nightly toolchain via env var (works inside build scripts).
        // CRITICAL: remove RUSTC — the parent cargo sets it to the absolute path
        // of its own rustc (e.g., stable's rustc), which the inner cargo would
        // inherit and use directly, bypassing toolchain selection entirely.
        // This is the same fix that spirv-builder uses.
        cmd.env("RUSTUP_TOOLCHAIN", &self.toolchain);
        cmd.env_remove("RUSTC");
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
        let target_dir = kernel_dir
            .join("target")
            .join(self.target.triple())
            .join(profile);

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

    // Check common locations for the .s file
    let candidates = [
        target_dir.join(format!("{}.s", crate_name)),
        target_dir.join(format!("lib{}.s", crate_name)),
        target_dir.join(format!("{}.ptx", crate_name)),
        target_dir.join("deps").join(format!("{}.s", crate_name)),
        target_dir.join("deps").join(format!("lib{}.s", crate_name)),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    // Fallback: search deps/ for .s files matching the crate name pattern
    let deps = target_dir.join("deps");
    if let Ok(entries) = std::fs::read_dir(&deps) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "s") {
                let fname = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                // Match crate_name-HASH.s pattern, skip core/compiler_builtins
                if fname.starts_with(&crate_name)
                    && !fname.starts_with("core-")
                    && !fname.starts_with("compiler_builtins-")
                {
                    return Ok(p);
                }
            }
        }
    }

    // Last resort: any non-core .s file in deps/ (sorted for determinism)
    if let Ok(entries) = std::fs::read_dir(&deps) {
        let mut candidates: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "s") && {
                    let fname = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    !fname.starts_with("core-")
                        && !fname.starts_with("compiler_builtins-")
                        && !fname.starts_with("warp_types-")
                }
            })
            .collect();
        candidates.sort();
        if let Some(p) = candidates.into_iter().next() {
            return Ok(p);
        }
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
