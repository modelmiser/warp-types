//! Build-time PTX compilation for warp-types GPU kernels.
//!
//! Use in your `build.rs` to cross-compile a kernel crate to PTX,
//! then load the generated PTX at runtime via cudarc.
//!
//! # Example
//!
//! ```rust,no_run
//! // build.rs
//! fn main() {
//!     warp_types_builder::WarpBuilder::new("my-kernels")
//!         .build()
//!         .expect("Failed to compile GPU kernels");
//! }
//! ```
//!
//! Then in your main crate:
//!
//! ```rust,ignore
//! // src/main.rs
//! include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
//! // Now you have: const KERNEL_PTX: &str = "...";
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Builder for cross-compiling kernel crates to PTX.
pub struct WarpBuilder {
    /// Path to the kernel crate (relative to the manifest dir).
    kernel_crate: PathBuf,
    /// Rust toolchain to use (default: "nightly").
    toolchain: String,
    /// Release mode (default: true).
    release: bool,
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

    /// Build the kernel crate, producing PTX and generating a Rust module.
    ///
    /// On success, writes two files to `OUT_DIR`:
    /// - `kernels.ptx` — the raw PTX assembly
    /// - `kernels.rs` — a Rust module with `const KERNEL_PTX: &str`
    ///
    /// Also prints `cargo:rerun-if-changed` for the kernel crate's source files.
    pub fn build(self) -> Result<BuildResult, BuildError> {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")
            .map_err(|_| BuildError::NotInBuildScript)?;
        let out_dir = env::var("OUT_DIR")
            .map_err(|_| BuildError::NotInBuildScript)?;

        let kernel_dir = Path::new(&manifest_dir).join(&self.kernel_crate);
        if !kernel_dir.exists() {
            return Err(BuildError::KernelCrateNotFound(kernel_dir));
        }

        // Tell cargo to rerun if kernel sources change
        let kernel_src = kernel_dir.join("src");
        if kernel_src.exists() {
            println!("cargo:rerun-if-changed={}", kernel_src.display());
        }
        println!("cargo:rerun-if-changed={}", kernel_dir.join("Cargo.toml").display());

        // Invoke cargo rustc for nvptx64 with --emit=asm to get PTX output.
        // Use RUSTUP_TOOLCHAIN env var instead of +nightly syntax, because
        // the +toolchain syntax requires rustup's proxy and doesn't work
        // when cargo is invoked from within a build script.
        let mut cmd = Command::new("cargo");
        cmd.arg("rustc")
            .arg("--target")
            .arg("nvptx64-nvidia-cuda")
            .arg("-Z")
            .arg("build-std=core")
            .current_dir(&kernel_dir);

        if self.release {
            cmd.arg("--release");
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

        let output = cmd.output()
            .map_err(|e| BuildError::CargoFailed(format!("Failed to run cargo: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BuildError::CompilationFailed(stderr.into_owned()));
        }

        // Find the generated PTX/assembly file
        let profile = if self.release { "release" } else { "debug" };
        let target_dir = kernel_dir
            .join("target")
            .join("nvptx64-nvidia-cuda")
            .join(profile);

        // The PTX file is either .s or in deps/
        let ptx_path = find_ptx_file(&target_dir, &kernel_dir)?;

        // Read PTX content
        let ptx_content = std::fs::read_to_string(&ptx_path)
            .map_err(|e| BuildError::PtxReadFailed(format!("{}: {}", ptx_path.display(), e)))?;

        // Write PTX to OUT_DIR
        let out_ptx = Path::new(&out_dir).join("kernels.ptx");
        std::fs::write(&out_ptx, &ptx_content)
            .map_err(|e| BuildError::WriteFailed(format!("{}: {}", out_ptx.display(), e)))?;

        // Generate Rust module with include_str!
        let out_rs = Path::new(&out_dir).join("kernels.rs");
        let rs_content = format!(
            "/// Auto-generated PTX for warp-types kernels.\n\
             /// Built from: {kernel_crate}\n\
             pub const KERNEL_PTX: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/kernels.ptx\"));\n",
            kernel_crate = self.kernel_crate.display(),
        );
        std::fs::write(&out_rs, rs_content)
            .map_err(|e| BuildError::WriteFailed(format!("{}: {}", out_rs.display(), e)))?;

        Ok(BuildResult {
            ptx_path: out_ptx,
            module_path: out_rs,
        })
    }
}

/// Result of a successful build.
pub struct BuildResult {
    /// Path to the generated PTX file in OUT_DIR.
    pub ptx_path: PathBuf,
    /// Path to the generated Rust module in OUT_DIR.
    pub module_path: PathBuf,
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
            BuildError::NotInBuildScript => write!(f, "Not running in a build script (CARGO_MANIFEST_DIR not set)"),
            BuildError::KernelCrateNotFound(p) => write!(f, "Kernel crate not found: {}", p.display()),
            BuildError::CargoFailed(s) => write!(f, "Cargo invocation failed: {}", s),
            BuildError::CompilationFailed(s) => write!(f, "Kernel compilation failed:\n{}", s),
            BuildError::PtxNotFound(s) => write!(f, "PTX file not found: {}", s),
            BuildError::PtxReadFailed(s) => write!(f, "Failed to read PTX: {}", s),
            BuildError::WriteFailed(s) => write!(f, "Failed to write output: {}", s),
        }
    }
}

impl std::error::Error for BuildError {}

/// Search for the PTX (.s) file in the target directory.
fn find_ptx_file(target_dir: &Path, kernel_dir: &Path) -> Result<PathBuf, BuildError> {
    // Get the crate name from Cargo.toml
    let cargo_toml = kernel_dir.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml)
        .map_err(|e| BuildError::PtxNotFound(format!("Can't read {}: {}", cargo_toml.display(), e)))?;

    // Simple TOML parsing — find name = "..."
    let crate_name = content.lines()
        .find_map(|line| {
            let line = line.trim();
            if line.starts_with("name") {
                let val = line.split('=').nth(1)?.trim().trim_matches('"');
                Some(val.replace('-', "_"))
            } else {
                None
            }
        })
        .unwrap_or_else(|| "kernels".to_string());

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

    // Fallback: find any .s file in the directory
    if let Ok(entries) = std::fs::read_dir(target_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "s") {
                return Ok(p);
            }
        }
    }

    // Also check deps/
    let deps = target_dir.join("deps");
    if let Ok(entries) = std::fs::read_dir(&deps) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "s")
                && !p.file_name().map_or(false, |n| n.to_string_lossy().starts_with("core-"))
            {
                return Ok(p);
            }
        }
    }

    Err(BuildError::PtxNotFound(format!(
        "No .s/.ptx file found in {}. Checked: {:?}",
        target_dir.display(),
        candidates.iter().map(|c| c.display().to_string()).collect::<Vec<_>>()
    )))
}
