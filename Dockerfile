# Dockerfile --- POPL Artifact Evaluation reproduction image for warp-types.
#
# Hermetic build pinned to the same toolchains the project's CI uses:
#   Lean 4.28.0 (from lean/lean-toolchain)
#   Rust nightly-2026-04-03 (from rust-toolchain.toml)
#
# Invocation for reviewers:
#   docker build -t warp-types-aec .
#   docker run --rm warp-types-aec
#
# On success, the container prints "ALL CHECKS PASSED" after running:
#   (a) lake build WarpTypes  --- 10 Lean files, zero sorry, zero user-declared axioms
#   (b) cargo test --workspace --lib  --- Rust type-system tests (GPU feature excluded,
#       requires CUDA; the paper's Lean mechanization is the primary artifact)
#
# First build: ~15 min (toolchain installs + full compile). Subsequent: ~3 min (cached).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:/root/.elan/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential libssl-dev pkg-config \
  && rm -rf /var/lib/apt/lists/*

# elan --- Lean toolchain manager. Installs Lean v4.28.0 on first `lake`
# invocation, driven by lean/lean-toolchain.
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none

# rustup --- Rust toolchain manager. Installs nightly-2026-04-03 on first
# `cargo` invocation, driven by rust-toolchain.toml.
RUN curl -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain none --profile minimal

WORKDIR /warp-types

# Stage 1 --- toolchain pins only. Cached independently of source changes
# so reviewers iterating on the paper don't re-download toolchains.
COPY lean/lean-toolchain ./lean/lean-toolchain
RUN cd lean && lake --version

COPY rust-toolchain.toml ./rust-toolchain.toml
RUN cargo --version

# Stage 2 --- full source.
COPY . ./

# Stage 3 --- verification runs at build time. The image cannot be built
# unless both checks pass, so `docker build` succeeding is itself the
# artifact-evaluation signal.
RUN cd lean && lake build WarpTypes
RUN cargo test --workspace --lib

CMD ["bash", "-c", "set -e; cd /warp-types/lean && lake build WarpTypes && cd /warp-types && cargo test --workspace --lib && echo && echo 'ALL CHECKS PASSED'"]
