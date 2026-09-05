#!/bin/bash
# gpu-session.sh — rent a GPU, verify, kill it.
#
#   tools/gpu-session.sh up [gpu]     create a pod and print how to reach it
#   tools/gpu-session.sh run [gpu]    up + run reproduce/runpod-h200.sh + DOWN
#   tools/gpu-session.sh status       what is running and what it is costing
#   tools/gpu-session.sh down         terminate every pod on the account
#
# WHY THIS EXISTS, and why it is not a CI runner. Hardware demand here is
# bursty, not continuous: five sessions in the six months to 2026-09, clustered
# around cold-review closure, artifact regeneration and paper claims — never
# around routine pushes. On-demand costs about $1 per six months; an always-on
# self-hosted runner at the same $0.20/hr is ~$865. And the one thing the runner
# would add — continuous compile-rot detection — is already covered by the
# `cargo check --features gpu` step in the pre-push hook, which is 7s and needs
# no pod at all. See TODO.
#
# A self-hosted runner was also DECLINED for a second reason: warp-types is a
# PUBLIC repo, and a self-hosted runner lets anyone who opens a pull request
# execute code on the runner host. Not something to point at a workstation
# holding SSH keys, a gh PAT and trading material.
#
# THE EXPENSIVE MISTAKE IS FORGETTING TO TERMINATE. Pods bill by the hour while
# RUNNING and survive across sessions and reboots. `run` always terminates, even
# when the verification fails. After a manual `up`, `down` is your job.
set -uo pipefail

GPU_DEFAULT="NVIDIA RTX 4000 Ada Generation"   # ~$0.20/hr community; compute 8.9,
                                               # same card as this workstation and
                                               # one of the two the paper cites.
# H200 (~$4.59/hr) is only worth it for sm_90-specific results: the zero-overhead
# PTX comparison at sm_90 and the paper's H200 claims. Everything else — sanitizer
# runs, gpu_launcher assert exercise, gpu-feature regression — runs on the Ada,
# or on this workstation for free.
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

die() { echo "gpu-session: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 not found"; }

pods_json() { runpodctl pod list 2>/dev/null; }

cmd_status() {
    local out; out=$(pods_json)
    python3 - "$out" <<'PY'
import json, sys
pods = json.loads(sys.argv[1] or "[]")
if not pods:
    print("no pods running — nothing billing")
    raise SystemExit(0)
total = 0.0
for p in pods:
    hr = p.get("costPerHr", 0) or 0
    total += hr
    print(f"  {p['id']}  {p.get('name','?')}  ${hr}/hr  {p.get('runtimeStatus','?')}")
print(f"TOTAL: ${total:.2f}/hr  = ${total*24:.2f}/day  — 'gpu-session.sh down' to stop")
PY
}

cmd_down() {
    local ids; ids=$(pods_json | python3 -c 'import json,sys; print(" ".join(p["id"] for p in json.load(sys.stdin)))')
    [ -z "$ids" ] && { echo "no pods to terminate"; return 0; }
    for id in $ids; do
        echo "terminating $id"
        runpodctl pod remove "$id" >/dev/null 2>&1 || echo "  WARNING: remove failed for $id — check the console"
    done
    # Verify, do not assume. A remove that silently failed looks exactly like one
    # that worked, and the difference is a running meter.
    sleep 3
    local left; left=$(pods_json | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
    [ "$left" = "0" ] && echo "confirmed: no pods remain" || { echo "STILL $left POD(S) RUNNING — check https://console.runpod.io"; return 1; }
}

cmd_up() {
    local gpu="${1:-$GPU_DEFAULT}"
    echo "creating pod: $gpu  (billing starts now)"
    runpodctl pod create \
        --gpu-id "$gpu" --image "$IMAGE" \
        --cloud-type COMMUNITY --public-ip --ssh \
        --ports '22/tcp' --container-disk-in-gb 30 \
        --name "warp-types-verify" \
        --wait --wait-timeout 10m \
      | python3 -c '
import json, sys
p = json.load(sys.stdin)
s = p.get("ssh", {})
print(f"\npod {p[\"id\"]}  ${p.get(\"costPerHr\",\"?\")}/hr")
print(f"ssh: {s.get(\"ssh_command\",\"(not ready)\")}")
print("\nnvcc is NOT on PATH in these images — export PATH=/usr/local/cuda/bin:$PATH")
print("REMEMBER: tools/gpu-session.sh down")
'
}

cmd_run() {
    local gpu="${1:-$GPU_DEFAULT}"
    cmd_up "$gpu" || die "pod creation failed"
    local ssh_cmd; ssh_cmd=$(pods_json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["ssh"]["ssh_command"])')
    echo "=== running verification ==="
    # trap, not && — the pod must die even if the verification fails, which is
    # exactly when a human forgets.
    trap cmd_down EXIT
    $ssh_cmd -o StrictHostKeyChecking=accept-new \
        'export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
         cd /tmp && git clone -q https://github.com/modelmiser/warp-types.git 2>/dev/null
         cd /tmp/warp-types && git pull -q && bash reproduce/runpod-h200.sh'
    echo "=== verification done; terminating ==="
}

need runpodctl; need python3
case "${1:-status}" in
    up)     cmd_up "${2:-}" ;;
    run)    cmd_run "${2:-}" ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    *)      die "usage: $0 {up|run|down|status} [gpu-id]" ;;
esac
