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
STATE="${STATE:-$PWD/target/gpu-session-pod.json}"   # last create response (id, ssh command)
# CLOUD_TYPE=SECURE when COMMUNITY has no instances (2026-10-18: no community H200 SXM; secure $5.29/hr).

die() { echo "gpu-session: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 not found"; }

# A list that fails (no key, expired key) must not read as "no pods running":
# on 2026-10-18 `status` said exactly that with ~/.runpod/config.toml's apikey
# blank (secrets moved to `pass` 2026-10-07), which is a check that cannot fail.
pods_json() { runpodctl pod list || die "runpodctl pod list failed (RUNPOD_API_KEY unset or invalid?)"; }

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
    # The create response is the ONLY place the ssh command is reported (`pod list`
    # has no "ssh" field: 2026-10-18 a run created a $5.29/hr pod, KeyError'd on
    # it, ran nothing, terminated). Keep it on disk for cmd_run.
    mkdir -p "$(dirname "$STATE")"
    runpodctl pod create \
        --gpu-id "$gpu" --image "$IMAGE" \
        --cloud-type "${CLOUD_TYPE:-COMMUNITY}" --public-ip --ssh \
        --ports '22/tcp' --container-disk-in-gb 30 \
        --name "warp-types-verify" \
        --wait --wait-timeout 10m \
      | tee "$STATE" | python3 -c '
import json, sys
p = json.load(sys.stdin)
s = p.get("ssh") or {}
pid = p.get("id", "?")
cost = p.get("costPerHr", "?")
cmd = s.get("ssh_command", "(not ready)")
print("")
print("pod " + str(pid) + "  $" + str(cost) + "/hr")
print("ssh: " + str(cmd))
print("")
print("nvcc is NOT on PATH in these images - export PATH=/usr/local/cuda/bin:$PATH")
print("REMEMBER: tools/gpu-session.sh down")
'
}

cmd_run() {
    local gpu="${1:-$GPU_DEFAULT}"
    # Trap FIRST, before anything can create a pod. Learned the hard way
    # 2026-09-05: `cmd_up || die` ran before the trap was armed, so a create that
    # succeeded-then-failed-to-print left a pod running while reporting failure —
    # the exact leak the trap exists to stop. Arm it, then create.
    trap cmd_down EXIT
    cmd_up "$gpu" || die "pod creation failed (pod terminated by trap if one was made)"
    local ssh_cmd
    ssh_cmd=$(python3 - "$STATE" <<'PY'
import json, sys, subprocess
p = json.load(open(sys.argv[1]))
cmd = (p.get("ssh") or {}).get("ssh_command") or ""
if not cmd and p.get("id"):
    # fall back to the pod's own record
    try:
        q = json.loads(subprocess.run(["runpodctl", "pod", "get", p["id"]], capture_output=True, text=True, check=True).stdout)
        cmd = (q.get("ssh") or {}).get("ssh_command") or ""
    except Exception:
        pass
print(cmd)
PY
    ) || die "could not read the create response ($STATE)"
    [ -n "$ssh_cmd" ] || die "no ssh command for the pod (see $STATE); nothing was run"
    # WARP_REF selects the branch or commit to verify (default main). The whole
    # remote transcript is also kept locally under results/, because the pod's
    # /tmp/h200-results.txt dies with the pod and the paper's H200 numbers had
    # no committed artifact until 2026-10-18.
    local ref="${WARP_REF:-main}"
    local log="results/h200-$(date -u +%Y-%m-%d)-${ref//\//_}.txt"
    mkdir -p results
    echo "=== running verification (ref $ref) -> $log ==="
    $ssh_cmd -o StrictHostKeyChecking=accept-new \
        "export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$PATH
         cd /tmp && git clone -q https://github.com/modelmiser/warp-types.git 2>/dev/null
         cd /tmp/warp-types && git fetch -q origin && git checkout -q '$ref' && (git pull -q origin '$ref' 2>/dev/null || true)
         bash reproduce/runpod-h200.sh" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || ! grep -q "=== DONE ===" "$log"; then
        echo "=== verification FAILED (ssh exit $rc); partial log in $log; terminating ===" >&2
        exit 1
    fi
    echo "=== verification done; log in $log; terminating ==="
}

need runpodctl; need python3
# runpodctl reads RUNPOD_API_KEY; the key lives in `pass` (runpod/api), never in a file.
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    need pass
    RUNPOD_API_KEY="$(pass show runpod/api | head -1)" || die "pass show runpod/api failed"
    export RUNPOD_API_KEY
fi
case "${1:-status}" in
    up)     cmd_up "${2:-}" ;;
    run)    cmd_run "${2:-}" ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    *)      die "usage: $0 {up|run|down|status} [gpu-id]" ;;
esac
