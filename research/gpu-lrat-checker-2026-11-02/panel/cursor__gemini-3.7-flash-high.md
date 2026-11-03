# cursor:gemini-3.7-flash-high

Q1: CONFIRMED — Enforced at `lrat.rs:414-416,426,500-511` (non-unit and missing conflict), `lrat.rs:407,493-497,505-508` (forward and deleted hints), `lrat.rs:516-518,522` (empty clause required for verification), and `lrat.rs:468-473` (missing deletion id) — none

Q2: CONFIRMED — Rejected at `lrat.rs:484-489` (`check_sequential`), `lrat.rs:567-572` (`prepass`), `lrat_warp.rs:88,130-135` (`flatten`), and `lrat_warp.rs:308-311` (`from_binary`); `FlatProof` and device kernel cannot receive negative hints — none

Q3: CONFIRMED — `parse_binary` follows standard encoding at `lrat.rs:154-233`; truncation/varint overflow/id 0 handled safely at `lrat.rs:160-170` and `lrat_warp.rs:252-324`; structural equality of `from_binary` and `flatten(parse_text(...))` is explicitly asserted in tests at `lrat_warp.rs:786-809,859-874` — none

Q4: CONFIRMED — Deleted hints require `deleted_at > lemma_step` across all four (`lrat.rs:468,610-614`, `lrat_warp.rs:499`, `sat-kernels/src/lib.rs:423`); satisfied hints with $\ge 2$ unassigned/true literals fail unit check identically (`lrat.rs:410-418`, `lrat_warp.rs:509-516`, `sat-kernels/src/lib.rs:436-449`); empty clause required by `combine()` and CPU paths (`lrat.rs:516-522,662-668`, `lrat_warp.rs:634-642`) — none

Q5: CONFIRMED — Table overflow ($\ge 4095$ probes) unconditionally sets code 5 at `sat-kernels/src/lib.rs:366,409,464`; monotonic stamp $i+1$ prevents stale collisions with `stamps[idx] != stamp || keys[idx] == 0` at `sat-kernels/src/lib.rs:338,356`; lane reads only access identical broadcast/self-inserted keys with uniform branch convergence at `sat-kernels/src/lib.rs:406-466` — none

Q6: CONFIRMED — External checker harness actually executes `lrat-check` and `cake_lpr` binaries and asserts verdict parity at `lrat.rs:859-927` and `tools/cert-campaign.py:10-15`; 33 corrupted proofs test flipped literals, dropped final clauses, and deleted original clauses (`research/gpu-lrat-checker.md:92`) — none

Q7: CONFIRMED — Measured span in `lrat_check.rs:87-89` wraps `GpuContext::check_lrat_codes` (`gpu_launcher.rs:420-451`), which includes all host-to-device uploads, kernel execution, and code download, matching 0.278 s in `h200-session2-results.txt:44-48`; 41.7 s `lrat-check` ran on the 2.06 GB text version (`gpu-lrat-checker.md:96-111`); arithmetic $(3.41 + 7.3) / (3.41 + 0.28) \approx 2.9\times$ and $(3.4 + 7.3) / 4.1 \approx 2.6\times \approx 2.7\times$ matches log — none

Q8: CONFIRMED — From knowledge: prior published work (e.g. Osama & Wijs, FMSD 2023) generates clausal proofs on GPU but verifies on CPU; distributed checkers like PalRUP (SAT 2024/2026) target multi-core CPU clusters, not GPUs — none

Q9: CONFIRMED — `warp-types-sat/README.md:214` claims "runs unchanged on SimWarp and on the device" (they are distinct implementations sharing algorithm structure) and line 217 omits "text form" when comparing to `lrat-check` — Replace `warp-types-sat/README.md:214,217` with: "the warp version checks one lemma per warp with `ballot` / `reduce_sum` / `broadcast_lane`, implemented on `SimWarp` and as an nvptx device kernel" and "6–10× `lrat-check` on the text form of the same proof"

### Unasked Findings (Severity Order)

1. **Table overflow maps to `Reason::NoConflict` instead of distinct error:** In `lrat_warp.rs:624`, `combine()` maps `LemmaCode::TableOverflow` to `Reason::NoConflict`. A proof exceeding the 4096 literal capacity is reported as invalid proof logic rather than capacity exhaustion.
2. **Dense ID indexing allocation hazard in `from_binary`:** `FlatProof::from_binary` (`lrat_warp.rs:245,284`) resizes `dense_of: Vec<u32>` directly to `id as usize + 1`. Sparse large clause IDs will trigger massive memory allocation or panic.
3. **Early exit ignores trailing corruptions:** `check_sequential` (`lrat.rs:517`) and `from_binary` (`lrat_warp.rs:320`) stop immediately at the first empty clause; trailing corrupted steps in the file are uninspected.

### Summary Table

| Q | Verdict | One-Line Reason | file:line |
|---|---|---|---|
| Q1 | CONFIRMED | Sequential checker strictly rejects non-unit, forward, deleted hints, non-existent deletions, and missing empty clause | `lrat.rs:407-426,468-473,500-522` |
| Q2 | CONFIRMED | Negative hints rejected at parse/flatten/prepass; cannot verify on any path | `lrat.rs:484-489,567-572`, `lrat_warp.rs:88,130-135,308-311` |
| Q3 | CONFIRMED | Binary parser handles varints/errors safely; flat decoder equality tested against AST flatten | `lrat.rs:154-233`, `lrat_warp.rs:252-324,786-809,859-874` |
| Q4 | CONFIRMED | All 4 implementations agree on strictly later deletions, satisfied hint rejection, and empty clause | `lrat.rs:400-522,609-668`, `lrat_warp.rs:496-643`, `sat-kernels/src/lib.rs:420-449` |
| Q5 | CONFIRMED | 4096-table overflow sets code 5; monotonic stamp prevents collisions; no warp divergence | `sat-kernels/src/lib.rs:338,356,366,406-466` |
| Q6 | CONFIRMED | Test harness executes external `lrat-check` and `cake_lpr` on 147 genuine and 33 corrupted proofs | `lrat.rs:859-927`, `tools/cert-campaign.py:10-15` |
| Q7 | CONFIRMED | Timed block includes full HtoD/DtoH; matches 0.278 s on 941 MB proof; 41.7 s baseline is text form | `gpu_launcher.rs:420-451`, `examples/lrat_check.rs:87-89`, `h200-session2-results.txt:44-48` |
| Q8 | CONFIRMED | Prior literature (e.g. Osama & Wijs, PalRUP) generates proofs on GPU or checks on CPU clusters | From knowledge |
| Q9 | CONFIRMED | README claims "runs unchanged" (separate implementations) and omits "text form" on `lrat-check` | `warp-types-sat/README.md:214,217` |

