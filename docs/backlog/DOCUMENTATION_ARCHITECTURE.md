# Documentation architecture

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-131 | P2 | open | Write the About line of each design topic by hand | each topic has an `**About:**` line of one sentence written for a cold reader |  |  | 2026-10-22 |
| RW-140 | P2 | open | Route Darknet's assignment through `AssignmentRule` | `head_delta.cpp`'s assignment is a `best_shape` type behind `AssignmentRule`, the delta writing reads its records, and the Darknet parity suites pass unchanged |  | `docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/REPORT.md` | 2026-10-25 |
| RW-141 | P2 | unmeasured | Refuse `if_absent: "local"` under `deterministic: true` | a training file whose cluster section lets a replica fall back to the master is refused while `deterministic` is true, naming both keys, and the tests and… |  | `docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/REPORT.md` | 2026-10-25 |
| RW-143 | P2 | open | Read the anchor-free loss terms from the step | the anchor-free step reports its box, class, distribution and L1 terms from the graph's own values, and the host evaluation runs only in tests |  | `docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/REPORT.md` | 2026-10-26 |
| RW-144 | P2 | unmeasured | Share the training loop and step checks of the two detection engines | the anchor and anchor-free loops share one run check and progress report, and the two steps one truth check |  | `docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/REPORT.md` | 2026-10-26 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
