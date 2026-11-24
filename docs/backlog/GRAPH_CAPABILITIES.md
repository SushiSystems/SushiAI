# Graph capabilities

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-120 | P2 | open | Carry out the SushiAI items of the graph capabilities plan | G2 to G5, G7 to G12, A2, A5, A6 and A12 of section 5 are closed |  | `docs/design/GRAPH_CAPABILITIES/README.md` | 2026-10-19 |
| RW-121 | P2 | unmeasured | Build on SushiBLAS's views, enums and `BLASScalar` | `sa build` of this tree is clean against SushiBLAS `main` at 802e59d or later |  |  | 2026-10-25 |
| RW-187 | P2 | blocked | Lower through SushiBLAS's public broadcast call, not Dispatch | No SushiAI source names SushiBLAS::Dispatch, and SushiAI pins the SushiBLAS tag that adds the call | external: SushiBLAS RW-163 | `modules/core/graph/source/lowering/loss.cpp` | 2026-11-17 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
