# Fusion ir

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-106 | P2 | open | Fuse row-wise log-softmax | a fusion catalog entry replaces the nine calls `K_LOG_SOFTMAX` lowers to (`modules/core/graph/include/SushiAI/graph/fusion.hpp`) |  |  | 2026-10-19 |
| RW-107 | P2 | open | Fuse `K_NLL_LOSS` | it lowers to fewer than three calls |  |  | 2026-10-19 |
| RW-108 | P2 | open | Skip the whole step on overflow | one overflowed gradient freezes every parameter for that step, through a task that reads the count buffer and writes a gradient |  |  | 2026-10-19 |
| RW-109 | P2 | open | Add bfloat16 | `MixedPrecisionPolicy` takes bfloat16 |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
