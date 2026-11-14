# General

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-38 | P2 | decision | Choose whether the raw cache needs a tool of its own | the owner has ruled that the dataset build's `write_sample_cache` is the tool, or named a command that writes the cache without starting a run |  |  | 2026-10-25 |
| RW-148 | P3 | decision | Decide whether sa test --label takes the shared labels instead of the suite names | the owner rules whether gating, all and package stay suite names or become --label small/medium/large with --all and a package flag |  |  | 2026-11-01 |
| RW-167 | P2 | open | Close the readiness list and cut 0.1.0 | Every item of the release readiness list is closed and the 0.1.0 tag is cut |  | `docs/design/PROGRAMME/README.md` | 2026-11-11 |
| RW-172 | P1 | decision | Release a SushiRuntime that carries in_process_transport for the distributed layer | A SushiRuntime tag holds SushiRuntime/in_process_transport/in_process_transport.hpp and sa build --distributed passes against the pin |  | `cli/sushistack.deps.toml` | 2026-11-14 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
| KI-18 | dependency | sa build links a sibling checkout over the pinned tag, so it can build an old library | Point SUSHI<NAME>_DIR at a clean clone of the pinned tag, under sources/<name>/<version> |  | `cli/sushistack.deps.toml` | 2026-11-14 |
