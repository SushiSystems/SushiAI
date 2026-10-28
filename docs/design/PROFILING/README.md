# Profiling

**Status:** Open — P1 to P5 planned, code not yet written; P6 waits on their build
**About:** How SushiAI measures where a training step and an inference batch spend their time, through SushiProfiler alone, down to the IR node each kernel belongs to.
**Modules:** modules/core/eager, modules/layers/train, modules/vision/training

## Problem

A YOLO training step on the RTX 3080 Ti takes about 5 s with the GPU busy 20 to 46 % of it, and
nothing in SushiAI says where the rest goes. The trainer sums per-node device times from
SushiRuntime's `RunReport`, which does not carry them, and only when the caller runs
`Trainer::fit` with `profile` set; the programs call `run_step` and never reach it. The eager
session times its own stages with a private stopwatch. No host phase of a step is timed, and no
kernel can be traced back to the IR node that emitted it.

## Decision

The owner, 2026-10-26: everything that profiles leaves the libraries for SushiProfiler, and the
engineering decisions are the orchestrator's. The rulings:

| Subject | Ruling |
| --- | --- |
| One profiler | SushiAI keeps no timing of its own: `TrainerOptions::profile`, `RunReport::node_timings`, `RunReport::duration_ms` and the eager session's stage times go; zones replace them |
| Where SushiProfiler is included | One module, `modules/core/profile`, includes it; the `SA_ZONE` family forwards to `SP_ZONE` under `SA_PROFILE_LEVEL` and compiles away under `SA_ENABLE_PROFILER` off |
| Build time | `<SushiAI/profile/profile.hpp>` is included by source files only, never by a public header |
| Periods | One period per training step on the timeline `train.step`, one per inference batch on `inference.batch` |
| Kernel to IR node | Each node lowering records carries the origin `<node id>:<op name>`, set through SushiBLAS's `TaskRecorder::origin` scope; each compiled step labels its graph with its role through `Graph::set_label` |
| The session | Lives in the programs, not in the library: `--profile <config.json>` on `sushiai_train` and `sushiai_inference`, forwarded by `sa train --profile` and `sa inference --profile`; absent, no session opens and every gate stays closed |
| Reading records | `sa profile <args>` forwards to SushiProfiler's `sp` |
| Dependencies | SushiProfiler v0.2.0, SushiRuntime and SushiBLAS are consumed as prebuilt packages through `find_package(<Name> CONFIG REQUIRED)` |

## Design

The work folder `docs/agent/2026_10_26_PROFILER_ADOPTION/` holds the interfaces and the task list.
The stack-wide design of SushiProfiler and its zone groups is SushiProfiler's
`docs/design/PROFILER/README.md`.

A record of one profiled run answers three questions:

1. How a step splits into data, host, copy, device, optimizer and evaluation time: the zones of
   each period, read by `breakdown`.
2. Which kernels bound the device time, and which IR node each belongs to: the device spans of
   `sushiruntime.device`, grouped by their `graph` and `origin` attributes, read by `roofline`.
3. What the host was doing while the GPU idled: `device_idle`, which names the innermost zone open
   on the submitting thread over each gap.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| P1 | The `profile` module, its options and the SushiProfiler package | planned |
| P2 | Zones and periods in training, inference, compilation and the eager session; the library's own timing removed | planned |
| P3 | Graph labels and node origins from lowering | planned |
| P4 | The programs' `--profile` session and `sa profile` | planned |
| P5 | SushiRuntime and SushiBLAS consumed as packages | planned |
| P6 | Where a YOLOv5n step goes, written up from two records | planned |

## Open questions

None.
