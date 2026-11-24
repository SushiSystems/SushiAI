# Cleanup: plan

**Status:** Open — written 2026-11-24

One task per item of the orchestrator's brief, done in order.

| Item | Change | Done when |
| --- | --- | --- |
| 1 | MODEL_IMPORT: phase states, the format engine text, `**Modules:**`, the `import weights` line, the EXPORT page and phases E1 to E4 | `sa design owner` returns MODEL_IMPORT for `io`, `interchange` and `vision/importers`, and `sa design check` passes |
| 2 | `region_executor.hpp`: the orphaned `@brief` moves onto `LiveRegion` | `clang -fsyntax-only` passes on a file that includes it |
| 3 | `modules/core/eager/README.md`: state is read with `sa backlog` | The line names the command |
| 4 | `applications/examples/yolov3_tiny/README.md`: lines 67 and 106 wrapped | No line passes 100 columns |
| 5 | EAGER_FRONT_END topic; PROFILING's module line judged against its text | `sa design show EAGER_FRONT_END` prints the topic |
| 6 | Stale backlog entries, each with evidence and the command | Every candidate listed as stale or kept, with why |
