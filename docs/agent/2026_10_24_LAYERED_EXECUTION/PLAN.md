# Layered execution in SushiAI: plan of the run

**Status:** Open — A0 to A4 landed
**Implements:** LAYERED_EXECUTION#A0, LAYERED_EXECUTION#A1, LAYERED_EXECUTION#A2, LAYERED_EXECUTION#A3, LAYERED_EXECUTION#A4, LAYERED_EXECUTION#A5, LAYERED_EXECUTION#A6, LAYERED_EXECUTION#A7

## How it runs

One orchestrator session runs the four repositories of the stack, from SushiRuntime's work folder
`docs/agent/2026_10_24_LAYERED_EXECUTION/` there. Up to four workers run at once, on tasks whose
files are disjoint. A worker checks its C++ with `clang -fsyntax-only` and builds nothing; the
orchestrator runs the checkers under `tools/` and commits each task. The owner ruled on
2026-10-24 that each repository is built and tested once, at the end of the programme, and that
AdaptiveCpp is checked then too. Work lands on `feature/layered_execution`.

## Tasks

| Id | Waits on | State |
| --- | --- | --- |
| A0 | nothing | built |
| A1 | `sushiruntime` R4 | built |
| A2 | A1, `sushiblas` B12, `sushiruntime` R11 | built |
| A3 | A2, `sushiblas` B11, `sushiruntime` R10, R12 | built |
| A4 | A3 | built; A4b follows |
| A5 | A4, two machines | planned |
| A6 | A5, `sushiruntime` R13, five machines | planned |
| A7 | A6 | planned |
