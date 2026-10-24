# Layered execution in SushiAI: report

**Status:** Open — A0 landed on 2026-10-24; not built yet.

## A0

`record_and_run(engine, record)` keeps its name and now builds a graph from the engine's runtime,
a `TaskRecorder` over it with the engine's scratch pool and mode, calls `record` with that
recorder and runs the graph; a failed recording needs no discard, since its graph dies with its
tasks. `create_graph()` and `run_once()` join it; `run_recorded()`, `discard_recorded()` and
`abandon_recording()` are gone. The initializers of `nn` take a `RandomOps&`. The checkpoint
savers rely on the caller having run the graph that wrote what they save, and keep an unread
`Engine&` until SushiBLAS's B3 takes the engine out of `IO`. Tests record on `TestGraph`, a
fixture in `modules/core/engine_fixture`.

The grep of every engine-graph call and every family built over `Engine&` gave 53 hits before
and none after; 84 `context.recorder()` calls of the lowering already recorded on an owned graph.

Seen and kept: `record_and_run()` takes the whole `Engine&` for its runtime, scratch pool and
mode, a wider dependency than it needs; whether a dataset's `read` may call it behind a loader
was not checked.

Verification: `clang++ -fsyntax-only` of 12 sources and 11 tests, with include roots rebuilt by
hand because `build/compile_commands.json` predates the module layout; the checkers under
`tools/` pass. `checkpoint.hpp`, `run_checkpoint.hpp` and the other users of `init.hpp` were
not checked. Not built.

## Dispatches

```
dispatch: A0 — sonnet medium, 93 turns, passed, rung 1
```
