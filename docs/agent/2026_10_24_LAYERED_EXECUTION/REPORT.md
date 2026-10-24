# Layered execution in SushiAI: report

**Status:** Open — A0 to A3 landed on 2026-10-24; not built yet.

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

## A1

`modules/core/distribution/` builds under `SA_ENABLE_DISTRIBUTED`, which also defines
`SA_DISTRIBUTED`; `SushiAI.hpp` includes its two headers under that macro, and the umbrella names
the module only when it is built (the owner's ruling of 2026-10-24). The spec's `cluster` section
holds `listen`, `secret_file` and `replicas`, each replica a `worker`, `rows`, `micro_batch`,
`if_absent`, an optional `wait_ms` and `priority`. `if_absent` has no default, so a run cannot
lose its repeatability by omission; the owner ruled it required, and ruled that the `cluster` key
raises `K_TRAIN_VERSION` when `parse_train` accepts it (A4). Replica 0 is the master and its
`worker` names the master's own machine, so no replica is a special case. In batch `b`, replica
`k` takes `r_k` positions of the epoch's order from `b·B + Σ_{j<k} r_j`, micro-batch `i` from
`i·m_k` past that, with weight `r_k / B`. `ReplicaPlan` includes nothing of SushiRuntime, so it
runs on every machine; `ClusterSection` builds one rule per replica, `sushiai.replica` exact with
key `k`, named `replica k`.

Seen and kept: a three-line path resolver duplicates a private helper of `train_config.cpp`;
`secret_file` is resolved but not opened when the section is read.

Owed at A4: the `cluster` row in `docs/guides/RUNNING_MODELS.md`'s table of optional keys.

## A0, follow-up

Once SushiBLAS's `IO` took no engine (its B3), the checkpoint savers lost the `Engine&` they kept
unread; each states as a precondition that the tasks which wrote its state have run.

## A2

The settled shape is in the design's "What A2 settled". The bit-for-bit test runs one micro-batch
(r = m = 4) against a single-machine replay of the same graph on a runtime of the same
configuration. Two micro-batches against one step over four rows split the sum inside the matrix
product, so that case compares within an absolute 1e-6 and a relative 1e-5; a third case matches
`(a + b) / 2` of two single-machine replays bit for bit. The kernel keeps one step per replica it
serves, so the master can serve a lost worker's replica, and rebuilds it on a new seed or run.
`SA_ENABLE_DISTRIBUTED` now turns on SushiRuntime's and SushiBLAS's distributed options too.

Seen and kept: `ReplicaModel` holds the traced model and the feed of its rows, which share the
pool's life; the module reaches `Detail::create_runtime()` of `execution`, another module's
internal surface; the micro-batch mean assumes an objective that is a mean over its rows.

## A3

The settled shape is in the design's "What A3 settled". A first round put the epoch and batch in
both the node's parameters and a position pack, and gave `StepRecorder` a second hook that
`GradientExchange` had to refuse; the orchestrator ruled one source for the position (the pack)
and one recording context (`StepRecording`) for every recorder, and the second round did both.
Every existing recorder reads only the task recorder it read before, so each behaves as it did.

The sum scatters each result pack into master copies and runs one `ordered_sum()` per target:
N×P extra copies per step, which SushiBLAS's B16 views remove in the follow-up. Two replicas
equal one step over the whole batch within an absolute 1e-6 and a relative 1e-5 in float32, as
A2's two micro-batches; two pinned runs equal each other bit for bit; every type crosses with
exact dyadic values. `execution.hpp` named `TaskRecorder` without including it since A0, which
broke `optimizer.cpp`; the include is added.

## Dispatches

```
dispatch: A0 — sonnet medium, 93 turns, passed, rung 1
dispatch: A1 — sonnet medium, 73 turns, passed, rung 1
dispatch: A0 follow-up — haiku high, 62 turns, passed, rung 1
dispatch: A2 — opus low, 108 turns, passed, rung 1
dispatch: A3 — opus low, 168 turns over two rounds, passed, rung 1
```
