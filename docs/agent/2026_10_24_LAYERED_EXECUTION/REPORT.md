# Layered execution in SushiAI: report

**Status:** Open — A0 to A4 landed on 2026-10-24; not built yet.

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

## A3, in-place sum

The exchange sums each filled type segment of the replicas' result packs into a master sum pack
through SushiBLAS's B16 segment views, one `ordered_sum()` per segment, and scatters the sum pack
into the gradients once; the master copies and their N scatters are gone. Each element is still
the sum of the same N inputs in the same order with the same weight and accumulation type, since
`ordered_sum()` never reduces across elements, so the bytes are those of the previous path.

## A4

`parse_train` keeps the cluster section as an opaque value, as it keeps `schedule`, and the
program hands it to `ClusterSection::read`: `core/config` cannot depend on `distribution`, which
a build without the flag does not have. The training file is version 2; version 1 still reads and
refuses a `cluster` key as an unknown one. `ReplicaModel` is implemented in the programs, since
putting distribution into `layers/train` would make that module conditional. The cluster run
links in through `cluster_training.cpp` or `cluster_training_none.cpp`, which names the flag,
so no source holds an `#if`. `sushiai_worker <train.json> [SushiRuntime worker flags]` hands the
flags to SushiRuntime's own parser; `--name` picks the replica. The master's runtime takes the
name of replica 0's machine, so replica 0 stays on the master; the master waits 60 s for its
workers and then lets each rule's absent policy decide.

The acceptance test runs the master's step and a worker's served kernel over SushiRuntime's
`InProcessFabric`: one compile, one step, one offload, the parameters changed. It does not reach
the TCP attach or `run_worker`; the repository has no child-process helper. A4b fixes what the
first round left: each replica reads the whole batch instead of its rows, the master plans
activation memory for rows it never runs, the FNV-1a constants exist twice, and `TrainingSetup`
exposes its members.

## A4b

`Dataset::read_rows(index, rows, batch)` is pure virtual: a default would read the whole batch
again, the cost this removes. The kinds that hold their batches slice views; the reader, table and
token datasets build only the rows asked for, and their `read()` is `read_rows` over the whole
batch, so the two reads are one code path. A replica reads its micro-batch's rows at each feed, so
over a step it reads exactly `[offset_k, offset_k + r_k)` and holds `m` rows, not `B`. The master
of a cluster run builds its setup in the `Results` scope, through `Train::bind_step_results`: the
state, a gradient per parameter and the loss, with no memory plan, no leaves and no host tables.
The FNV-1a is one header-only `hash` module with the values unchanged, pinned by published vectors
and by the model fingerprint computed outside SushiAI. `TrainingSetup` has an invariant, its pool
bound to its graph and state, so it hides its members behind accessors.

Seen and kept: `Dataset` grew a second pure virtual, which all seven kinds and eight test fakes
implement; `holds_rows_of` exists in three test programs that share no support header; three more
private FNV-1a copies (`region_signature.cpp`, `sample_cache.cpp`, `Shape::hash()`) moved to
`hash` in A4c, each the same algorithm, so every value stays; the sample cache's hash names its
files on disk, so existing caches stay found. Each site pins a value computed outside SushiAI.

## A9, the branch review

Two reviews found nineteen points. The one a user would meet first: a worker whose training file
differed from the master's trained other rows than the master weighed, since the replica
fingerprint held only the architecture and the objective type. It now folds the architecture's
bytes, the objective and data settings, the seed, the run index, the replica count and each
replica's rows and micro-batch, and leaves out what only names where output goes. Every host copy
of a pack asserts the tensor is host-visible, which holds because SushiAI's storage, the pack and
the worker's borrowed views are all shared memory. `SA_DISTRIBUTED` is a public definition of the
distribution module, so an installed consumer sees it. `check_rows` returns a status. The trainer
and the cluster run share one epoch loop; the vision domain keeps its own, which validates. The
configure stops when a sibling's distributed option is set off by hand while SushiAI's is on; a
build tree whose cache holds that option at its default off stops too, until it is set or the tree
is configured afresh.

The orchestrator's ruling on the master's wait was wrong: an all-`fail` section that waits not at
all fails before any worker can attach. A9b waits until every named worker attaches, matched by the HELLO
name the rules target, or until `attach_timeout_ms` passes (60000 by default, raised to the largest
`wait_ms`), with no polling: one snapshot, then a wait for the next attach. A missing worker under
`fail` ends the run naming it. Each SushiAI recording owner (a region emitter, the trailing
region, a compiled step, `record_and_run`) owns one SushiBLAS late-count hold for its recording's
life, and each pack takes its tensors' device. A9c counts and awaits distinct workers, compared as
SushiRuntime's placement compares machine names, leaves out a replica on the master's own machine,
and names each missing worker with the policy of each of its replicas.

## The audit

The cluster section's duplicate check and the served kernel compared worker names exactly, while
placement compares them by their lower-case first label, so two names of one machine passed the
check and a worker started under another case found no replica; both use SushiRuntime's
`machine_names_match`. The audit also found a replica's draws following its own step's run count, which only the
master restores, and the fallback warning logged nowhere. A10 adds a step, counted across epochs,
to the position pack, and the replica runs under the draw index `step × m + i` through
`CompiledStep::run_at()`, so a replica started at any step draws what an uninterrupted run draws
there, and single-machine training draws as before. SushiRuntime exposes no node's placement
outcome, so the master's own replica kernel tells an observer each task it serves; the master warns
once per replica and cause, for a worker missing at attach and for a replica's first step run on
the master.

## A8, SushiBLAS's `UInt32`

SushiBLAS's count type `UInt32` reaches SushiAI's `dtype_size` (4) and `dtype_name`; SushiAI has no
unsigned type of its own, so a tensor of it is sized truthfully and computes nothing. The `.sushi`
format keeps codes 0 to 6 and its version: the reader refuses code 7 in words and the writer
refuses to write it. A model's inputs already allow only float32 and int32.

Seen and kept: `trainer.cpp`'s `current_loss` switches with a `default:` that fails fatally, so a
next type would not warn there at compile time.

## Dispatches

```
dispatch: A0 — sonnet medium, 93 turns, passed, rung 1
dispatch: A1 — sonnet medium, 73 turns, passed, rung 1
dispatch: A0 follow-up — haiku high, 62 turns, passed, rung 1
dispatch: A2 — opus low, 108 turns, passed, rung 1
dispatch: A3 — opus low, 168 turns over two rounds, passed, rung 1
dispatch: A8 — sonnet medium, passed, rung 1
dispatch: A3 in-place sum — opus low, passed, rung 1
dispatch: A4 — opus low, 170 turns, passed, rung 1
dispatch: A4b — opus low, 131 turns, passed, rung 1
dispatch: A4c — sonnet low, 32 turns, passed, rung 1
dispatch: A9 — opus low, two rounds, passed, rung 1
dispatch: A9b — opus low, 107 turns, passed, rung 1
dispatch: A9c — sonnet low, three rounds, passed, rung 1
dispatch: A10 — opus low, passed, rung 1
```
