# Graph capabilities
**Status:** Open — phase 0 of 4; plan written and its decisions ruled, nothing implemented

The changes below SushiAI's layers tier that lift the limits found while the language, dataset,
model-import and Darknet work was written in October 2026. The session that owns SushiRuntime
and SushiBLAS carries out sections 3 and 4; the SushiAI session carries out section 5.

How to read a path: one that starts with `sushiblas/` or `sushiruntime/` is in that sibling
checkout, and every other path is in this repository. A path under a heading "Adds" does not
exist yet. Line numbers are those of SushiAI `bc0ef19`, SushiBLAS `244fa5b` and SushiRuntime
`ae9ace3`, each with its uncommitted work of 2026-10-11.

Nothing in this plan was built or measured: builds are closed until SushiBLAS follows
SushiRuntime v0.4.0. Every cost stated here is read from the code or is an estimate, and each
estimate names the benchmark that would replace it.

## 1. The items

Owners: **R** is SushiRuntime, **B** is SushiBLAS, **G** is SushiAI's graph
(the `core` tier: `modules/core/graph`, `modules/core/autograd` and the modules beneath them), **A** is SushiAI above
the graph. Size counts units, a unit being a file added or an existing file edited, tests
included: small is 1 to 3, medium 4 to 8, large more than 8.

| Id | Limit | Owner | Unblocks in SushiAI | Size | Depends on |
| --- | --- | --- | --- | --- | --- |
| R1 | Host-written memory read by a replayed plan is relied on and stated nowhere | R | G1, and what the gather offsets, the mask vectors and `StepScalars` already do | small | none |
| R2 | A run over several machines | R | Moved to [`LAYERED_EXECUTION.md`](LAYERED_EXECUTION.md) | not sized | that design's phase 0 |
| B1 | A random fill fixes its seed and offset when recorded | B | Dropout (G1) | medium | R1 |
| B2 | Row gather goes through a sparse product: int32 `r * V + label`, no bounds rule, host-filled offsets | B | Losses over integer labels of any size (G5), the embedding | large | none |
| B3 | No broadcasting: operands of one elementwise call share one shape | B | Broadcast operations in the IR (G3), shorter special forms (G4) | large | none |
| B4 | Upsampling is nearest only | B | A bilinear segmentation head (G7) | medium | none |
| B5 | No index of the maximum along an axis | B | Segmentation metrics without `R * C` host reads (G10) | small | none |
| B6 | GELU is the tanh approximation only | B | A model trained with the error-function form | small | none |
| B7 | Tensor creation under concurrent calls is not stated or tested | B | The loader on threads (A8) | small | none |
| B8 | Host access to a Shared tensor between executions is not stated or tested | B | The same as R1, at the level SushiAI calls | small | R1 |
| B9 | No mask of the k largest of each row | B | The gate of a mixture in one node (G11) | small | none |
| B10 | No index of the rows a mask selects | B | An expert run on its own rows (G12) | medium | B2 |
| G1 | A lowering row cannot run host work before each replay; the IR has no random operation. Written with a host-drawn mask; the device fill waits on B1 | G | `NN::Dropout` (A11) | large | B1 |
| G2 | No square root, logarithm or power in the IR | G | Compositions that need them; a reciprocal root is a power | medium | none; now |
| G3 | `add`, `sub`, `mul`, `div` take equal shapes | G | The position table, a fed mask, the key and value cache's scores | medium | B3 |
| G4 | Special forms spell a broadcast with ones vectors, rank-4 views and `[N, C]` scratch | G | Fewer tasks in layer normalisation, log-softmax and its gradient | medium | B3, a measurement |
| G5 | A loss over labels builds `r * V + label`; `N * T * V` and `N * H * W * C` must fit in int32 | G | Next-token and per-pixel cross entropy at any batch size | large | B2 |
| G6 | `K_SCATTER_ADD_ROWS` has no backward rule | G | Dice without the `R * C` one-hot leaf | medium | none; now |
| G7 | `K_UPSAMPLE2D` is nearest | G | `SegmentHead` with class boundaries finer than its scale | medium | B4 |
| G8 | The masked score becomes minus infinity in half precision | G | Attention under `MixedPrecisionPolicy` | small | none; now |
| G9 | Whether a step made of recorders alone compiles is not pinned | G | The update step of gradient accumulation (A2) | small | none; now |
| G10 | No argmax in the IR | G | `measure_segmentation` reading `R` indices | medium | B5 |
| G11 | No top-k or row maximum in the IR; spelled as a pooling window, an upsample, a comparison and a select, 4 nodes a level | G | `MoEFeedForward`'s gate in one node | medium | B9 |
| G12 | The rows of an expert and their count come from the data, and a graph is traced for fixed shapes | G | An expert that runs on a bounded count of rows | large | B10, B2 |
| A1 | The key and value cache: one fed leaf, buffer updates in training only, one trace form | A (`train`, `nn`, `language/blocks`) | Written since, on a fed `[1, T]` mask leaf and without G3; a slide still rebuilds it | done | none |
| A2 | Every replay rewrites the gradients and the update sits behind backward | A (`optim`, `train`, `config`) | `subdivisions`, `max_batches` | medium | G9 |
| A3 | The optimizer's step count was not restorable | A (`optim`, `io`) | Written since, as the run-state section of version 2 of the weight file | done | none |
| A4 | The whole-document JSON parser needed about 8 GB for a 450 MB COCO file, by estimate | A (`config`, `vision/datasets`) | Written since, as `Config::read_json_events` | done | none |
| A5 | No check of a token id against the table before a step | A (`language/training`) | A named error where B2 gives a defined zero | small | none |
| A6 | `Dataset::read` has no epoch argument | A (`data`) | Window starts redrawn at each epoch | medium | none |
| A7 | A training file named one objective | A (`train`) | Written since, as the `sum` objective | done | none |
| A8 | The loader on threads | A (`data`) | Written since, as `Data::ThreadedLoader`; B7's test still confirms what it relies on | done | B7 |
| A9 | `ParameterInit` had no normal initialiser of fixed deviation | A (`nn`) | Written since, as `ParameterInit::normal`; `Embedding` starts at 0.02 | done | none |
| A10 | The clip squares each norm, so a norm above the square root of the largest float gives a factor of 0 | A (`optim`) | Clipping of an exploded gradient | small | none |
| A11 | `Dropout` passes its input through. Written | A (`nn`) | A model that states a rate | small | G1 |
| A12 | Objectives, heads and metrics still use the forms G5, G6, G7 and G10 replace | A (`language/training`, `vision/training`, `vision/blocks`) | The limits above, in the files a user trains from | medium | G5, G6, G7, G10 |

How the reported limits map to items: dropout is B1, G1 and A11. Broadcasting and the missing
math are B3, G2, G3 and G4. Bilinear upsampling is B4 and G7. The row gather's index width,
its bounds and its host-filled offsets are B2 and G5; no device iota is asked for, because the
offsets exist only to build the sparse matrix B2 replaces. The scatter-add rule is G6. The
`[N, C]` scratch of the log-softmax gradient goes with G4. The key and value cache is A1, with
G3 for its mask. Gradient accumulation is G9 and A2 and needs no kernel: `Level1::axpy` is the
accumulate kernel (`sushiblas/modules/operations/blas/include/SushiBLAS/blas/level1.hpp:47`).
The masked score is G8. Concurrent allocation is B7, and the code already answers it (section
4, B7). Host-filled scratch is R1 and B8. The step count is A3, the JSON parser A4, several
machines R2.

Found in the source documents beyond that list: B5 with G10
(`modules/vision/training/README.md:190-195`), B6 (`docs/design/LANGUAGE.md:79`), A6
(`docs/design/LANGUAGE.md:90-91`), A7 (`docs/design/DATASETS.md:159-161`), A9
(`docs/design/REMAINING_WORK.md:255-256`) and A10 (`modules/core/optim/README.md:185-186`).

Added on 2026-10-11 by the mixture of experts, after this plan was handed over, and not yet
sectioned below: B9 as `TaskHandle ReductionOps::top_k_mask_along(const Tensor& t, int64_t k,
Tensor& mask)` over a rank-2 `t`, the mask of `t`'s type holding 1 at the k largest of each
row, the lowest index winning a tie; B10 as `TaskHandle IndexOps::select_rows(const Tensor&
mask, Tensor& index, Tensor& count)`, `index` an INT32 `[C]` of the first C selected row
numbers ascending with the row count in unused slots. `ReductionOps::max_along` exists
(`sushiblas/modules/operations/reductions/include/SushiBLAS/reductions/reductions.hpp:158`);
a top-k, a sort and a compaction do not.

## 2. Phases

No item changes an existing signature of SushiBLAS or SushiRuntime, so every item of R and B
can be written beside the v0.4.0 migration of SushiBLAS and none edits a file the migration
moves, apart from the one-line registrations each section names. An item marked **free** can be
scheduled at any time around the migration; one marked **after** needs the migrated tree to
build before its tests can run.

| Phase | Items | Closes with, in the owning repository | SushiAI then checks |
| --- | --- | --- | --- |
| 1 | R1 (free); B7, B8 (after); G2, G6, G8, G9 (written now) | `sr build` and `sr test` with the R1 test; `sb build` and `sb test --suite unit` with the B7 and B8 tests | `sa build`, `sa test --suite unit`: the gradient checks of G2 and G6, the half-precision mask test of G8, the recorder-only step of G9 |
| 2 | B1, B4, B5, B6 (after); then G1, G7, G10, A11 | `sb build`, `sb test`, and a tagged SushiBLAS release | The pin in `cli/sushistack.deps.toml` moves to that tag; a dropout step draws two different masks from one compile; a bilinear head passes the one-batch overfit test of segmentation; `measure_segmentation` equals its host form |
| 3 | B2, B3 (after); then G5, G3, G4, A12 | `sb build`, `sb test`, the two benchmarks of B2 and B3 run on the RTX 3080 Ti, and a tagged release | The next-token and per-pixel losses and gradients equal their current values within the tolerances of their tests, with the int32 refusals removed; each re-lowering of G4 lands only with its before and after timing |
| 4 | R2 | Phases of [`LAYERED_EXECUTION.md`](LAYERED_EXECUTION.md) | That design's section 7 |

Phases 2 and 3 do not depend on each other and may swap. Inside a phase the B items are
independent of one another. Rule 4 of the `api-stability` skill holds: SushiAI consumes a phase
only from a SushiBLAS tag, never from a branch.

## 3. SushiRuntime

### R1. State and test that a replayed plan reads memory the host wrote between executions

**Limit and evidence.** SushiAI writes Shared USM from the host with no task and expects the
kernels of the next execution to read it:

- the row offsets of a row selection, once at lowering
  (`modules/core/graph/source/lowering/linalg.cpp:97-99`);
- the two `[T * T]` vectors of the causal mask, once at lowering
  (`modules/core/graph/source/lowering/linalg.cpp:148-158`);
- the optimizer's scalars, before every replay (`modules/core/optim/source/optimizer.cpp:174-177`,
  `modules/core/graph/source/compiled_step.cpp:116-120`).

The memory is Shared: `TensorFactory::create` defaults to `Residency::Shared`
(`sushiblas/modules/data/tensor/include/SushiBLAS/tensor/tensor_factory.hpp:103-107`) and
`TensorPool::create` passes no residency (`modules/core/graph/source/tensor_pool.cpp:63-66`).
`sushiruntime/docs/design/GRAPH_REPLAY.md` section D designs the same pattern for late-bound
counts: a kernel reads a host-USM word the step machinery writes before each submission. No
header states it as a contract of `SushiRuntime::Graph` and its plan, and no test replays a plan
around a host write.

**Design.** No code changes. One contract sentence and one test.

The sentence, on `CompiledGraph::rearm` in
`sushiruntime/modules/engine/graph/include/SushiRuntime/graph/compiled_graph.hpp` and in
`sushiruntime/modules/engine/graph/README.md`:

> Between two executions of one compiled plan the caller may write a Shared or Host USM
> allocation that no node of the plan declares as written. Every node of the next execution
> observes the bytes written. The caller writes only while no execution of the plan is in
> flight.

**Semantics.** The rule holds on the scheduler path and on layers 1 and 2 of `GRAPH_REPLAY.md`.
Layer 2 is not built; this item binds it: a recorded command graph fixes a kernel's arguments,
which are addresses, and not the bytes behind them. A Device allocation is outside the rule,
since the host cannot address it. The rule adds no ordering: the host write is not a node, so
the caller orders it by writing between executions.

**Adds.** `sushiruntime/modules/engine/execution/tests/test_replay_host_write.cpp`.

**Must not change.** Every signature of `SushiRuntime::Graph`, its plan and the execution engine.

**Tests.** One kernel node reads a Shared word `w` and writes `out = w + 1`. The plan is
compiled once. For `k` in 0 to 7 the host writes `w = k`, re-arms and executes; `out` is `k + 1`
each time. The case runs on every device queue the context has.

**SushiAI then.** Nothing changes; `modules/core/graph/README.md` cites the contract where it
describes the host-filled scratch.

### R2. Several machines

The design is [`LAYERED_EXECUTION.md`](LAYERED_EXECUTION.md); its section 4 holds what the
runtime has to provide.

## 4. SushiBLAS

Common to every item of this section, unless an item says otherwise:

- Every method returns the `TaskHandle` of the task it recorded and records through
  `TaskRecorder::record_pair`, whose body is "callable more than once"
  (`sushiblas/modules/engine/engine/include/SushiBLAS/engine/task_recorder.hpp:152-158`). A
  task captures addresses and scalars at record time and reads tensor contents at each
  execution, so a replayed plan computes from the current contents.
- An operand has storage, is contiguous and is checked when the call is recorded; a violation
  throws `std::invalid_argument` naming the operation and the operand, and nothing is recorded.
  A device-only operand under `ForceReference` throws `UnsupportedExecution`.
- Each operation has a reference kernel and a SYCL kernel, in the two folders every kernel
  module has, chosen by `KernelPairDispatcher`. No kernel uses `sycl::reduction`, `sycl::atomic_ref` on a
  float or any combination whose order the backend chooses, so a result is the same bytes on
  every run of one device under `SR_DETERMINISTIC_FP`.
- A new unit registers in the tier list of `sushiblas/cmake/SushiBLASLayers.cmake`, in the
  `CMakeLists.txt` of its tier, in the umbrella header of
  `sushiblas/modules/library/sushiblas` and in the tables of `sushiblas/modules/README.md`,
  `sushiblas/modules/operations/README.md` and `sushiblas/modules/kernels/README.md`. Those
  are the only existing files an item edits.
- Under the `api-stability` skill: no existing declaration, operation identity, metadata slot
  or exported symbol changes, so a plan recorded by a consumer today records the same tasks
  afterwards.

### B1. A random fill that reads its stream position when it runs

**Limit and evidence.** `Dispatch::execute_random` reserves one stream position when the call
is recorded and captures the seed and the offset by value
(`sushiblas/modules/operations/random/include/SushiBLAS/random/dispatch.hpp:302-308`).
`CompiledStep` lowers a graph once (`modules/core/graph/source/compiled_step.cpp:62-65`) and
each `run()` re-arms and executes the same plan
(`modules/core/graph/source/compiled_step.cpp:111-127`). A Bernoulli fill in that plan would
draw one mask for the whole run, so `NN::Dropout` emits no node
(`modules/layers/nn/include/SushiAI/nn/dropout.hpp:45-47`, `docs/design/LANGUAGE.md:74-78`).

**Design.** A second view, `PositionedRandomOps`, whose fill takes the position as a tensor
operand and reads it on the device at each execution. The caller writes the position between
executions, on the host, under the contract of R1. This is the shape `Optim::StepScalars`
has: a value that varies between replays lives in memory the kernel reads
(`modules/core/optim/include/SushiAI/optim/step_scalars.hpp`).

Rejected:

- Reserving inside the task body at each execution. Bodies run on whichever worker reaches
  them (`sushiruntime/docs/design/GRAPH_REPLAY.md`, section A), so two fills of one step would
  take their offsets in an order that differs between runs.
- Adding the replay count to the recorded offset. `RNGStream::reserve` hands consecutive
  offsets to consecutive tasks
  (`sushiblas/modules/engine/engine/include/SushiBLAS/engine/rng_stream.hpp:75-81`), so offset
  plus `k` is the position of a later task and one (key, counter) pair would be drawn twice.
- Recording the step again at each replay. That is one compilation per step.
- A mask drawn on the host and fed as a leaf. It moves one element per activation across the
  host boundary at every step.

**Signature.**

```cpp
// <SushiBLAS/random/rng_position_tensor.hpp>
namespace SushiBLAS
{
    /** @brief Holds the element count of a position tensor: the seed, then the offset. */
    inline constexpr int64_t K_RNG_POSITION_ELEMENTS = 2;

    /**
     * @brief Writes @p value into @p position on the calling thread.
     * @param position Rank-1 INT64 of K_RNG_POSITION_ELEMENTS elements, host-addressable.
     * @pre No recorded task that reads @p position is pending or running.
     * @throws std::invalid_argument on a rank, dtype, extent or residency violation.
     */
    void write_rng_position(Tensor& position, RNGPosition value);
} // namespace SushiBLAS

// <SushiBLAS/random/positioned_dispatch.hpp>
namespace SushiBLAS::Dispatch
{
    /**
     * @brief Records a random fill that reads its seed and offset from @p position when run.
     * @tparam Func `(T, const RandomFill&, const std::uint64_t* position, int64_t size, T*)
     *         -> sycl::event`; the draw reads position[0] and position[1] inside the kernel.
     */
    template<Domain Dom, typename Func>
    TaskHandle execute_random_at(TaskRecorder& recorder, const Tensor& position, Tensor& t,
                                 const char* name, SushiRuntime::OperationID op_id,
                                 std::initializer_list<double> params, Func&& task_func);
} // namespace SushiBLAS::Dispatch

// <SushiBLAS/random/positioned_random.hpp>
namespace SushiBLAS
{
    /** @brief Records random fills whose stream position is an operand. */
    class PositionedRandomOps final
    {
        public:
            explicit PositionedRandomOps(TaskRecorder& recorder) noexcept;
            explicit PositionedRandomOps(Engine& engine) noexcept;

            /**
             * @brief Fills @p t from a Bernoulli distribution at the position @p position holds.
             * @param t FLOAT32 or FLOAT64, contiguous.
             * @param p Probability of a 1; within [0, 1].
             * @param position Rank-1 INT64 of K_RNG_POSITION_ELEMENTS elements.
             */
            TaskHandle bernoulli(Tensor& t, double p, const Tensor& position);
    };
} // namespace SushiBLAS
```

**Semantics.**

- Value: with `position` holding `{seed, offset}`, `t` is byte for byte what
  `RandomOps::bernoulli(t, p)` writes when recorded at that stream position
  (`sushiblas/modules/operations/random/source/bernoulli.cpp:38-62`).
- Dtypes: `t` as `RandomOps::bernoulli`, HALF refused by `Sampling::DrawsValues`. `position`
  is INT64 and its two elements are read as raw `uint64_t` bits, as the metadata slots are
  (`sushiblas/modules/operations/random/include/SushiBLAS/random/dispatch.hpp:57-64`).
- Shapes: `t` of any rank. `position` may be a view into a larger allocation.
- Aliasing: `position` may not overlap `t`. The task declares `reads(position).writes(t)`.
- Replay: each execution reads `position` anew. An unchanged position gives the same bytes.
- Metadata: slots `K_RNG_SEED_PARAM` and `K_RNG_OFFSET_PARAM` hold 0, since neither is known
  when the call is recorded; `p` is in slot 0.
- Paths: the reference path and the device path call one draw functor, as `RandomFill` does,
  and agree bit for bit.
- Determinism: the fill is counter-based and accumulates nothing.
- Cost, estimated: two 8-byte loads per Philox block, which is 4 FLOAT32 or 2 FLOAT64
  elements. A `benchmark_random_fill` case of 2^24 elements against `RandomOps::bernoulli`
  replaces the estimate.

`execute_random_at` is general; `bernoulli` is the one public fill, because dropout is the one
evidenced use. A further distribution is one method and one source file.

**Adds.** In `sushiblas/modules/operations/random/`:
`include/SushiBLAS/random/rng_position_tensor.hpp`,
`include/SushiBLAS/random/positioned_dispatch.hpp`,
`include/SushiBLAS/random/positioned_random.hpp`, `source/rng_position_tensor.cpp`,
`source/positioned_bernoulli.cpp`, `tests/test_positioned_bernoulli.cpp`. Operation identity
`"random.bernoulli_at"_op`.

**Must not change.** `RandomOps`, `RNGStream`, `execute_random` and the eight metadata slots.

**Tests.** Equality with `RandomOps::bernoulli` at one position, FLOAT32 and FLOAT64. One
compiled plan executed three times: positions A, B, A give masks `m1`, `m2`, `m1` with
`m1 != m2`. The mean of 2^20 draws within 0.005 of `p`. `ForceReference` equals the device bit
for bit. `p` of 0 and 1. Each refusal: HALF, a FLOAT32 position, three elements, an overlap, a
`p` outside the range.

**SushiAI then.** G1 lowers `K_DROPOUT_MASK` through it in a new
`modules/core/graph/source/lowering/random.cpp`.

### B2. An index family: rows by index, and one element per row

**Limit and evidence.** `GATHER_ROWS` and `SCATTER_ADD_ROWS` lower to `SparseOps::spmm` over a
CSR matrix with a single one per row
(`modules/core/graph/source/lowering/linalg.cpp:285`, `:312`). Four limits follow.

- A loss over integer labels reads one element per row, which the row gather can only spell as
  a table of `R * V` rows of width 1 and the index `r * V + label`. The index is INT32
  (`modules/core/graph/source/lowering/linalg.cpp:197-202`), so the objectives refuse
  `N * T * V` and `N * H * W * C` above 2 147 483 647
  (`modules/language/training/source/next_token_objective.cpp:54`,
  `modules/vision/training/source/pixel_cross_entropy_objective.cpp:48`): 54 images of 512 by
  512 over 150 classes (`modules/vision/training/README.md:221-224`).
- The gradient of that gather is the transposed product with one right-hand column. Its SYCL
  kernel launches one work-item per column, and that work-item zeroes every output and walks
  every row in order
  (`sushiblas/modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/sycl/spmm.hpp:102-126`).
  For 32 windows of 256 tokens over 8192 classes, one work-item writes 67 108 864 elements.
- An index outside the table reads or writes outside it. Only the reference kernel checks, and
  only in a build without `NDEBUG`
  (`sushiblas/modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/reference/spmm.hpp:62`,
  `modules/core/graph/README.md:65-66`).
- The matrix needs `R + 1` row offsets, which the host fills because no task writes them
  (`modules/core/graph/source/lowering/linalg.cpp:94-99`), and `R` ones for its values.

**Design.** A family `IndexOps` in a new module `operations/indexing`, with its kernels in a
new internal module `kernels/index_kernels`. Four operations: rows by index and their sum back
into a table, one element per row and its write back. The index is the label itself, so no
product is built and no width limit follows from the table's size.

Rejected:

- Bounds checks inside `spmm`. A sparse kernel takes its structure on trust by design
  (`sushiblas/modules/kernels/README.md`, under
  `sushiblas/modules/kernels/sparse_kernels/include/SushiBLAS/sparse_kernels/reference/structure.hpp`), and a check would
  be paid by every sparse product of every consumer.
- An INT64 index into the same product. `CsrPattern` reads INT64, so the width limit would go,
  but the `R * V`-row product and the single work-item of its transpose would stay.
- A device iota for the row offsets. It keeps the sparse detour and removes one of its four
  costs.
- A row write `put_rows` for the key and value cache. The cache's alternative is a `[T, dim]`
  copy per buffer per token (`modules/language/generation/README.md:121`): 98 304 elements for
  `T` of 256 and `dim` of 384. That is accepted, so no kernel is asked for.

**Signature.**

```cpp
// <SushiBLAS/indexing/indexing.hpp>
namespace SushiBLAS
{
    /**
     * @class IndexOps
     * @brief Records gathers and their transposes by an integer index.
     * @note An index outside its range selects nothing: a gather reads it as zero and a
     *       transpose skips it, on both paths and in every build type.
     */
    class IndexOps final
    {
        public:
            explicit IndexOps(TaskRecorder& recorder) noexcept;
            explicit IndexOps(Engine& engine) noexcept;

            /** @brief Copies table rows: `rows[r, d] = table[index[r], d]`. */
            TaskHandle take_rows(const Tensor& table, const Tensor& index, Tensor& rows);

            /**
             * @brief Sums rows into a table: `table[v, d] = beta * table[v, d]` plus every
             *        `rows[r, d]` with `index[r] == v`, added in ascending r.
             * @param beta Exactly zero does not read table.
             */
            TaskHandle add_rows(const Tensor& rows, const Tensor& index, Tensor& table,
                                double beta = 0.0);

            /** @brief Picks one element per row: `picked[r] = values[r, index[r]]`. */
            TaskHandle take_along(const Tensor& values, const Tensor& index, Tensor& picked);

            /**
             * @brief Writes one element per row and zero elsewhere:
             *        `values[r, c] = (c == index[r]) ? picked[r] : 0`.
             */
            TaskHandle put_along(const Tensor& picked, const Tensor& index, Tensor& values);
    };
} // namespace SushiBLAS
```

**Semantics.**

| | `take_rows` | `add_rows` | `take_along` | `put_along` |
| --- | --- | --- | --- | --- |
| Shapes | `table [V, D]`, `index [R]`, `rows [R, D]` | the same | `values [R, C]`, `index [R]`, `picked [R]` | the same |
| Element types | `Domain::All`; it moves elements | `Domain::Real` | `Domain::All` | `Domain::All` |
| Index out of range | The row is zeros | The row is skipped | The element is zero | The row is zeros |
| Written | Every element of `rows` | Every element of `table` | Every element of `picked` | Every element of `values` |
| Work-items | One per element of `rows` | One per column `d`, walking `r` upwards, as the transposed `spmm` | One per row | One per element of `values` |
| Accumulates | No | Yes, in `Accumulator<T>`, rounded once on the store | No | No |

- Index: rank 1, INT32 or INT64, zero-based. The range is `[0, V)` for the row operations and
  `[0, C)` for the element operations. A negative index is out of range.
- Rank and layout: the matrices are rank 2 and are addressed through their own strides, so
  both layouts work and a view into an arena works.
- Aliasing: the written operand may not overlap any read operand. `add_rows` declares `table`
  with `updates` when `beta` is not zero and with `writes` when it is.
- Replay: the index is read at each execution, so a leaf refilled between replays is honoured.
- Paths: the three operations that do not accumulate agree bit for bit between the reference
  and the device. `add_rows` sums in one order, ascending `r`, on both; the reference sums in
  `ReferenceSum` as the other reference kernels do, so the two paths agree to the precision of
  the device sum.
- Error contract: rank, dtype, extent and overlap are refused at record time. The content of
  an index cannot be read then; at execution an index out of range has the defined result
  above and reports nothing. A caller that wants a named error checks on the host (A5).
- Cost of the range rule: two integer compares per work-item beside the index load the
  gather needs anyway. Estimated as below the memory traffic of the copy; not measured.

`add_rows` keeps the work-item shape of the transposed `spmm` in this plan. Its two uses after
G5 are the embedding gradient, where `D` is the embedding width and the launch is `D` wide,
and the class sums of Dice, where `D` is 1. A shape that is parallel over `r` (a stable
counting sort of the index, then one fixed-order sum per table row) is a later change, taken
only if `benchmark_index_rows` shows the column walk in a profile of a training step.

**Adds.** `sushiblas/modules/operations/indexing/` with `CMakeLists.txt`, `README.md`,
`include/SushiBLAS/indexing/indexing.hpp`, `source/take_rows.cpp`, `source/add_rows.cpp`,
`source/take_along.cpp`, `source/put_along.cpp` and `tests/test_index_rows.cpp`,
`tests/test_index_along.cpp`. `sushiblas/modules/kernels/index_kernels/` with `CMakeLists.txt`,
`README.md` and, under `include/SushiBLAS/index_kernels/`, `index.hpp`, `reference/rows.hpp`,
`reference/along.hpp`, `sycl/rows.hpp`, `sycl/along.hpp`.
`sushiblas/tests/benchmark/benchmark_index_rows.cpp`. Operation identities `"index.take_rows"_op`,
`"index.add_rows"_op`, `"index.take_along"_op`, `"index.put_along"_op`.

**Must not change.** `SparseOps`, `SparseMatrix`, `CsrPattern` and every sparse kernel.

**Tests.** Each operation against a host loop on INT32 and INT64 indices and on HALF, FLOAT32
and FLOAT64; `take_rows` and the two element operations also on INT32 values. A repeated index
in `add_rows` gives the same bytes on ten runs. `add_rows` with `beta` of 0, 1 and 0.5. An
index of -1 and of `V` in each operation, on both paths, with the neighbours of the affected
row unchanged. `take_along` after `put_along` returns `picked`. A view at a non-zero
`storage_offset`. Each record-time refusal. The benchmark prints, at `V` of 8192, `D` of 384
and `R` of 8192: `take_rows` beside `spmm` without a transpose, `add_rows` beside `spmm` with
one, and `put_along` at `C` of 8192 beside the transposed `spmm` over `R * C` rows of width 1.

**SushiAI then.** G5: `modules/core/graph/source/lowering/linalg.cpp` records `take_rows` and
`add_rows` where it builds a row selection today, and two new operations lower to the element
pair.

### B3. A broadcast seam: one value per row or per column, any functor

**Limit and evidence.** "The operands of one call must share one shape: the library does not
broadcast" (`sushiblas/modules/operations/elementwise/include/SushiBLAS/elementwise/elementwise.hpp:40-41`),
enforced by `require_same_shape`
(`sushiblas/modules/operations/elementwise/include/SushiBLAS/elementwise/dispatch.hpp:117-120`).
SushiAI spells each broadcast another way:

| Where | How it is spelled today |
| --- | --- |
| `BIAS_ADD`, and the position table through it | A copy, then `ger` with a vector of ones (`modules/core/graph/source/lowering/linalg.cpp:238-239`, `modules/language/blocks/source/position_embedding.cpp:95`) |
| `LAYER_NORM` | 14 tasks; a value per row goes through `channel_affine` over a transposed `[1, 1, D, rows]` view, with a vector of ones or zeros for the half of the affine it does not need (`modules/core/graph/source/lowering/norm.cpp:345-361`, `modules/core/graph/README.md:78-82`) |
| `CAUSAL_MASK` | `channel_affine` over `[matrices, 1, 1, T * T]` with two host-filled vectors, computing `x * 0 + fill` (`modules/core/graph/source/lowering/linalg.cpp:143-158`, `:343`) |
| `LOG_SOFTMAX` | Two `ger` calls with ones to subtract a value per row (`modules/core/graph/source/lowering/loss.cpp:75`, `:84`) |
| `LOG_SOFTMAX_BACKWARD` | An `[N, C]` scratch that holds the row sum in every column (`modules/core/graph/source/lowering/loss.cpp:109`, `:120-127`, `modules/core/graph/README.md:105-108`) |

**Design.** One executor of the dispatch seam, `Dispatch::execute_broadcast`, and a view
`BroadcastOps` with the four arithmetic operations built on it. A matrix operand `A` of
`outer * inner` elements meets up to four vectors. A **leading** vector holds one value per
outer index; a **trailing** vector holds one value per inner index. The caller's functor
receives the element of `A` and the selected value of each vector. A leading and a trailing
vector may be mixed in one call, which is what layer normalisation's apply half needs:
`(x - mean[r]) * inv[r] * gamma[d] + beta[d]` is one task.

Every evidenced use is a prefix or a suffix of the matrix's shape: `[B, T, D] + [T, D]`,
`[N, H, W, C] * [C]`, `[rows, D] - [rows]`, `[matrices, T, T]` against `[T, T]`.

Rejected:

- Broadcasting between any two shapes, as NumPy defines it. A broadcast over a middle axis
  needs an index decomposition per element over the rank; no document asks for one.
- Letting `ElementwiseOps::add` accept unequal shapes. It changes a stated contract that every
  consumer's checks rely on, SushiEngine's among them.
- `channel_affine` in place over a transposed view, which exists and would remove the
  `[N, C]` scratch alone. It is a multiply-add only, it needs a ones or a zeros vector for the
  half it does not use, and its contract does not state that `out` may be `in`
  (`sushiblas/modules/operations/spatial/include/SushiBLAS/spatial/spatial.hpp:190-198`).

**Signature.**

```cpp
// <SushiBLAS/broadcast/broadcast_operand.hpp>
namespace SushiBLAS
{
    /** @brief Names which index of the matrix selects a vector's element. */
    enum class BroadcastAxis : std::uint8_t
    {
        Leading, /**< One value per outer index: `v[o]` meets every element of row o. */
        Trailing /**< One value per inner index: `v[n]` meets element n of every row. */
    };

    /** @brief Holds one vector of a broadcast call and the index that selects from it. */
    struct BroadcastOperand
    {
        const Tensor* vector = nullptr;               /**< Never null in a recorded call. */
        BroadcastAxis axis = BroadcastAxis::Trailing; /**< The index that selects. */
    };
} // namespace SushiBLAS

// <SushiBLAS/broadcast/dispatch.hpp>
namespace SushiBLAS::Dispatch
{
    /**
     * @brief Records `C[o, n] = op(A[o, n], v0, ..., vN-1)`, each vk selected by o or by n.
     * @param C May be A itself; any other overlap with an operand is refused.
     * @param op_func Callable `(auto x, auto v0, ..., auto vN-1) -> auto`, or a bind_scalars
     *        bundle of one.
     */
    template<Domain Dom = Domain::Real, std::size_t N, typename Func, typename Param = double>
    TaskHandle execute_broadcast(TaskRecorder& recorder, const Tensor& A,
                                 const std::array<BroadcastOperand, N>& vectors, Tensor& C,
                                 const char* name, SushiRuntime::OperationID op_id,
                                 Func&& op_func, std::initializer_list<Param> params = {});
} // namespace SushiBLAS::Dispatch

// <SushiBLAS/broadcast/broadcast.hpp>
namespace SushiBLAS
{
    /** @brief Records arithmetic between a tensor and a vector broadcast along its rows or columns. */
    class BroadcastOps final
    {
        public:
            explicit BroadcastOps(TaskRecorder& recorder) noexcept;
            explicit BroadcastOps(Engine& engine) noexcept;

            /** @brief Computes `C = A + v`. */
            TaskHandle add(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A - v`. */
            TaskHandle sub(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A * v`. */
            TaskHandle mul(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);

            /** @brief Computes `C = A / v`. */
            TaskHandle div(const Tensor& A, const Tensor& v, Tensor& C, BroadcastAxis axis);
    };
} // namespace SushiBLAS
```

**Semantics.**

- Geometry: `outer` is the element count of the leading vectors and `inner` that of the
  trailing ones; every vector of one kind has the same count. When one kind is absent its
  count is `A.num_elements` divided by the other. `outer * inner` equals `A.num_elements`, and
  `outer` equals a product of leading extents of `A`'s shape. Anything else is refused at
  record time with the shape and the count.
- Rank: `A` and `C` have any rank and one shape. A vector has any rank; only its element count
  and its flat order matter, so a `[T, D]` table is a trailing vector of `[B, T, D]`.
- Layout: `A`, `C` and the vectors are `RowMajor`. A `ColumnMajor` operand is refused by name,
  because its flat order is not prefix by suffix above rank 1.
- Element types: `Domain::Real` by default, one type for every operand. `N` is 1 to 4, by
  `static_assert`.
- Aliasing: `C` may be the same view as `A`; each work-item reads its element before it
  writes it. `C` may not partly overlap `A`, and may not overlap a vector.
- Empty: an `A` with no element records nothing and returns an empty handle, as
  `execute_binary` does.
- Launch: `sycl::range<2>{outer, inner}`, the shape `ger` uses, so no work-item divides
  (`sushiblas/modules/kernels/README.md`, under
  `sushiblas/modules/kernels/blas_kernels/include/SushiBLAS/blas_kernels/sycl/level2/ger.hpp`). The reference path is
  two nested host loops over the same functor, as `MapTask` runs one functor on both paths
  (`sushiblas/modules/engine/functor_map/README.md:37-41`), so the paths agree bit for bit.
- Determinism: nothing accumulates.
- Cost per element, read from the code: one load of `A`, one load per vector and one store.
  `channel_affine` does the same with two vectors and one work-item per element
  (`sushiblas/modules/kernels/README.md`, under
  `sushiblas/modules/kernels/spatial_kernels/include/SushiBLAS/spatial_kernels/sycl/broadcast.hpp`). The seam has no
  `sycl::vec` path; neither has `channel_affine`. `benchmark_broadcast` prints `mul` with a
  leading and with a trailing vector beside `channel_affine` at `[8192, 384]` and at
  `[384, 65536]`, and the `range<2>` launch beside `Launch::indexed` with a division.

The equal-shape operations of `ElementwiseOps` keep their kernels and their vector path.
Nothing that exists is routed through the new seam inside SushiBLAS.

**Adds.** `sushiblas/modules/engine/functor_map/include/SushiBLAS/functor_map/broadcast_task.hpp`,
the task body, installed because a consumer's functor instantiates it.
`sushiblas/modules/operations/broadcast/` with `CMakeLists.txt`, `README.md`,
`include/SushiBLAS/broadcast/broadcast_operand.hpp`, `include/SushiBLAS/broadcast/dispatch.hpp`,
`include/SushiBLAS/broadcast/broadcast.hpp`, `source/add.cpp`, `source/sub.cpp`,
`source/mul.cpp`, `source/div.cpp` and `tests/test_broadcast.cpp`, `tests/test_broadcast_seam.cpp`.
`sushiblas/tests/benchmark/benchmark_broadcast.cpp`. Operation identities `"broadcast.add"_op`
to `"broadcast.div"_op`.

**Must not change.** `ElementwiseOps`, `SpatialOps::channel_affine`, `MapTask`, `ZipTask`.

**Tests.** Each operation and each axis against a host loop at ranks 2, 3 and 4, on HALF,
FLOAT32 and FLOAT64. `C` as `A`. A seam call with one leading and two trailing vectors and a
`bind_scalars` functor. A vector that is a view into an arena. `ForceReference` equals the
device bit for bit. Each refusal: a count that is no prefix product, a `ColumnMajor` operand,
a partial overlap, a mixed dtype, a null vector.

**SushiAI then.** G3 lowers four new operations through `BroadcastOps`; G4 moves the rows of
the table above onto `execute_broadcast`, each behind its own measurement.

### B4. Bilinear upsampling and its gradient

**Limit and evidence.** `SpatialOps` has `upsample_nearest` and its gradient only
(`sushiblas/modules/operations/spatial/include/SushiBLAS/spatial/spatial.hpp:153-170`), so
`K_UPSAMPLE2D` is nearest (`modules/core/graph/source/lowering/spatial.cpp:194`) and a class
boundary of `SegmentHead` falls on a multiple of the head's scale
(`docs/design/DATASETS.md:162-163`).

**Design.** Two members of `SpatialOps` beside the nearest pair, shaped like them: integer
scales, NHWC through the operands' strides, the gradient as a gather. Sample positions are
half-pixel centres, the convention PyTorch names `align_corners=False`, because with an
integer scale the weights then depend on the output index modulo the scale only.

Rejected: a mode flag on `upsample_nearest`, which would put a branch in a kernel that copies
bit patterns today; a real-valued scale, which no head asks for.

**Signature.**

```cpp
/**
 * @brief Interpolates each output pixel from the four input pixels around its centre.
 * @param out Its extents must be exactly `scale_h * H` by `scale_w * W`.
 * @param scale_h Must be positive, as must scale_w.
 * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree,
 *         or out overlaps in.
 */
TaskHandle upsample_bilinear(const Tensor& in, Tensor& out, int64_t scale_h, int64_t scale_w);

/**
 * @brief Sums onto each input pixel the output gradients it contributed to, by its weights.
 * @param scale_h The factor the forward used, as is scale_w; both must be positive.
 * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree,
 *         or dx overlaps dy.
 */
TaskHandle upsample_bilinear_backward(const Tensor& dy, Tensor& dx,
                                      int64_t scale_h, int64_t scale_w);
```

**Semantics.**

- For output row `h_o`: `s = (h_o + 0.5) / scale_h - 0.5`, `h0 = floor(s)`, `t_h = s - h0`,
  and the two source rows are `h0` and `h0 + 1`, each clamped to `[0, H - 1]`. Columns
  likewise with `t_w`. The value is
  `(a * (1 - t_w) + b * t_w) * (1 - t_h) + (c * (1 - t_w) + d * t_w) * t_h`, evaluated in
  that order in `Accumulator<T>` and rounded once on the store. `a`, `b` are the pixels of row
  `h0` and `c`, `d` those of row `h0 + 1`.
- A scale of 1 on an axis copies along that axis. Both scales 1 give the input's bit patterns.
- Dtypes: `Domain::Real`. Rank 4 `[N, H, W, C]`, any strides, as `upsample_span_for` reads
  them (`sushiblas/modules/operations/README.md`, under `upsample_nearest.cpp`).
- The gradient gives each input pixel one work-item. It visits the output rows from
  `(h - 1) * scale_h` to `(h + 2) * scale_h - 1`, clamped to the output, and the columns
  likewise, in ascending `(h_o, w_o)` order, and adds `dy` times the weight the forward gave
  this pixel, which is the sum of both taps when clamping makes them the same pixel. Every
  element of `dx` is written. No atomic is used.
- Paths: the forward evaluates one expression on both paths. The gradient sums in one order on
  both; the reference sums in `ReferenceSum`.
- Replay: no state.

**Adds.** In `sushiblas/modules/operations/spatial/`: two declarations in
`include/SushiBLAS/spatial/spatial.hpp`, `source/upsample_bilinear.cpp`,
`tests/test_spatial_upsample_bilinear.cpp`. In `sushiblas/modules/kernels/spatial_kernels/`
under `include/SushiBLAS/spatial_kernels/`: `reference/upsample_bilinear.hpp`,
`sycl/upsample_bilinear.hpp`, and one include line each in `spatial.hpp`. Operation identities
`"spatial.upsample_bilinear"_op` and `"spatial.upsample_bilinear_backward"_op`.

**Must not change.** `upsample_nearest`, its gradient and their kernels.

**Tests.** A 2 by 2 input at scale 2 against the sixteen values computed by hand. A constant
input gives that constant. Scales 1, 2, 3 and 4, unequal on the two axes. The gradient against
finite differences in FLOAT64, and the identity `sum(dy * forward(x)) == sum(backward(dy) * x)`
within rounding. HALF beside the FLOAT32 result within one HALF unit. A strided view. The
reference against the device.

**SushiAI then.** G7 adds `K_UPSAMPLE2D_BILINEAR` in
`modules/core/graph/source/lowering/spatial.cpp`.

### B5. The index of the maximum along one axis

**Limit and evidence.** `ReductionOps::argmax` reduces the whole tensor to one index, and the
per-axis reductions are `max_along`, `min_along` and `sum_along`
(`sushiblas/modules/operations/reductions/include/SushiBLAS/reductions/reductions.hpp:98-105`,
`:150-177`). `measure_segmentation` therefore picks each pixel's class on the host from
`R * C` logits, 39 million reads for one 512 by 512 image over 150 classes, and fails above
4096 classes (`modules/vision/training/README.md:190-195`).

**Signature**, a member of `ReductionOps` beside `max_along`:

```cpp
/**
 * @brief Computes the index of the maximum along one axis of a rank-2 tensor.
 * @param axis 0 for a per-column index, 1 for a per-row index.
 * @param result Rank-1 INT64 tensor of the surviving length; each element receives the
 *        lowest index of its segment's maximum, counted along the reduced axis.
 * @throws std::invalid_argument if t is empty or on a rank, axis, length or dtype mismatch.
 */
TaskHandle argmax_along(const Tensor& t, int axis, Tensor& result);
```

**Semantics.** It folds `(value, index)` pairs with `Reduce::RunningArgMax` in the segmented
shape `max_along` uses, so the order is total, the lowest index wins a tie and the NaN rule is
that of `argmax` (`sushiblas/modules/operations/README.md`, under
`sushiblas/modules/operations/reductions/source/argmax.cpp`).
The result is INT64 for the reason `argmax` gives. `t` is `Domain::Real`. Both paths give the
same indices. No state on replay.

**Adds.** One declaration in `reductions.hpp`,
`sushiblas/modules/operations/reductions/source/argmax_along.cpp`,
`sushiblas/modules/operations/reductions/tests/test_argmax_along.cpp`. Identity
`"reductions.argmax_along"_op`.

**Must not change.** `argmax`, `max_along`, `ArgReductionDispatch`.

**Tests.** Both axes against a host loop; a tie; a NaN in a segment; a segment of one element;
agreement with `argmax` on a one-row matrix; each refusal.

**SushiAI then.** G10 adds `K_ARGMAX_ROWS`.

### B6. GELU by the error function

**Limit and evidence.** `NonLinearOps::gelu` is the tanh approximation
(`sushiblas/modules/operations/nonlinear/include/SushiBLAS/nonlinear/nonlinear.hpp:160-168`),
so the language reference is run with the same one (`docs/design/LANGUAGE.md:79`). Weights
trained with the error-function form would be evaluated with another function.

**Signature**, two members of `NonLinearOps` beside `gelu`:

```cpp
/** @brief Computes `t = 0.5 * t * (1 + erf(t / sqrt(2)))` in place. */
TaskHandle gelu_erf(Tensor& t);

/** @brief Computes `dx = dy * (0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x * x / 2) / sqrt(2 * pi))`. */
TaskHandle gelu_erf_backward(const Tensor& dy, const Tensor& x, Tensor& dx);
```

**Semantics.** As `gelu` and `gelu_backward` in every respect but the function: the same
operand checks, the same in-place rule, the per-element seam on both paths. HALF computes in
`float`. `gelu` keeps its formula, because recorded plans and the fused form
`K_GEMM_BIAS_GELU` compute it.

**Adds.** Two declarations in `nonlinear.hpp`,
`sushiblas/modules/operations/nonlinear/source/gelu_erf.cpp`, a test file beside the
activation tests. Identities `"nonlinear.gelu_erf"_op` and `"nonlinear.gelu_erf_backward"_op`.

**Tests.** Eleven points against values computed in double; the gradient against finite
differences; the reference against the device.

**SushiAI then.** An `approximate` key on the layer that emits GELU, above the graph. Not
scheduled until a model needs it.

### B7. State and test that tensors may be created from several threads

**Limit and evidence.** The first read of a `ReaderDataset` into a new `Batch` allocates, and a
loader on threads does that concurrently (`docs/design/REMAINING_WORK.md:193-194`). The code
supports it: `TensorFactory::create` is `const` and holds a context pointer and a layout
(`sushiblas/modules/data/tensor/include/SushiBLAS/tensor/tensor_factory.hpp:103-107`),
`Storage` calls the allocator and nothing else
(`sushiblas/modules/data/tensor/include/SushiBLAS/tensor/storage.hpp:65-89`), and every method
of `USMAllocator` "may be called from any thread at any time"
(`sushiruntime/modules/platform/memory/include/SushiRuntime/memory/usm_allocator.hpp:50-51`).
Recording and `Engine::execute` are single-threaded by precondition
(`sushiblas/modules/engine/engine/include/SushiBLAS/engine/engine.hpp:105-118`). No SushiBLAS
header says which calls are safe, and no SushiBLAS test allocates from two threads.

**Design.** No code changes. One sentence on `TensorFactory::create` and on
`Engine::create_tensor`: "May be called from any thread at any time, also while another
thread records or executes." One test.

**Adds.** `sushiblas/modules/data/tensor/tests/test_tensor_factory_threads.cpp`.

**Tests.** Eight threads each create and release 256 tensors of sizes from 1 element to 1 MiB.
Every address is aligned to `K_STORAGE_ALIGNMENT`, no two live tensors overlap, and every
release returns without an error. The case also runs under the sanitizer build when the `sb`
CLI has one.

**SushiAI then.** A8, the loader on threads, in `modules/core/data`.

### B8. State and test host access to a Shared tensor between executions

**Limit and evidence.** SushiAI reads and writes Shared tensors through `Tensor::data_as` on
the host (the cases of R1; the loss and the heads a training step reads,
`modules/vision/training/README.md:92-93`). `sushiblas/modules/README.md` documents the `IO`
calls as the way to move elements, and says nothing of direct access.

**Design.** No code changes. A paragraph under `Tensor::data` in `sushiblas/modules/README.md`:
a tensor of `Residency::Shared` or `Residency::Host` may be read and written through
`data_as` on the host while no recorded task that touches its region is pending or running;
a task executed afterwards reads the bytes written; a `Residency::Device` tensor is moved with
`IO`. One test.

**Adds.** `sushiblas/modules/data/tensor/tests/test_tensor_host_access.cpp`.

**Tests.** An INT32 and a FLOAT32 tensor filled on the host, copied by `Level1::copy` in a
plan compiled once and executed three times with the source rewritten between executions; the
destination equals the source each time. On every device queue of the context.

**SushiAI then.** `modules/core/graph/README.md` cites it.

## 5. SushiAI graph

Each item is the SushiAI session's work. "Now" means it needs nothing from sections 3 and 4
and is written before them; like the rest of the tree, it is verified at the first build.

Common to the items: a new operation is one `OpName` in
`modules/core/ops/include/SushiAI/ops/ops.hpp` and its entry in `K_ALL`, one emitter of
`GraphBuilder` that returns a `Traced` and refuses as `gather_rows` does, one row of its
domain's lowering table, and one row of its domain's rule table when it is differentiable. The
static assertions `every_differentiable_op_has_a_rule` and `no_gradient_barrier_has_a_rule`
hold after each item. No existing operation changes its identity, its arity or its attribute
slots.

### G1. A random mask, and host work before each replay (after B1)

**State.** Written without B1, not yet built: `ReplayPrelude` takes the run index,
`K_DROPOUT` is one node with outputs `[y, saved_mask]`, and the mask is drawn on the host
by `record_keep_draw` (`modules/core/graph/source/lowering/random.cpp`), which is the one
function B1 replaces. A11 is written on it.

**Limit.** Section 4, B1. On the SushiAI side: `CompiledStep::run` calls `before_replay` on
the trailing recorder only (`modules/core/graph/source/compiled_step.cpp:116-117`), and a
lowering row receives a `LoweringContext` that can register nothing
(`modules/core/graph/include/SushiAI/graph/lowering/context.hpp:63-239`).

**Design.** Two bricks.

A seam: a lowering row hands the step host work to run before each replay.

```cpp
// <SushiAI/graph/replay_prelude.hpp>
/** @brief Names host work a lowered node needs before each replay of its step. */
using ReplayPrelude = std::function<void()>;

// LoweringContext
/**
 * @brief Asks the step to run @p prelude on the calling thread before each replay.
 * @param prelude Records no device work and reads no device result.
 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
 */
[[nodiscard]] Status before_each_replay(ReplayPrelude prelude) const;
```

`Lowering` collects the preludes in node order and `CompiledStep` keeps them and runs them at
the top of `run()`, before the trailing recorder's `before_replay`. `LoweringContext` gains a
second constructor that takes the list; the existing one stays and forwards with a list that
refuses, so no caller breaks.

An operation: `K_DROPOUT_MASK`, a gradient barrier with no operand.

```cpp
/**
 * @brief Emits a mask of @p shape whose elements are 1 with probability @p keep and 0
 *        otherwise, drawn anew at each replay.
 */
Traced dropout_mask(Shape shape, DType dtype, double keep) const;
```

**Semantics.** The row allocates an INT64 `[2]` scratch, registers a prelude that calls
`SushiBLAS::write_rng_position(position, stream.reserve(1))`, and records
`PositionedRandomOps::bernoulli`. Preludes run in node order on one thread, so the positions a
run takes are a function of the seed and of the number of replays before it. `dtype` is
FLOAT32 or FLOAT64; the operation does not narrow under `MixedPrecisionPolicy`, and the cast
pass converts its result where a HALF operand meets it. An evaluation trace emits no node. A
step with a mask node needs an engine whose stream the caller seeded.

**Adds.** `modules/core/graph/include/SushiAI/graph/replay_prelude.hpp`,
`modules/core/graph/include/SushiAI/graph/lowering/random.hpp` with `K_RANDOM_LOWERINGS`,
`modules/core/graph/source/lowering/random.cpp`; one line in
`modules/core/graph/include/SushiAI/graph/lowering/all.hpp`.

**Tests.** One compiled step run twice gives two masks that differ, with `compile_count()` of
1. Two engines seeded alike give equal masks at equal run indices. The mean of a `[1024, 1024]`
mask is within 0.005 of `keep`. A step without a mask node runs no prelude.

**Above the graph.** A11: `NN::Dropout` emits `scale(mul(x, mask), 1 / (1 - rate))` in its
training form.

### G2. Square root, logarithm and power (now)

**Limit.** The IR has `K_EXP`, `K_ABS` and `K_ATAN` as its unary math
(`modules/core/ops/include/SushiAI/ops/ops.hpp:149-164`). SushiBLAS has the three kernels
(`sushiblas/modules/operations/elementwise/include/SushiBLAS/elementwise/elementwise.hpp:108`,
`:148`, `:163`).

**Design.** `K_SQRT`, `K_LOG` and `K_POW`, the exponent of the last in attribute slot 0.
Emitters `ValueID sqrt(ValueID x) const`, `ValueID log(ValueID x) const` and
`ValueID pow(ValueID x, double exponent) const`, beside `exp`
(`modules/core/graph/include/SushiAI/graph/builder.hpp:343`). The first two lower through
`record_copy_then` as `K_EXP` does (`modules/core/graph/source/lowering/elementwise.cpp:171-174`);
`K_POW` has its own row for the argument. A reciprocal root is `pow(x, -0.5)`; no fourth
operation is added.

Rules: `sqrt` contributes `scale(div(dy, y), 0.5)` with `y` the forward result; `log`
contributes `div(dy, x)`; `pow` contributes `scale(mul(dy, pow(x, e - 1)), e)`. A value outside
a function's domain gives what the kernel gives, a NaN or an infinity, and the graph adds no
check.

**Tests.** A gradient check of each on an engine, with inputs kept above 0.1; `pow` at
exponents 2, 0.5 and -0.5.

### G3. Broadcast arithmetic and a sum along an axis (after B3)

**Limit.** `add`, `sub`, `mul` and `div` refuse unequal shapes, so the position table is added
by a reshape to `[B, T * D]` rows (`modules/language/blocks/source/position_embedding.cpp:95`)
and the key and value cache has no way to add its fed `[1, T]` mask to each head's scores
(`modules/language/generation/README.md:119`).

**Design.** Four operations, `K_BROADCAST_ADD`, `K_BROADCAST_SUB`, `K_BROADCAST_MUL` and
`K_BROADCAST_DIV`, with the side in attribute flag 0, and `K_SUM_ALONG`. The equal-shape
operations and the fusion patterns that match them
(`modules/core/graph/source/fusion.cpp:144-145`) are not touched.

```cpp
/** @brief Names whether a vector's shape is a prefix or a suffix of the tensor's. */
enum class BroadcastSide : std::uint8_t { Leading, Trailing };

Traced broadcast_add(ValueID x, ValueID vector, BroadcastSide side) const;
Traced broadcast_sub(ValueID x, ValueID vector, BroadcastSide side) const;
Traced broadcast_mul(ValueID x, ValueID vector, BroadcastSide side) const;
Traced broadcast_div(ValueID x, ValueID vector, BroadcastSide side) const;

/** @brief Sums @p x down to @p kept, a prefix or a suffix of its shape. */
Traced sum_along(ValueID x, Shape kept, BroadcastSide side) const;
```

**Semantics.** The vector's shape is a prefix (`Leading`) or a suffix (`Trailing`) of `x`'s
shape; the result has `x`'s shape and type. The four lower to `BroadcastOps`. `K_SUM_ALONG`
views `x` as `[outer, inner]` and records `ReductionOps::sum_along`, axis 1 to keep a prefix
and axis 0 to keep a suffix; that kernel exists
(`sushiblas/modules/operations/reductions/include/SushiBLAS/reductions/reductions.hpp:170-177`).

Rules, with `S` for `sum_along` down to the vector's shape: `add` gives `dy` to `x` and
`S(dy)` to the vector; `sub` gives `dy` and `-S(dy)`; `mul` gives `broadcast_mul(dy, v)` and
`S(mul(dy, x))`; `div` gives `broadcast_div(dy, v)` and `-S(mul(dy, broadcast_div(y, v)))`,
with `y` the forward result.

**Tests.** A gradient check of each operation on each side at ranks 2 and 3. The position
table added by `broadcast_add` equals the reshape form bit for bit.

### G4. The special forms through the broadcast seam (after B3 and a measurement)

**Limit.** The table of section 4, B3.

**Design.** The node identities, the builder signatures and the results stay. Only the rows of
the lowering tables change, one row per commit, each with a `perf(graph)` message that carries
both timings as the `performance` skill asks. A row whose timing does not improve is not
changed.

| Row | Change | Removes |
| --- | --- | --- |
| `lower_bias_add` | One `BroadcastOps::add`, trailing | The copy and the ones vector |
| `lower_causal_mask` | One `execute_broadcast` with a `[T * T]` keep vector and the functor `keep != 0 ? x : fill` | The shift vector, the HALF conversions, and `x * 0`, which is NaN for an infinite `x` |
| `lower_layer_norm` | The three `channel_affine` calls become one seam call with the mean and the inverse deviation leading and gamma and beta trailing | The transposed views, the ones and zeros vectors, two `[rows, D]` scratch tensors |
| `lower_log_softmax` | The two `ger` calls become two in-place leading subtractions | The ones vector |
| `lower_log_softmax_backward` | `dx = exp(log_p)`, then one in-place leading call `-x * s`, then `axpy` | The `[N, C]` scratch and two tasks |

Not measured: whether each row is faster. The count of loads per element is the same or lower
in every row, read from the code.

**Tests.** The existing tests of the five operations pass unchanged; `operation_count()`
assertions move with the rows.

### G5. One element per row, and the row operations through `IndexOps` (after B2)

**Limit.** Section 4, B2.

**Design.** Two operations, `K_TAKE_ALONG` (differentiable) and `K_PUT_ALONG`
(differentiable), each the other's gradient:

```cpp
/** @brief Picks `values[r, index[r]]` from a `[R, C]` value; the result is `[R]`. */
Traced take_along(ValueID values, ValueID index) const;

/** @brief Writes `picked[r]` at column `index[r]` of a zeroed `[R, columns]` result. */
Traced put_along(ValueID picked, ValueID index, int64_t columns) const;
```

`lower_gather_rows` records `IndexOps::take_rows` and `lower_scatter_add_rows` records
`IndexOps::add_rows` with `beta` of 0. `row_selection`, its offsets scratch and its ones
vector are deleted (`modules/core/graph/source/lowering/linalg.cpp:83-111`). The index of all
four operations is INT32 or INT64. An index out of range has the result B2 defines, and
`modules/core/graph/README.md:65-66` is rewritten to say so.

**Tests.** Gradient checks of the two new operations. `gather_rows` and `scatter_add_rows`
equal their current results on the existing fixtures. An index of -1 and of the bound in each,
with no write outside the result.

**Above the graph.** A12: both cross-entropy objectives build `take_along(log_p, labels)` with
the label as the index, an ignored pixel keeps the ignore index and reads as zero, and the
`K_LARGEST_INDEX` refusals go.

### G6. A backward rule for the scatter-add (now)

**Limit.** `K_SCATTER_ADD_ROWS` is declared without a gradient
(`modules/core/ops/include/SushiAI/ops/ops.hpp:274`), so Dice feeds an `R * C` one-hot leaf
from the host at each step, 157 MB for one 512 by 512 image over 150 classes
(`modules/vision/training/README.md:255-259`).

**Design.** The operation becomes differentiable. Its rule, `scatter_add_rows_backward` in
`modules/core/autograd/source/rules/linalg.cpp` beside `gather_rows_backward`, contributes
`gather_rows(dy, indices)` to the rows operand. The index operand gets no gradient.

Dice's class sums are then `scatter_add_rows` of the picked probabilities by class. Its
lowering has one right-hand column, so one work-item walks the `R` pixels (section 4, B2).
After G5 the same sums are `sum_along(put_along(picked, labels, C))`, parallel over `R * C`
elements at the cost of an `[R, C]` device value. A12 chooses between the two by measuring.

**Tests.** A gradient check with a repeated index and with an index that no row names.

### G7. A bilinear upsample (after B4)

**Design.** `K_UPSAMPLE2D_BILINEAR` and `K_UPSAMPLE2D_BILINEAR_BACKWARD`, with the emitters
`upsample2d_bilinear` and `upsample2d_bilinear_backward` shaped as the nearest pair
(`modules/core/graph/include/SushiAI/graph/builder.hpp:636`, `:675`), two rows in
`modules/core/graph/source/lowering/spatial.cpp` and one rule in
`modules/core/autograd/source/rules/spatial.cpp`. Two operations and not an attribute of
`K_UPSAMPLE2D`: an attribute would put a mode switch in an existing row and its rule.

**Tests.** A gradient check at scales 2 and 3; the forward against the hand-computed values of
B4.

**Above the graph.** A12: `NN::Upsample2d` and `SegmentHead` read a `mode` key, `nearest` when
absent.

### G8. A masked score that is finite in every element type (now)

**Limit.** `CausalSelfAttention` masks with -1e9
(`modules/language/blocks/source/causal_self_attention.cpp:46`), and the lowering converts the
fill to the operand's type (`modules/core/graph/source/lowering/linalg.cpp:157`, `:170-171`).
In HALF that is minus infinity (`docs/design/LANGUAGE.md:80`).

**Design.** `triangle_affine` bounds the fill to the lowest finite value of the operand's
element type before it writes the vector: -65504 in HALF. The attribute and the emitter are
unchanged. The product `x * 0` for an infinite score stays until G4's select form.

**Tests.** A HALF `[2, 4, 4]` input: every masked entry is -65504 and none is infinite.

### G9. A step made of recorders alone (now)

**Limit.** Gradient accumulation needs "a replay without the update"
(`docs/design/MODEL_IMPORT.md:121-122`). The update is the trailing recorder of the step that
runs backward (`modules/core/graph/source/compiled_step.cpp:78-85`). Two steps solve it: one
of forward, backward and an `axpy` of each gradient into an accumulator, and one that holds
the update alone. `CompiledStep::compile` takes a graph; reading
`modules/core/graph/source/compiled_step.cpp:47-109`, an empty graph with a trailing recorder
compiles and is not `empty()`, but no test pins it.

**Design.** A test, and whatever the test shows to be missing. If it passes, nothing else
changes.

**Tests.** A step compiled from an empty `Graph` and a recorder that records one `scal`; three
runs scale the tensor three times; `operation_count()` is 1.

### G10. The class of each row (after B5)

**Design.** `K_ARGMAX_ROWS`, a gradient barrier: `Traced argmax_rows(ValueID x) const` over a
`[R, C]` value gives INT64 `[R]`. It lowers to `ReductionOps::argmax_along` with axis 1 and
does not narrow under `MixedPrecisionPolicy`.

**Tests.** Against a host argmax, with a tie.

**Above the graph.** A12: `measure_segmentation` reads `R` indices.

## 6. Decisions

No item changes or removes an existing public declaration of SushiBLAS or SushiRuntime, so no
consumer, SushiEngine included, has to change to keep compiling. The author of this plan did
not read SushiEngine; that statement rests on the plan adding declarations only.

Ruled on 2026-10-11. The owner approved the new operation modules and delegated the rest to
the engineer of the SushiAI session, who took the recommendation in each case:

| # | Subject | Ruling | By |
| --- | --- | --- | --- |
| 1 | Two operation modules in SushiBLAS, `indexing` and `broadcast`, and one kernel module, `index_kernels` | As written. Not members on `SparseOps` and `ElementwiseOps`, whose contracts say the opposite of what the new operations do | Owner |
| 2 | An index out of range | Defined in every build type: zero on a gather, skipped on a transpose. SushiAI adds its own named error above it (A5) | Engineer |
| 3 | `BroadcastOps` and `ColumnMajor` | Refused in the first release. If SushiEngine needs it, the geometry rule gets a second arm | Engineer |
| 4 | Bilinear sampling | Half-pixel centres, integer scales. An `align_corners` form would be a further pair of operations, added when a model needs it | Engineer |
| 5 | The positioned random fill | `PositionedRandomOps` as its own view with one fill (Bernoulli); more distributions on evidence | Engineer |
| 6 | R1 and layer 2 of `GRAPH_REPLAY.md` | Open: the session that owns that design confirms the sentence or rewords it | That session |
| 7 | B6, the error-function GELU | Not now. It waits for a model trained with that form to be imported | Engineer |
| 8 | Releases | SushiBLAS tags a MINOR release at the end of phase 2 and of phase 3; SushiAI moves its pin in one `build(deps)` commit each time | Engineer |
| 9 | B7 and B8 | Inside the last build of the v0.4.0 migration: two test files and two paragraphs | Engineer |

## 7. What the plan leaves out

- **A design for several machines.** [`LAYERED_EXECUTION.md`](LAYERED_EXECUTION.md) holds it.
- **Removing a special form.** `K_LAYER_NORM`, `K_CAUSAL_MASK`, `K_BIAS_ADD` and
  `K_LOG_SOFTMAX_BACKWARD` stay as nodes. G4 changes how they lower, row by row, and only on a
  measured gain. Whether a composition from G2 and G3 could replace a node is not claimed: it
  would add IR values the memory plan has to place, and nothing here measures that.
- **A fused row-wise log-softmax, a fused `K_NLL_LOSS`, bfloat16, skipping a whole step on
  overflow, feeding a batch without two copies, a contiguous `K_RESHAPE` as a view.** They are
  older entries of `docs/design/REMAINING_WORK.md` under "Memory planning", "Fusion and the
  IR" and "Convolutional networks", outside the sections this plan was asked to read. The
  batch copy grows with A1, which feeds three leaves.
- **A row write for the key and value cache.** Section 4, B2: the copy it would save is 98 304
  elements per buffer per token.
- **A parallel `add_rows`.** Section 4, B2: gated on a profile.
- **Sine and cosine in the IR.** They serve rotary positions, which the owner has not chosen
  over the learned table (`docs/design/LANGUAGE.md:161-162`). SushiBLAS has both kernels; the
  work would be the shape of G2.
- **Restoring the random stream on resume.** After G1 a resumed run draws its masks from
  offset 0, because `RNGStream::reseed` rewinds the offset and nothing sets it
  (`sushiblas/modules/engine/engine/include/SushiBLAS/engine/rng_stream.hpp:57-62`).
  `sushiblas/modules/README.md`, under `RNGStream::reseed`, already names the shape such a
  method would take. It belongs with A3 and is raised when A3 is designed.
- **Top-k for the mixture of experts.** Phase 7 of `docs/design/LANGUAGE.md` is not started
  and no document states what it lacks.
- **Everything in the A rows** beyond the table of section 1. They are SushiAI's own and are
  planned in `docs/design/LANGUAGE.md`, `docs/design/DATASETS.md` and
  `docs/design/MODEL_IMPORT.md`.
