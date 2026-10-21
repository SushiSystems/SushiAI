# SushiBLAS items: the random fill and the index family

Part of the [graph capabilities plan](README.md).

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

## B1. A random fill that reads its stream position when it runs

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

## B2. An index family: rows by index, and one element per row

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
