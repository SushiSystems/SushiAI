# SushiBLAS items: broadcast and the other operations

Part of the [graph capabilities plan](README.md).

The rules common to every B item are at the top of [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md).

## B3. A broadcast seam: one value per row or per column, any functor

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

## B4. Bilinear upsampling and its gradient

**Limit and evidence.** `SpatialOps` has `upsample_nearest` and its gradient only
(`sushiblas/modules/operations/spatial/include/SushiBLAS/spatial/spatial.hpp:153-170`), so
`K_UPSAMPLE2D` is nearest (`modules/core/graph/source/lowering/spatial.cpp:194`) and a class
boundary of `SegmentHead` falls on a multiple of the head's scale
(`docs/design/DATASETS/README.md:162-163`).

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

## B5. The index of the maximum along one axis

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

## B6. GELU by the error function

**Limit and evidence.** `NonLinearOps::gelu` is the tanh approximation
(`sushiblas/modules/operations/nonlinear/include/SushiBLAS/nonlinear/nonlinear.hpp:160-168`),
so the language reference is run with the same one (`docs/design/LANGUAGE/README.md:79`). Weights
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

## B7. State and test that tensors may be created from several threads

**Limit and evidence.** The first read of a `ReaderDataset` into a new `Batch` allocates, and a
loader on threads does that concurrently (`docs/backlog/README.md:193-194`). The code
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

## B8. State and test host access to a Shared tensor between executions

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
