# SushiAI graph items

Part of the [graph capabilities plan](README.md).

Each item is the SushiAI session's work. "Now" means it needs nothing from the SushiRuntime and
SushiBLAS pages and is written before them; like the rest of the tree, it is verified at the first build.

Common to the items: a new operation is one `OpName` in
`modules/core/ops/include/SushiAI/ops/ops.hpp` and its entry in `K_ALL`, one emitter of
`GraphBuilder` that returns a `Traced` and refuses as `gather_rows` does, one row of its
domain's lowering table, and one row of its domain's rule table when it is differentiable. The
static assertions `every_differentiable_op_has_a_rule` and `no_gradient_barrier_has_a_rule`
hold after each item. No existing operation changes its identity, its arity or its attribute
slots.

## G1. A random mask, and host work before each replay (after B1)

**State.** Written without B1, not yet built: `ReplayPrelude` takes the run index,
`K_DROPOUT` is one node with outputs `[y, saved_mask]`, and the mask is drawn on the host
by `record_keep_draw` (`modules/core/graph/source/lowering/random.cpp`), which is the one
function B1 replaces. A11 is written on it.

**Limit.** B1 in [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md). On the SushiAI side: `CompiledStep::run` calls `before_replay` on
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

## G2. Square root, logarithm and power (now)

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

## G3. Broadcast arithmetic and a sum along an axis (after B3)

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

## G4. The special forms through the broadcast seam (after B3 and a measurement)

**Limit.** The table of B3 in [BLAS_BROADCAST_AND_OPERATIONS.md](BLAS_BROADCAST_AND_OPERATIONS.md).

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

## G5. One element per row, and the row operations through `IndexOps` (after B2)

**Limit.** B2 in [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md).

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

## G6. A backward rule for the scatter-add (now)

**Limit.** `K_SCATTER_ADD_ROWS` is declared without a gradient
(`modules/core/ops/include/SushiAI/ops/ops.hpp:274`), so Dice feeds an `R * C` one-hot leaf
from the host at each step, 157 MB for one 512 by 512 image over 150 classes
(`modules/vision/training/README.md:255-259`).

**Design.** The operation becomes differentiable. Its rule, `scatter_add_rows_backward` in
`modules/core/autograd/source/rules/linalg.cpp` beside `gather_rows_backward`, contributes
`gather_rows(dy, indices)` to the rows operand. The index operand gets no gradient.

Dice's class sums are then `scatter_add_rows` of the picked probabilities by class. Its
lowering has one right-hand column, so one work-item walks the `R` pixels (B2 in [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md)).
After G5 the same sums are `sum_along(put_along(picked, labels, C))`, parallel over `R * C`
elements at the cost of an `[R, C]` device value. A12 chooses between the two by measuring.

**Tests.** A gradient check with a repeated index and with an index that no row names.

## G7. A bilinear upsample (after B4)

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

## G8. A masked score that is finite in every element type (now)

**Limit.** `CausalSelfAttention` masks with -1e9
(`modules/language/blocks/source/causal_self_attention.cpp:46`), and the lowering converts the
fill to the operand's type (`modules/core/graph/source/lowering/linalg.cpp:157`, `:170-171`).
In HALF that is minus infinity (`docs/design/LANGUAGE.md:80`).

**Design.** `triangle_affine` bounds the fill to the lowest finite value of the operand's
element type before it writes the vector: -65504 in HALF. The attribute and the emitter are
unchanged. The product `x * 0` for an infinite score stays until G4's select form.

**Tests.** A HALF `[2, 4, 4]` input: every masked entry is -65504 and none is infinite.

## G9. A step made of recorders alone (now)

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

## G10. The class of each row (after B5)

**Design.** `K_ARGMAX_ROWS`, a gradient barrier: `Traced argmax_rows(ValueID x) const` over a
`[R, C]` value gives INT64 `[R]`. It lowers to `ReductionOps::argmax_along` with axis 1 and
does not narrow under `MixedPrecisionPolicy`.

**Tests.** Against a host argmax, with a tie.

**Above the graph.** A12: `measure_segmentation` reads `R` indices.
