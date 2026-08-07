# Plan B — SushiAI: IR and autograd

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give SushiAI's IR the vocabulary, shape inference and gradients for convolutional networks, and split `autograd/backward.cpp` into one translation unit per domain *before* the twenty new identities land in it.

**Architecture:** Three layers, in strict order. First the IR grows the room it needs — eight attribute slots and a graph-wide `TensorLayout` invariant. Then twenty operation identities gain shape inference in `GraphBuilder`, so a wrong convolution geometry is a throw at build time rather than a wrong number at run time. Then autograd is restructured into `rules/<domain>.{hpp,cpp}` and only afterwards do the new rules get written, each one born in its final file. The plan closes with the `IM2COL` rematerialisation pass, which is a graph-to-graph transform over the differentiated graph and depends on everything before it.

**Tech Stack:** C++17 (hard ABI constraint — no concepts, no `consteval`, no `std::span`, no `std::format`), SYCL via SushiBLAS, GoogleTest, CMake + Ninja, the `sa` developer CLI.

## Global Constraints

- **C++17 across the whole stack.** Use `SushiRuntime::span`, `SushiAI::format`, `SA_THROW_IF(cond, "msg {}", args...)`. `constexpr`, never `consteval`.
- **Build and test only through `sa`.** `sa build`, `sa test --suite <tier>`, `sa run`. Never invoke `cmake`, `ninja` or `ctest` directly.
- **No commits.** The user commits and pushes; this plan never runs `git commit`.
- **Warnings are errors.** `-Wall -Wextra -Werror -pedantic`; a warning fails the build.
- **The GTest suite-name prefix is the only thing assigning a CTest label.** `Unit_*` → `unit`, `Integration_*` → `integration`, `Regression_*` → `regression`. A test in the wrong-prefixed suite silently never runs under its tier.
- **Every public declaration carries Doxygen** with `@brief`, `@param` and `@return`. This is enforced by review, not by the compiler.
- **NHWC is the single native layout (CV-3).** `LAYOUT_CONVERT` is the only node permitted to change it.
- **Every spatial backward SushiBLAS provides is a gather (CV-6).** No rule in this plan may introduce a floating-point atomic scatter to work around a missing kernel; if a kernel is missing, that is a Plan A gap to report, not a thing to route around.

---

## Decisions taken before writing this plan

Both were put to the user and answered; they are settled, and a task below implements each.

**D-B1 — `Graph::TensorLayout` is a new enum, not an extension of `SushiBLAS::Core::Layout`.**
`include/SushiAI/core/common.hpp:56` aliases `SushiAI::Layout` to `SushiBLAS::Core::Layout`, which is `{ROW_MAJOR, COLUMN_MAJOR}` — a matrix storage order, a different question from NHWC/NCHW. The alias stays exactly as it is and keeps serving GEMM operands. Adding `NHWC` to the BLAS enum would leave every matrix operation in another repository accepting two values it cannot act on.

**D-B2 — `Graph::propagate_requires_grad()` is renamed to `finalise()`.**
The spec (CV-4) and `include/SushiAI/graph/ir.hpp:202`'s own doc comment already say `finalise()`; only the method disagreed. `finalise()` names the phase rather than one of its steps, so it stays truthful when CV-4's layout-uniformity proof becomes its second step. Three call sites move: `src/autograd/backward.cpp:511`, `tests/functional/unit/test_ir.cpp:83` and `:101`.

---

## Amendments made during execution

Both came out of Task 2+3's review, were put to the user, and are settled. They
override the task text below wherever the two disagree.

**A-1 — There is no `Graph::mutable_value()`. Use the existing `Graph::value(ValueId)`.**
Task 2 originally added `mutable_value` as an unchecked write accessor. `ir.hpp:356`
already has a public non-const `Value& value(ValueId)` that does the same thing and
guards with `SA_THROW_IF(id >= values_.size(), ...)`. The new one made
`graph.value(graph.value_count()).layout = ...` an out-of-bounds write with no
diagnostic, where the accessor that already existed throws. Every `mutable_value(` in
the tasks below reads `value(`.

**A-2 — Layout is carried by `declare_like`, not stamped by each emitter.**
As first written, every emitter in Tasks 5, 6, 7, 14, 15 and 16 followed its `declare(...)`
with `graph_.value(out).layout = in.layout;` — the same line eighteen times, any
one of which could be forgotten. A forgotten line is not a crash: the value silently
keeps the `NHWC` default, and the first graph that puts it downstream of a
`LAYOUT_CONVERT` is rejected by `finalise()` for a violation it did not commit.

`GraphBuilder` gains:

```cpp
                /**
                 * @brief Declares a value that inherits @p like's layout.
                 *
                 * The layout is a graph-wide invariant (CV-4), so an emitter that
                 * forgot to carry it forward would not fail where it was written -
                 * it would fail in `finalise()`, on a graph that is correct, naming
                 * an operation that did nothing wrong. Carrying it here means no
                 * emitter can forget.
                 *
                 * @param like The value whose layout the result adopts.
                 * @param shape The result's extents.
                 * @param dtype The result's element type.
                 * @param lifetime Which arena owns the result.
                 * @param debug_name The profiling label.
                 * @return The new value's id.
                 */
                ValueId declare_like(ValueId like, Shape shape, DType dtype,
                                     Lifetime lifetime, std::string_view debug_name) const;
```

Every `declare(...)` + layout-stamp pair in the tasks below becomes one `declare_like(...)`
call. `layout_convert` is the single exception and keeps declaring plainly, then setting
the layout it was asked for — it is the one node permitted to change it.

---

## File structure

**Modified — IR and identities**

| File | Responsibility after this plan |
|---|---|
| `include/SushiAI/graph/ir.hpp` | `Attributes::CAPACITY` 8; `TensorLayout`; `Value::layout`; `Graph::finalise()` |
| `include/SushiAI/core/ops.hpp` | Twenty new identities, the `differentiable` flag, `LEAKY_RELU` in `ACTIVATIONS` |
| `include/SushiAI/graph/builder.hpp` | Emitter + shape inference declarations for every new identity |
| `src/graph/builder.cpp` | Their definitions |

**Created — autograd, after the split**

| File | Responsibility |
|---|---|
| `include/SushiAI/autograd/tape.hpp` | `GradientTape` — the only interface a rule sees |
| `include/SushiAI/autograd/rule.hpp` | `BackwardRule`, `RuleEntry`, the `constexpr` concat helper |
| `include/SushiAI/autograd/rules/{elementwise,linalg,activation,loss,shape,spatial,norm}.hpp` | One `inline constexpr std::array<RuleEntry, N>` per domain |
| `include/SushiAI/autograd/rules/all.hpp` | `RULES` = concat of every domain, and CV-9's `static_assert` |
| `src/autograd/tape.cpp` | `GradientTape`'s out-of-line methods |
| `src/autograd/differentiator.cpp` | Traversal only: run / contribute / skip / resolve |
| `src/autograd/rules/*.cpp` | The rule bodies, one file per domain |
| `src/graph/rematerialise.cpp` | The `IM2COL`-specific rematerialisation pass |

**Deleted**

`src/autograd/backward.cpp` — its contents are distributed by Tasks 8 and 9. `include/SushiAI/autograd/backward.hpp` survives but shrinks to `differentiate()` and `has_backward_rule()`.

**Created — tests**

`tests/functional/unit/test_layout.cpp`, `test_spatial_shapes.cpp`, `test_rule_table.cpp`, `test_rematerialise.cpp`; and new cases appended to the existing `tests/functional/integration/test_autograd_numeric.cpp`.

---

## How a new gradient is proved in this repository

Read this before Task 10; every rule task depends on it.

`tests/functional/integration/test_autograd_numeric.cpp` already holds a `Probe` class that builds a **FLOAT64** graph, declares one gradient-requiring leaf, differentiates it, compiles and executes, and compares the analytic gradient against a numeric one. Its loss is always the mean negative log-likelihood of the model's `[N, C]` output against a target matrix.

That `[N, C]` requirement is exactly what `SPATIAL_MEAN` produces from an `[N, H, W, C]` value, so a convolutional probe ends:

```cpp
const ValueId pooled = builder.spatial_mean(feature_map);   // [N,H,W,C] -> [N,C]
const ValueId logits = builder.log_softmax(pooled);
```

and the existing harness takes it from there. **Every new rule in Tasks 10–15 is proved this way** — a structural test asserting "a node of the right identity was emitted" is not a proof that the gradient is right, and this plan does not accept one in place of a numeric check.

---

# Task 1: Eight attribute slots

CV-5 needs eight: `kernel_h/w`, `stride_h/w`, `dilation_h/w`, `pad_top`, `pad_left`.

**Files:**
- Modify: `include/SushiAI/graph/ir.hpp:159-178`
- Test: `tests/functional/unit/test_ir.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `Graph::Attributes::CAPACITY == 8`; `Attributes::values` is `double[8]`. Tasks 5–7 fill slots 0–7 by the CV-5 order.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/unit/test_ir.cpp`:

```cpp
TEST(Unit_IR, AttributesHoldTheEightSlotsAConvolutionNeeds)
{
    // CV-5's eight: kernel_h/w, stride_h/w, dilation_h/w, pad_top, pad_left.
    // Eight is not a round number chosen for comfort - it is exactly
    // SushiRuntime's TaskMetadata::K_MAX_PARAMS, which is what an attribute
    // is ultimately copied into, so a ninth slot could not be lowered.
    EXPECT_EQ(Graph::Attributes::CAPACITY, std::size_t{8});

    Graph::Attributes a;
    for (std::size_t i = 0; i < Graph::Attributes::CAPACITY; ++i)
        a.values[i] = static_cast<double>(i + 1);

    EXPECT_EQ(a.get(7), 8.0);
    EXPECT_EQ(a.get(8), 0.0) << "out-of-range reads must stay defined and yield zero";
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_IR.AttributesHoldTheEightSlots"`
Expected: FAIL — `CAPACITY` is 4, so `get(7)` returns `0.0`.

- [ ] **Step 3: Widen the array**

In `include/SushiAI/graph/ir.hpp`, replace the `CAPACITY` constant and the `values` initialiser:

```cpp
            /**
             * @brief How many slots are meaningful.
             *
             * Eight, because that is @c SushiRuntime::TaskMetadata's
             * @c K_MAX_PARAMS (task_types.hpp:146) - the array an attribute is
             * ultimately copied into. A ninth slot would build here and be
             * silently dropped at lowering, so the two are pinned together by
             * the static_assert below rather than left to agree by memory.
             */
            static constexpr std::size_t CAPACITY = 8;

            /** @brief The scalar values, interpreted per operation. */
            double values[CAPACITY] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
```

- [ ] **Step 4: Pin it to the runtime's limit**

Immediately after the closing `};` of `struct Attributes`, add:

```cpp
        static_assert(Attributes::CAPACITY <= SushiRuntime::TaskMetadata::K_MAX_PARAMS,
                      "A graph attribute slot has no lowered destination: TaskMetadata "
                      "carries K_MAX_PARAMS scalars and an attribute beyond that would "
                      "be dropped by graph/lowering.cpp without any diagnostic.");
```

Add `#include <SushiRuntime/graph/task_types.hpp>` to `ir.hpp`'s include block if it is not already there.

- [ ] **Step 5: Run the test and the IR suite**

Run: `sa test --suite unit --filter "Unit_IR"`
Expected: PASS, and every pre-existing `Unit_IR` case still passes — `Attributes::get`'s bounds check is what makes widening safe for existing callers.

- [ ] **Step 6: Prove lowering did not silently change**

Run: `sa test --suite all`
Expected: PASS. `src/graph/lowering.cpp` reads attributes one index at a time (`node.attributes.flag(0)`, `.get(0)` — lines 519, 600-601, 755, 819) rather than bulk-copying the array, which is why widening it touches nothing. If this run fails, that assumption was wrong and the failure is the finding — report it rather than adjusting the test.

---

# Task 2: `TensorLayout`, and `finalise()` proving it

CV-4: layout is a graph-level invariant, and `finalise()` proves it.

**Files:**
- Modify: `include/SushiAI/graph/ir.hpp` (`Value`, `Graph`)
- Modify: `src/autograd/backward.cpp:511`, `tests/functional/unit/test_ir.cpp:83,101` (the rename's call sites)
- Test: `tests/functional/unit/test_layout.cpp` (create)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's widened `Attributes`.
- Produces: `Graph::TensorLayout { NHWC, NCHW }`; `Value::layout` defaulting to `NHWC`; `Graph::finalise()` replacing `propagate_requires_grad()`. Every later task calls `finalise()`, never the old name.

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_layout.cpp` (header block copied from `test_ir.cpp`, then):

```cpp
namespace SushiAI
{
    namespace Test
    {
        using Graph::GraphBuilder;
        using Graph::TensorLayout;
        using Graph::ValueId;

        TEST(Unit_Layout, EveryValueIsNhwcUnlessSaidOtherwise)
        {
            // CV-3: one native layout. A value that never mentions layout is
            // NHWC, so a graph built by code that predates this field is
            // uniform by construction rather than by accident.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");

            EXPECT_EQ(graph.values()[x].layout, TensorLayout::NHWC);
        }

        TEST(Unit_Layout, FinaliseAcceptsAUniformGraph)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            (void)builder.matmul(x, w);

            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, FinaliseRejectsAMixedGraphAndNamesTheValue)
        {
            // The failure this exists to catch is a layer emitting an NCHW
            // value into an NHWC network. Nothing downstream would throw: the
            // shapes still line up, every kernel still runs, and the model
            // simply learns from transposed data.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            const ValueId y = builder.matmul(x, w);

            graph.value(y).layout = TensorLayout::NCHW;

            EXPECT_THROW(graph.finalise(), Error);
        }

        TEST(Unit_Layout, LayoutConvertIsAllowedToChangeIt)
        {
            // The one exemption, and the reason the check is per-node rather
            // than a single sweep over every value.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4, 5}, "x");
            const ValueId y = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.values()[y].layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, FinaliseStillPropagatesRequiresGrad)
        {
            // The rename must not quietly drop the behaviour the old name
            // described; this is the same assertion test_ir.cpp made.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            const ValueId y = builder.matmul(x, w);

            graph.finalise();

            EXPECT_TRUE(graph.values()[y].requires_grad);
        }
    }
}
```

Register it: add `functional/unit/test_layout.cpp` to `tests/CMakeLists.txt` beside `functional/unit/test_ir.cpp`.

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_Layout"`
Expected: FAIL to **compile** — `TensorLayout`, `Value::layout`, `Graph::finalise`, `Graph::mutable_value` and `builder.layout_convert` do not exist. A compile failure is the correct first failure here.

- [ ] **Step 3: Declare the layout**

In `include/SushiAI/graph/ir.hpp`, immediately before `struct Value`:

```cpp
        /**
         * @enum TensorLayout
         * @brief Which axis order a rank-4 value's extents are in.
         *
         * Deliberately not @c SushiAI::Layout, which aliases SushiBLAS's
         * @c {ROW_MAJOR, COLUMN_MAJOR} - that is a matrix's storage order and a
         * different question. Mixed-layout GEMM is a real BLAS use case;
         * a layout change in the middle of a convolutional network is a bug,
         * which is why this one is a graph-wide invariant (CV-4) rather than a
         * per-operand choice.
         */
        enum class TensorLayout : std::uint8_t
        {
            /** @brief Channels innermost - the single native layout (CV-3). */
            NHWC,
            /** @brief Channels before the spatial axes; a boundary only. */
            NCHW,
        };
```

Add to `struct Value`, after `dtype`:

```cpp
            /**
             * @brief The axis order this value's extents are in.
             *
             * Graph-wide rather than free-floating: see @ref TensorLayout and
             * @c Graph::finalise(), which proves every node preserves it.
             */
            TensorLayout layout = TensorLayout::NHWC;
```

- [ ] **Step 4: Rename, and give the new name its second step**

In `include/SushiAI/graph/ir.hpp`, move the existing `propagate_requires_grad()` body into the private section unchanged, and add the public entry point plus the proof:

```cpp
                /**
                 * @brief Completes the forward graph: propagation, then proof.
                 *
                 * Call once, after the forward graph is complete and before
                 * differentiating. Named for the phase rather than for one of
                 * its steps, so that gaining the layout proof did not make the
                 * name a lie.
                 *
                 * @throws Error If any node other than @c LAYOUT_CONVERT
                 *         produces a result whose layout differs from its
                 *         operands' (CV-4).
                 */
                void finalise()
                {
                    propagate_requires_grad();
                    prove_layout_uniformity();
                }

                /** @brief A mutable reference to value @p id, for passes that rewrite it. */
                Value& mutable_value(ValueId id) { return values_[id]; }

            private:
                /**
                 * @brief Propagates @c requires_grad forward from the leaves.
                 *
                 * A value needs a gradient exactly when some input it descends
                 * from needs one. One forward sweep suffices because insertion
                 * order is topological.
                 */
                void propagate_requires_grad() noexcept
                {
                    for (const Node& n : nodes_)
                    {
                        bool any = false;
                        for (ValueId in : n.inputs)
                            if (values_[in].requires_grad) { any = true; break; }

                        if (any)
                            for (ValueId out : n.outputs)
                                values_[out].requires_grad = true;
                    }
                }

                /**
                 * @brief Proves no node but LAYOUT_CONVERT changes the layout.
                 *
                 * Checked per node rather than by sweeping every value for a
                 * single dominant layout, because the sweep could not express
                 * the one legitimate exemption without also accepting a graph
                 * that is half converted.
                 */
                void prove_layout_uniformity() const
                {
                    for (const Node& n : nodes_)
                    {
                        if (n.op == Ops::LAYOUT_CONVERT.id || n.inputs.empty())
                            continue;

                        const TensorLayout expected = values_[n.inputs.front()].layout;
                        for (ValueId in : n.inputs)
                            SA_THROW_IF(values_[in].layout != expected,
                                        "Node '{}' reads operands in two different layouts; "
                                        "only LAYOUT_CONVERT may cross layouts (CV-4).",
                                        Ops::name_of(n.op));

                        for (ValueId out : n.outputs)
                            SA_THROW_IF(values_[out].layout != expected,
                                        "Node '{}' produces a result whose layout differs "
                                        "from its operands'. Only LAYOUT_CONVERT may do that "
                                        "(CV-4); everything else would silently train the "
                                        "model on transposed data.",
                                        Ops::name_of(n.op));
                    }
                }
```

`prove_layout_uniformity` names `Ops::LAYOUT_CONVERT`, which Task 3 declares. Until Task 3 lands, this file does not compile — so **Tasks 2 and 3 are committed together**, and Task 3's steps below finish what this one starts. Add `#include <SushiAI/core/ops.hpp>` to `ir.hpp`.

- [ ] **Step 5: Move the three call sites**

```bash
# src/autograd/backward.cpp:511, tests/functional/unit/test_ir.cpp:83 and :101
```

Replace `propagate_requires_grad()` with `finalise()` at each. There are exactly three; `grep -rn "propagate_requires_grad" --include=*.cpp --include=*.hpp .` must return only the private definition afterwards.

- [ ] **Step 6: Proceed to Task 3, then return**

The suite cannot build until `Ops::LAYOUT_CONVERT` exists. Do Task 3, then run:

Run: `sa test --suite unit --filter "Unit_Layout|Unit_IR"`
Expected: PASS.

---

# Task 3: The twenty operation identities

**Files:**
- Modify: `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/unit/test_ops.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: every identity named in spec §5, each an `inline constexpr OpName`, all of them in `Ops::ALL`. Later tasks name them as `Ops::IM2COL.id` etc.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/unit/test_ops.cpp`:

```cpp
TEST(Unit_Ops, EverySpatialIdentityIsListedInAll)
{
    // ALL is what all_ids_unique() proves over and what name_of() searches.
    // An identity declared but left out of ALL is not covered by the collision
    // proof and prints as "sushiai.unknown" in every profile - both silent.
    const std::array<Ops::OpID, 20> declared = {
        Ops::RESHAPE.id, Ops::LAYOUT_CONVERT.id,
        Ops::IM2COL.id, Ops::COL2IM.id,
        Ops::MAXPOOL2D.id, Ops::MAXPOOL2D_BACKWARD.id,
        Ops::AVGPOOL2D.id, Ops::AVGPOOL2D_BACKWARD.id,
        Ops::SPATIAL_MEAN.id, Ops::SPATIAL_MEAN_BACKWARD.id,
        Ops::UPSAMPLE2D.id, Ops::UPSAMPLE2D_BACKWARD.id,
        Ops::CONCAT_CHANNELS.id, Ops::SLICE_CHANNELS.id,
        Ops::BATCHNORM2D.id, Ops::BATCHNORM2D_BACKWARD.id,
        Ops::CHANNEL_AFFINE.id,
        Ops::BATCHNORM_UPDATE_STATS.id,
        Ops::LEAKY_RELU.id, Ops::LEAKY_RELU_BACKWARD.id,
    };

    for (Ops::OpID id : declared)
        EXPECT_NE(Ops::name_of(id), "sushiai.unknown")
            << "an identity was declared but not added to Ops::ALL";
}

TEST(Unit_Ops, ThereIsNoConv2dIdentity)
{
    // CV-1: convolution is IM2COL + MATMUL in the IR. A CONV2D identity would
    // be a second, opaque spelling of the same thing, and the first person to
    // add one will do it by analogy rather than by reading the spec.
    EXPECT_EQ(Ops::name_of(Ops::op("sushiai.conv2d").id), "sushiai.unknown");
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_Ops"`
Expected: FAIL to compile — none of the twenty exist.

- [ ] **Step 3: Declare them**

In `include/SushiAI/core/ops.hpp`, after the existing activation-backward block, add:

```cpp
        // --- Structural -----------------------------------------------------

        /** @brief Reinterprets extents without moving data. */
        inline constexpr OpName RESHAPE = op("sushiai.reshape");
        /** @brief The only node permitted to change a value's TensorLayout (CV-4). */
        inline constexpr OpName LAYOUT_CONVERT = op("sushiai.layout_convert");

        // --- Convolution, as CV-1 spells it ---------------------------------
        //
        // There is deliberately no CONV2D. A convolution is IM2COL followed by
        // MATMUL, which is why its weight gradient needs no rule of its own -
        // matmul_backward already produces it.

        /** @brief Gathers each window into a row of a column matrix. */
        inline constexpr OpName IM2COL = op("sushiai.im2col");
        /** @brief The adjoint of IM2COL; a gather, never an atomic scatter (CV-6). */
        inline constexpr OpName COL2IM = op("sushiai.col2im");

        // --- Pooling --------------------------------------------------------

        /** @brief Windowed maximum; also writes the argmax indices. */
        inline constexpr OpName MAXPOOL2D = op("sushiai.maxpool2d");
        /** @brief Routes each gradient back through the recorded argmax. */
        inline constexpr OpName MAXPOOL2D_BACKWARD = op("sushiai.maxpool2d_backward");
        /** @brief Windowed mean, dividing by the window area (count_include_pad). */
        inline constexpr OpName AVGPOOL2D = op("sushiai.avgpool2d");
        /** @brief Spreads each gradient evenly over the window it came from. */
        inline constexpr OpName AVGPOOL2D_BACKWARD = op("sushiai.avgpool2d_backward");

        // --- Whole-plane spatial --------------------------------------------

        /** @brief Global average over H and W: [N,H,W,C] -> [N,C]. */
        inline constexpr OpName SPATIAL_MEAN = op("sushiai.spatial_mean");
        /**
         * @brief Broadcasts an [N,C] gradient over [N,H,W,C], scaled by 1/(H*W).
         *
         * A distinct operation rather than a reuse of BIAS_ADD's or
         * CHANNEL_AFFINE's broadcast: this one is per-(n,c), those are per-row
         * and per-channel. Its absence was a real gap in an earlier draft.
         */
        inline constexpr OpName SPATIAL_MEAN_BACKWARD = op("sushiai.spatial_mean_backward");
        /** @brief Nearest-neighbour upsample by an integer scale. */
        inline constexpr OpName UPSAMPLE2D = op("sushiai.upsample2d");
        /** @brief Sums each source pixel's replicas back together. */
        inline constexpr OpName UPSAMPLE2D_BACKWARD = op("sushiai.upsample2d_backward");

        // --- Structure ------------------------------------------------------

        /** @brief Joins several values along the channel axis. */
        inline constexpr OpName CONCAT_CHANNELS = op("sushiai.concat_channels");
        /**
         * @brief Extracts a channel range.
         *
         * Declared as an operation in its own right rather than as
         * CONCAT_CHANNELS's private backward, because it is independently
         * reachable: Darknet's route layer with a group index is exactly this.
         */
        inline constexpr OpName SLICE_CHANNELS = op("sushiai.slice_channels");

        // --- Normalisation --------------------------------------------------

        /** @brief Batch normalisation's forward pass over an NHWC value. */
        inline constexpr OpName BATCHNORM2D = op("sushiai.batchnorm2d");
        /** @brief Batch normalisation's gradient, for input, gamma and beta. */
        inline constexpr OpName BATCHNORM2D_BACKWARD = op("sushiai.batchnorm2d_backward");
        /** @brief Per-channel scale and shift: y = gamma * x + beta. */
        inline constexpr OpName CHANNEL_AFFINE = op("sushiai.channel_affine");
        /**
         * @brief Updates a BatchNorm's running statistics.
         *
         * Not part of the differentiated graph (CV-7): it reads batch
         * statistics and writes buffers no gradient flows through, so it is
         * emitted into the training step beside the graph rather than into it.
         */
        inline constexpr OpName BATCHNORM_UPDATE_STATS = op("sushiai.batchnorm_update_stats");

        // --- Activation -----------------------------------------------------

        /** @brief Leaky ReLU; its slope is attribute slot 0. */
        inline constexpr OpName LEAKY_RELU = op("sushiai.leaky_relu");
        /** @brief Gradient of LEAKY_RELU; second operand is the forward output. */
        inline constexpr OpName LEAKY_RELU_BACKWARD = op("sushiai.leaky_relu_backward");
```

- [ ] **Step 4: Add every one of them to `ALL`**

Extend the `ALL` initialiser (`ops.hpp:272`):

```cpp
        inline constexpr std::array ALL = {
            LEAF, ACCUMULATE, CAST,
            MATMUL, BIAS_ADD,
            ADD, SUB, MUL, SCALE, DYNAMIC_SCALE, EXP, ZERO,
            RELU, GELU, SIGMOID, TANH, LEAKY_RELU,
            RELU_BACKWARD, GELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
            LEAKY_RELU_BACKWARD,
            LOG_SOFTMAX, NLL_LOSS,
            SUM_ROWS,
            ADAMW_UPDATE, SGD_UPDATE,
            GEMM_BIAS, GEMM_BIAS_RELU, GEMM_BIAS_SIGMOID, GEMM_BIAS_TANH, GEMM_BIAS_GELU,
            EXP_SUB_SCALE,
            RESHAPE, LAYOUT_CONVERT,
            IM2COL, COL2IM,
            MAXPOOL2D, MAXPOOL2D_BACKWARD, AVGPOOL2D, AVGPOOL2D_BACKWARD,
            SPATIAL_MEAN, SPATIAL_MEAN_BACKWARD, UPSAMPLE2D, UPSAMPLE2D_BACKWARD,
            CONCAT_CHANNELS, SLICE_CHANNELS,
            BATCHNORM2D, BATCHNORM2D_BACKWARD, CHANNEL_AFFINE, BATCHNORM_UPDATE_STATS,
        };
```

- [ ] **Step 5: Run the tests**

Run: `sa test --suite unit --filter "Unit_Ops|Unit_Layout|Unit_IR"`
Expected: PASS. `all_ids_unique()`'s `static_assert` now covers fifty-odd names; if it fires, two names collided and the fix is to rename one, never to remove it from `ALL`.

- [ ] **Step 6: Full suite**

Run: `sa test --suite all`
Expected: PASS.

---

# Task 4: `LEAKY_RELU` joins the activation table

**Files:**
- Modify: `include/SushiAI/core/ops.hpp` (`ACTIVATIONS`)
- Test: `tests/functional/unit/test_ops.cpp`, `tests/functional/unit/test_fusion.cpp`

**Interfaces:**
- Consumes: Task 3's `LEAKY_RELU`, `LEAKY_RELU_BACKWARD`.
- Produces: `ACTIVATIONS` sized 5; `activation_of(Ops::LEAKY_RELU.id)` non-null. Task 15's rule reads it.

- [ ] **Step 1: Write the failing test**

```cpp
TEST(Unit_Ops, LeakyReluDifferentiatesAtItsOutput)
{
    // For alpha > 0, leaky_relu(x) > 0 exactly when x > 0, so reading the
    // output instead of the input is a bit-identical substitution - the same
    // argument the table already records for RELU, not an approximation.
    // It is what makes the fold free: an ordinary use count licenses it with
    // no activation-specific knowledge in the fusion pass.
    const Ops::Activation* entry = Ops::activation_of(Ops::LEAKY_RELU.id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->backward, Ops::LEAKY_RELU_BACKWARD.id);
    EXPECT_EQ(entry->gradient_source, Ops::GradientSource::OUTPUT);
    EXPECT_TRUE(Ops::folds_without_spill(Ops::LEAKY_RELU.id));
}
```

Use the real spelling of the "no spill needed" predicate — read `ops.hpp` around line 415 and match it; the plan calls it `folds_without_spill` from its doc comment, and if the identifier differs, the identifier wins.

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_Ops.LeakyRelu"`
Expected: FAIL — `activation_of` returns null.

- [ ] **Step 3: Add the row**

```cpp
        /** @brief Every activation this library differentiates, in one table. */
        inline constexpr std::array<Activation, 5> ACTIVATIONS = {{
            {RELU.id,       RELU_BACKWARD.id,       GradientSource::OUTPUT},
            {GELU.id,       GELU_BACKWARD.id,       GradientSource::INPUT},
            {SIGMOID.id,    SIGMOID_BACKWARD.id,    GradientSource::OUTPUT},
            {TANH.id,       TANH_BACKWARD.id,       GradientSource::OUTPUT},
            {LEAKY_RELU.id, LEAKY_RELU_BACKWARD.id, GradientSource::OUTPUT},
        }};
```

- [ ] **Step 4: Run the fusion proof**

Run: `sa test --suite unit --filter "Unit_Ops|Unit_Fusion"`
Expected: PASS. `graph/fusion.cpp`'s `every_fused_activation_survives()` is a `static_assert`, so a violation is a build failure rather than a test failure — a clean build **is** the assertion here.

---

# Task 5: Shape inference — `im2col` and `col2im`

CV-1's whole argument rests on this: if the geometry is wrong, it must throw here, at build time, not produce a plausible wrong number.

**Files:**
- Modify: `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`
- Test: `tests/functional/unit/test_spatial_shapes.cpp` (create), registered in `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  ```cpp
  struct ConvGeometry { int64_t kernel_h, kernel_w, stride_h, stride_w,
                        dilation_h, dilation_w, pad_top, pad_left; };
  ValueId im2col(ValueId x, const ConvGeometry& g) const;
  ValueId col2im(ValueId cols, Shape image_shape, const ConvGeometry& g) const;
  ```
  Tasks 11 and 16 call both.

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_spatial_shapes.cpp`:

```cpp
namespace SushiAI
{
    namespace Test
    {
        using Graph::GraphBuilder;
        using Graph::ValueId;

        TEST(Unit_SpatialShapes, Im2ColProducesTheColumnMatrixAGemmExpects)
        {
            // [N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5
            // windows per image, and each row is one window's kernel_h *
            // kernel_w * C taps - channels innermost, so a sub-group reads a
            // contiguous run (CV-3).
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId cols = builder.im2col(x, g);

            EXPECT_EQ(graph.values()[cols].shape, Shape({2 * 5 * 5, 3 * 3 * 3}));
        }

        TEST(Unit_SpatialShapes, Im2ColHonoursStrideAndDilation)
        {
            // A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows;
            // stride 2 over the same input yields 2. Both numbers are wrong
            // under the naive (H - K)/S + 1 that forgets dilation.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 7, 7, 2}, "x");

            Graph::ConvGeometry dilated;
            dilated.kernel_h = 3; dilated.kernel_w = 3;
            dilated.dilation_h = 2; dilated.dilation_w = 2;
            EXPECT_EQ(graph.values()[builder.im2col(x, dilated)].shape,
                      Shape({1 * 3 * 3, 3 * 3 * 2}));

            Graph::ConvGeometry strided;
            strided.kernel_h = 3; strided.kernel_w = 3;
            strided.stride_h = 2; strided.stride_w = 2;
            EXPECT_EQ(graph.values()[builder.im2col(x, strided)].shape,
                      Shape({1 * 3 * 3, 3 * 3 * 2}));
        }

        TEST(Unit_SpatialShapes, Im2ColRejectsAWindowLargerThanThePaddedInput)
        {
            // Produces a zero extent, which Shape refuses anyway - but the
            // message a caller gets should name the geometry, not the Shape
            // invariant it happened to trip.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 1}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 5; g.kernel_w = 5;

            EXPECT_THROW((void)builder.im2col(x, g), Error);
        }

        TEST(Unit_SpatialShapes, Im2ColRejectsANonSpatialInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({4, 8}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 1; g.kernel_w = 1;

            EXPECT_THROW((void)builder.im2col(x, g), Error);
        }

        TEST(Unit_SpatialShapes, Col2ImReturnsToTheImageItWasAskedFor)
        {
            // col2im cannot derive its output extents: ConvGeometry carries
            // pad_top and pad_left but no pad_bottom or pad_right, so an
            // asymmetric padding would be silently mis-shaped. The caller
            // states the image shape instead - it always has it, because it
            // held the value im2col consumed.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({2 * 5 * 5, 3 * 3 * 3}, "cols");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId image = builder.col2im(cols, Shape({2, 5, 5, 3}), g);

            EXPECT_EQ(graph.values()[image].shape, Shape({2, 5, 5, 3}));
        }

        TEST(Unit_SpatialShapes, Col2ImRejectsAColumnMatrixThatDoesNotMatchTheImage)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({2 * 5 * 5, 3 * 3 * 3}, "cols");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            EXPECT_THROW((void)builder.col2im(cols, Shape({2, 6, 6, 3}), g), Error);
        }
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_SpatialShapes"`
Expected: FAIL to compile — `ConvGeometry`, `im2col` and `col2im` do not exist.

- [ ] **Step 3: Declare the geometry and the emitters**

In `include/SushiAI/graph/builder.hpp`, before `class GraphBuilder`:

```cpp
        /**
         * @struct ConvGeometry
         * @brief A 2-D window's shape and placement, in CV-5's eight slots.
         *
         * Mirrors SushiBLAS's @c Spatial::ConvGeometry field for field, and for
         * the same reason it carries no @c pad_bottom or @c pad_right: those
         * only ever decide the output extent, which the host computes and
         * states outright. Deriving them back from an extent would assume
         * symmetric padding, which YOLOv3-tiny's 13->13 pool violates.
         */
        struct ConvGeometry
        {
            int64_t kernel_h = 1;   /**< @brief Window height, in taps. */
            int64_t kernel_w = 1;   /**< @brief Window width, in taps. */
            int64_t stride_h = 1;   /**< @brief Vertical step between windows. */
            int64_t stride_w = 1;   /**< @brief Horizontal step between windows. */
            int64_t dilation_h = 1; /**< @brief Vertical spacing between taps. */
            int64_t dilation_w = 1; /**< @brief Horizontal spacing between taps. */
            int64_t pad_top = 0;    /**< @brief Rows of zeros above the input. */
            int64_t pad_left = 0;   /**< @brief Columns of zeros left of the input. */
        };

        /**
         * @brief The number of windows one axis yields.
         *
         * The one formula for this in SushiAI, so that a layer's shape
         * inference and the geometry it hands SushiBLAS cannot disagree.
         *
         * @param in The input extent.
         * @param pad_before Padding before the first element.
         * @param pad_after Padding after the last.
         * @param dilation The spacing between taps.
         * @param kernel The number of taps.
         * @param stride The step between windows.
         * @return The output extent, or 0 when the window does not fit.
         */
        constexpr int64_t output_extent(int64_t in, int64_t pad_before, int64_t pad_after,
                                        int64_t dilation, int64_t kernel,
                                        int64_t stride) noexcept
        {
            if (stride <= 0 || kernel <= 0 || dilation <= 0) return 0;
            const int64_t covered = dilation * (kernel - 1) + 1;
            const int64_t padded = in + pad_before + pad_after;
            if (padded < covered) return 0;
            return (padded - covered) / stride + 1;
        }
```

Inside `class GraphBuilder`'s public section:

```cpp
                /**
                 * @brief Emits @c IM2COL: each window becomes one row of a matrix.
                 *
                 * The result is @c [N*H_o*W_o, kernel_h*kernel_w*C], which is
                 * exactly the left operand of the GEMM a convolution is (CV-1).
                 * Padding is assumed symmetric here and only here, because a
                 * caller that needs otherwise states its own extents through
                 * @ref im2col_with_extents.
                 *
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The column matrix's id.
                 * @throws Error If @p x is not rank 4, or the window does not
                 *         fit the padded input.
                 */
                ValueId im2col(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c IM2COL with the output extents stated outright.
                 *
                 * The overload an asymmetrically-padded layer uses.
                 *
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @param out_h The number of window rows.
                 * @param out_w The number of window columns.
                 * @return The column matrix's id.
                 * @throws Error If @p x is not rank 4 or either extent is not positive.
                 */
                ValueId im2col_with_extents(ValueId x, const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c COL2IM: the adjoint of @ref im2col.
                 *
                 * A gather over the input pixels, never an atomic scatter
                 * (CV-6), which is what makes it bit-reproducible.
                 *
                 * @param cols The column matrix.
                 * @param image_shape The @c [N,H,W,C] shape to return to.
                 * @param g The window geometry @p cols was produced with.
                 * @return The image's id.
                 * @throws Error If @p image_shape is not rank 4, or @p cols
                 *         does not have the row and column counts that
                 *         geometry implies.
                 */
                ValueId col2im(ValueId cols, Shape image_shape, const ConvGeometry& g) const;
```

- [ ] **Step 4: Define them**

In `src/graph/builder.cpp`:

```cpp
        namespace
        {
            /** @brief Writes a ConvGeometry into CV-5's eight attribute slots. */
            Attributes geometry_attributes(const ConvGeometry& g) noexcept
            {
                Attributes a;
                a.values[0] = static_cast<double>(g.kernel_h);
                a.values[1] = static_cast<double>(g.kernel_w);
                a.values[2] = static_cast<double>(g.stride_h);
                a.values[3] = static_cast<double>(g.stride_w);
                a.values[4] = static_cast<double>(g.dilation_h);
                a.values[5] = static_cast<double>(g.dilation_w);
                a.values[6] = static_cast<double>(g.pad_top);
                a.values[7] = static_cast<double>(g.pad_left);
                return a;
            }
        }

        ValueId GraphBuilder::im2col(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());

            // Symmetric padding is assumed here and stated in the header. A
            // layer that pads asymmetrically calls im2col_with_extents.
            const int64_t out_h = output_extent(in.shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);

            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "A {}x{} window dilated by {}x{} covers more than the "
                        "{}x{} input padded by {},{}; it yields no windows.",
                        g.kernel_h, g.kernel_w, g.dilation_h, g.dilation_w,
                        in.shape[1], in.shape[2], g.pad_top, g.pad_left);

            return im2col_with_extents(x, g, out_h, out_w);
        }

        ValueId GraphBuilder::im2col_with_extents(ValueId x, const ConvGeometry& g,
                                                  int64_t out_h, int64_t out_w) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "im2col was given non-positive output extents {}x{}", out_h, out_w);

            const int64_t rows = in.shape[0] * out_h * out_w;
            const int64_t cols = g.kernel_h * g.kernel_w * in.shape[3];

            const ValueId out = declare(Shape({rows, cols}), in.dtype,
                                        Lifetime::ACTIVATION, "cols");
            graph_.value(out).layout = in.layout;
            emit(Ops::IM2COL.id, {x}, {out}, geometry_attributes(g));
            return out;
        }

        ValueId GraphBuilder::col2im(ValueId cols, Shape image_shape,
                                     const ConvGeometry& g) const
        {
            const Value& c = graph_.values()[cols];
            SA_THROW_IF(image_shape.rank() != 4,
                        "col2im returns a rank-4 [N,H,W,C] image; got rank {}",
                        image_shape.rank());
            SA_THROW_IF(c.shape.rank() != 2,
                        "col2im reads a rank-2 column matrix; got rank {}", c.shape.rank());

            const int64_t out_h = output_extent(image_shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(image_shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);
            const int64_t rows = image_shape[0] * out_h * out_w;
            const int64_t width = g.kernel_h * g.kernel_w * image_shape[3];

            SA_THROW_IF(c.shape[0] != rows || c.shape[1] != width,
                        "col2im was given a [{}, {}] column matrix, but that geometry "
                        "over a [{}, {}, {}, {}] image implies [{}, {}].",
                        c.shape[0], c.shape[1], image_shape[0], image_shape[1],
                        image_shape[2], image_shape[3], rows, width);

            const ValueId out = declare(image_shape, c.dtype, Lifetime::ACTIVATION, "image");
            graph_.value(out).layout = c.layout;
            emit(Ops::COL2IM.id, {cols}, {out}, geometry_attributes(g));
            return out;
        }
```

Match `declare` and `emit` to their real spellings in `builder.cpp` — read the file's existing private helpers first and follow them exactly rather than introducing a second convention.

- [ ] **Step 5: Run the tests**

Run: `sa test --suite unit --filter "Unit_SpatialShapes"`
Expected: PASS, all six.

- [ ] **Step 6: Full suite**

Run: `sa test --suite all`
Expected: PASS.

# Task 6: Shape inference — pooling and upsample

**Files:**
- Modify: `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`
- Test: `tests/functional/unit/test_spatial_shapes.cpp`

**Interfaces:**
- Consumes: Task 5's `ConvGeometry` and `output_extent`.
- Produces:
  ```cpp
  struct PoolResult { ValueId output; ValueId indices; };
  PoolResult maxpool2d(ValueId x, const ConvGeometry& g) const;
  ValueId avgpool2d(ValueId x, const ConvGeometry& g) const;
  ValueId upsample2d(ValueId x, int64_t scale_h, int64_t scale_w) const;
  ```
  Task 12's rules call all three.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/unit/test_spatial_shapes.cpp`:

```cpp
        TEST(Unit_SpatialShapes, MaxPoolProducesAValueAndItsArgmaxTogether)
        {
            // The indices are a second *result*, not a second output tensor the
            // caller allocates: the backward rule needs exactly the indices
            // this forward produced, and a pair returned together cannot be
            // mismatched the way two separate calls could.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4, 4, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 2; g.kernel_w = 2;
            g.stride_h = 2; g.stride_w = 2;

            const Graph::PoolResult pooled = builder.maxpool2d(x, g);

            EXPECT_EQ(graph.values()[pooled.output].shape, Shape({2, 2, 2, 3}));
            EXPECT_EQ(graph.values()[pooled.indices].shape, Shape({2, 2, 2, 3}));
            EXPECT_EQ(graph.values()[pooled.indices].dtype, DType::INT64)
                << "an index is not a number to interpolate; a float index would "
                   "compare unequal to the integer the forward wrote";
        }

        TEST(Unit_SpatialShapes, MaxPoolHandlesYoloTinysAsymmetricThirteenToThirteen)
        {
            // YOLOv3-tiny's last pool: 2x2 stride 1 over 13x13, padded only on
            // the bottom and right, staying 13x13. ConvGeometry cannot express
            // that padding, so the caller states the extents - and this is the
            // case that proves the explicit overload is load-bearing rather
            // than decorative.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 13, 13, 256}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 2; g.kernel_w = 2;

            const Graph::PoolResult pooled = builder.maxpool2d_with_extents(x, g, 13, 13);
            EXPECT_EQ(graph.values()[pooled.output].shape, Shape({1, 13, 13, 256}));

            // Without the explicit extents the same geometry gives 12x12.
            EXPECT_EQ(graph.values()[builder.maxpool2d(x, g).output].shape,
                      Shape({1, 12, 12, 256}));
        }

        TEST(Unit_SpatialShapes, AvgPoolProducesOnlyAValue)
        {
            // No indices: its backward divides by a constant window area and
            // needs nothing the forward recorded (CV-6).
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 6, 6, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.stride_h = 3; g.stride_w = 3;

            EXPECT_EQ(graph.values()[builder.avgpool2d(x, g)].shape, Shape({2, 2, 2, 3}));
        }

        TEST(Unit_SpatialShapes, UpsampleMultipliesTheSpatialExtentsOnly)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 5, 7}, "x");

            EXPECT_EQ(graph.values()[builder.upsample2d(x, 2, 2)].shape, Shape({2, 6, 10, 7}));
        }

        TEST(Unit_SpatialShapes, UpsampleRejectsANonPositiveScale)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 1}, "x");

            EXPECT_THROW((void)builder.upsample2d(x, 0, 2), Error);
        }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_SpatialShapes"`
Expected: FAIL to compile.

- [ ] **Step 3: Declare them**

In `builder.hpp`, before `class GraphBuilder`:

```cpp
        /**
         * @struct PoolResult
         * @brief A max-pool's two results, returned together.
         *
         * Returned as a pair rather than left to two calls because the backward
         * rule must route through exactly the indices *this* forward wrote.
         * Two independent values could be mismatched, and the mismatch would
         * not throw - it would route every gradient to a plausible wrong pixel.
         */
        struct PoolResult
        {
            ValueId output;   /**< @brief The pooled @c [N,H_o,W_o,C] value. */
            ValueId indices;  /**< @brief The INT64 argmax, same shape. */
        };
```

Inside `GraphBuilder`:

```cpp
                /**
                 * @brief Emits @c MAXPOOL2D, with symmetric padding assumed.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value and its argmax indices.
                 * @throws Error If @p x is not rank 4 or the window does not fit.
                 */
                PoolResult maxpool2d(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c MAXPOOL2D with the output extents stated outright.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @param out_h The pooled height.
                 * @param out_w The pooled width.
                 * @return The pooled value and its argmax indices.
                 * @throws Error If @p x is not rank 4 or either extent is not positive.
                 */
                PoolResult maxpool2d_with_extents(ValueId x, const ConvGeometry& g,
                                                  int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c AVGPOOL2D. Divides by the full window area.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value's id.
                 * @throws Error If @p x is not rank 4 or the window does not fit.
                 */
                ValueId avgpool2d(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c UPSAMPLE2D: nearest-neighbour, integer scale.
                 * @param x The @c [N,H,W,C] input.
                 * @param scale_h The vertical replication factor.
                 * @param scale_w The horizontal replication factor.
                 * @return The upsampled value's id.
                 * @throws Error If @p x is not rank 4 or either scale is below one.
                 */
                ValueId upsample2d(ValueId x, int64_t scale_h, int64_t scale_w) const;
```

- [ ] **Step 4: Define them**

In `src/graph/builder.cpp`:

```cpp
        PoolResult GraphBuilder::maxpool2d(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());

            const int64_t out_h = output_extent(in.shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "A {}x{} pool window does not fit a {}x{} input padded by {},{}.",
                        g.kernel_h, g.kernel_w, in.shape[1], in.shape[2],
                        g.pad_top, g.pad_left);

            return maxpool2d_with_extents(x, g, out_h, out_w);
        }

        PoolResult GraphBuilder::maxpool2d_with_extents(ValueId x, const ConvGeometry& g,
                                                        int64_t out_h, int64_t out_w) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "maxpool2d was given non-positive output extents {}x{}",
                        out_h, out_w);

            const Shape pooled({in.shape[0], out_h, out_w, in.shape[3]});
            const ValueId out = declare(pooled, in.dtype, Lifetime::ACTIVATION, "pooled");
            const ValueId idx = declare(pooled, DType::INT64, Lifetime::ACTIVATION, "argmax");
            graph_.value(out).layout = in.layout;
            graph_.value(idx).layout = in.layout;

            emit(Ops::MAXPOOL2D.id, {x}, {out, idx}, geometry_attributes(g));
            return PoolResult{out, idx};
        }

        ValueId GraphBuilder::avgpool2d(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "avgpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());

            const int64_t out_h = output_extent(in.shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "A {}x{} pool window does not fit a {}x{} input padded by {},{}.",
                        g.kernel_h, g.kernel_w, in.shape[1], in.shape[2],
                        g.pad_top, g.pad_left);

            const ValueId out = declare(Shape({in.shape[0], out_h, out_w, in.shape[3]}),
                                        in.dtype, Lifetime::ACTIVATION, "pooled");
            graph_.value(out).layout = in.layout;
            emit(Ops::AVGPOOL2D.id, {x}, {out}, geometry_attributes(g));
            return out;
        }

        ValueId GraphBuilder::upsample2d(ValueId x, int64_t scale_h, int64_t scale_w) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "upsample2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(scale_h < 1 || scale_w < 1,
                        "upsample2d needs scales of at least one; got {}x{}",
                        scale_h, scale_w);

            const ValueId out = declare(Shape({in.shape[0], in.shape[1] * scale_h,
                                               in.shape[2] * scale_w, in.shape[3]}),
                                        in.dtype, Lifetime::ACTIVATION, "upsampled");
            graph_.value(out).layout = in.layout;

            Attributes a;
            a.values[0] = static_cast<double>(scale_h);
            a.values[1] = static_cast<double>(scale_w);
            emit(Ops::UPSAMPLE2D.id, {x}, {out}, a);
            return out;
        }
```

- [ ] **Step 5: Run the tests**

Run: `sa test --suite unit --filter "Unit_SpatialShapes"`
Expected: PASS, all eleven.

---

# Task 7: Shape inference — channels, normalisation, structure

**Files:**
- Modify: `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`
- Test: `tests/functional/unit/test_spatial_shapes.cpp`

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces:
  ```cpp
  ValueId spatial_mean(ValueId x) const;
  ValueId concat_channels(SushiRuntime::span<const ValueId> parts) const;
  ValueId slice_channels(ValueId x, int64_t offset, int64_t count) const;
  ValueId channel_affine(ValueId x, ValueId gamma, ValueId beta) const;
  ValueId batchnorm2d(ValueId x, ValueId gamma, ValueId beta, double epsilon) const;
  ValueId reshape(ValueId x, Shape shape) const;
  ValueId layout_convert(ValueId x, TensorLayout to) const;
  ```
  Tasks 10, 13 and 14 call these; Task 2's `Unit_Layout.LayoutConvertIsAllowedToChangeIt` already depends on `layout_convert`.

- [ ] **Step 1: Write the failing test**

```cpp
        TEST(Unit_SpatialShapes, SpatialMeanReducesToNAndC)
        {
            // The [N,C] the numeric Probe's loss needs, which is why every
            // convolutional gradient test in this plan ends with this call.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({4, 7, 7, 512}, "x");

            EXPECT_EQ(graph.values()[builder.spatial_mean(x)].shape, Shape({4, 512}));
        }

        TEST(Unit_SpatialShapes, ConcatSumsTheChannelAxisAndNothingElse)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId a = builder.input({2, 8, 8, 16}, "a");
            const ValueId b = builder.input({2, 8, 8, 32}, "b");
            const std::array<ValueId, 2> parts = {a, b};

            EXPECT_EQ(graph.values()[builder.concat_channels(
                          SushiRuntime::span<const ValueId>(parts.data(), parts.size()))].shape,
                      Shape({2, 8, 8, 48}));
        }

        TEST(Unit_SpatialShapes, ConcatRejectsOperandsThatDisagreeOffTheChannelAxis)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId a = builder.input({2, 8, 8, 16}, "a");
            const ValueId b = builder.input({2, 4, 8, 16}, "b");
            const std::array<ValueId, 2> parts = {a, b};

            EXPECT_THROW((void)builder.concat_channels(
                SushiRuntime::span<const ValueId>(parts.data(), parts.size())), Error);
        }

        TEST(Unit_SpatialShapes, SliceTakesAChannelRangeAndRefusesToOverrun)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 48}, "x");

            EXPECT_EQ(graph.values()[builder.slice_channels(x, 16, 32)].shape,
                      Shape({2, 8, 8, 32}));
            EXPECT_THROW((void)builder.slice_channels(x, 32, 32), Error);
        }

        TEST(Unit_SpatialShapes, ChannelAffineKeepsTheShapeAndChecksTheParameters)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x");
            const ValueId gamma = builder.parameter({16}, "gamma");
            const ValueId beta = builder.parameter({16}, "beta");

            EXPECT_EQ(graph.values()[builder.channel_affine(x, gamma, beta)].shape,
                      Shape({2, 8, 8, 16}));

            const ValueId wrong = builder.parameter({8}, "wrong");
            EXPECT_THROW((void)builder.channel_affine(x, wrong, beta), Error);
        }

        TEST(Unit_SpatialShapes, ReshapeRefusesToChangeTheElementCount)
        {
            // The one failure mode a reshape has: it moves no data, so a
            // mismatched count is not a resize, it is reading past the end.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4}, "x");

            EXPECT_EQ(graph.values()[builder.reshape(x, Shape({6, 4}))].shape, Shape({6, 4}));
            EXPECT_THROW((void)builder.reshape(x, Shape({6, 5})), Error);
        }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_SpatialShapes"`
Expected: FAIL to compile.

- [ ] **Step 3: Declare and define**

Declarations in `builder.hpp` follow the pattern of Tasks 5–6 exactly: `@brief`, one `@param` per argument, `@return`, `@throws`. Definitions in `src/graph/builder.cpp`:

```cpp
        ValueId GraphBuilder::spatial_mean(ValueId x) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "spatial_mean needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());

            const ValueId out = declare(Shape({in.shape[0], in.shape[3]}), in.dtype,
                                        Lifetime::ACTIVATION, "spatial_mean");
            graph_.value(out).layout = in.layout;
            emit(Ops::SPATIAL_MEAN.id, {x}, {out}, Attributes{});
            return out;
        }

        ValueId GraphBuilder::concat_channels(SushiRuntime::span<const ValueId> parts) const
        {
            SA_THROW_IF(parts.empty(), "concat_channels needs at least one operand");

            const Value& first = graph_.values()[parts[0]];
            SA_THROW_IF(first.shape.rank() != 4,
                        "concat_channels needs rank-4 operands; got rank {}",
                        first.shape.rank());

            int64_t channels = 0;
            for (ValueId id : parts)
            {
                const Value& p = graph_.values()[id];
                SA_THROW_IF(p.shape.rank() != 4,
                            "concat_channels needs rank-4 operands; got rank {}",
                            p.shape.rank());
                SA_THROW_IF(p.shape[0] != first.shape[0] || p.shape[1] != first.shape[1]
                                || p.shape[2] != first.shape[2],
                            "concat_channels operands must agree on N, H and W; "
                            "got [{}, {}, {}, *] against [{}, {}, {}, *].",
                            p.shape[0], p.shape[1], p.shape[2],
                            first.shape[0], first.shape[1], first.shape[2]);
                SA_THROW_IF(p.dtype != first.dtype,
                            "concat_channels operands must share a dtype");
                channels += p.shape[3];
            }

            const ValueId out = declare(
                Shape({first.shape[0], first.shape[1], first.shape[2], channels}),
                first.dtype, Lifetime::ACTIVATION, "concat");
            graph_.value(out).layout = first.layout;

            emit(Ops::CONCAT_CHANNELS.id,
                 std::vector<ValueId>(parts.begin(), parts.end()), {out}, Attributes{});
            return out;
        }

        ValueId GraphBuilder::slice_channels(ValueId x, int64_t offset, int64_t count) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "slice_channels needs a rank-4 input; got rank {}", in.shape.rank());
            SA_THROW_IF(offset < 0 || count <= 0 || offset + count > in.shape[3],
                        "slice_channels [{}, {}) does not fit {} channels",
                        offset, offset + count, in.shape[3]);

            const ValueId out = declare(
                Shape({in.shape[0], in.shape[1], in.shape[2], count}),
                in.dtype, Lifetime::ACTIVATION, "slice");
            graph_.value(out).layout = in.layout;

            Attributes a;
            a.values[0] = static_cast<double>(offset);
            emit(Ops::SLICE_CHANNELS.id, {x}, {out}, a);
            return out;
        }

        ValueId GraphBuilder::channel_affine(ValueId x, ValueId gamma, ValueId beta) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "channel_affine needs a rank-4 input; got rank {}", in.shape.rank());

            for (ValueId p : {gamma, beta})
            {
                const Value& v = graph_.values()[p];
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "channel_affine's per-channel parameters must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
            }

            const ValueId out = declare(in.shape, in.dtype, Lifetime::ACTIVATION, "affine");
            graph_.value(out).layout = in.layout;
            emit(Ops::CHANNEL_AFFINE.id, {x, gamma, beta}, {out}, Attributes{});
            return out;
        }

        ValueId GraphBuilder::reshape(ValueId x, Shape shape) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.element_count() != shape.element_count(),
                        "reshape moves no data, so it cannot change the element count: "
                        "{} elements cannot be seen as {}.",
                        in.shape.element_count(), shape.element_count());

            const ValueId out = declare(shape, in.dtype, Lifetime::ACTIVATION, "reshape");
            graph_.value(out).layout = in.layout;
            emit(Ops::RESHAPE.id, {x}, {out}, Attributes{});
            return out;
        }

        ValueId GraphBuilder::layout_convert(ValueId x, TensorLayout to) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "layout_convert needs a rank-4 input; got rank {}", in.shape.rank());

            // The extents move with the axes: NHWC [N,H,W,C] is NCHW [N,C,H,W].
            const Shape converted = (to == TensorLayout::NCHW)
                ? Shape({in.shape[0], in.shape[3], in.shape[1], in.shape[2]})
                : Shape({in.shape[0], in.shape[2], in.shape[3], in.shape[1]});

            const ValueId out = declare(converted, in.dtype, Lifetime::ACTIVATION, "converted");
            graph_.value(out).layout = to;
            emit(Ops::LAYOUT_CONVERT.id, {x}, {out}, Attributes{});
            return out;
        }
```

`batchnorm2d` is declared here but its emitter is written in Task 14 alongside its rule, because its result set (`y`, plus the saved mean and inverse standard deviation its backward reads) is decided by that rule.

Use `Shape::element_count()`'s real spelling; read `include/SushiAI/core/shape.hpp` and match it.

- [ ] **Step 4: Run the tests**

Run: `sa test --suite unit --filter "Unit_SpatialShapes|Unit_Layout"`
Expected: PASS. `Unit_Layout.LayoutConvertIsAllowedToChangeIt`, written back in Task 2, now passes for the first time.

- [ ] **Step 5: Full suite**

Run: `sa test --suite all`
Expected: PASS.

---

# Task 8: Extract the tape and the rule vocabulary

Spec §6 orders the restructuring **before** any new rule. This task and Task 9 are that restructuring, and they add no behaviour: the suite must be green at the end with exactly the same assertions it had at the start.

**Files:**
- Create: `include/SushiAI/autograd/tape.hpp`, `include/SushiAI/autograd/rule.hpp`, `src/autograd/tape.cpp`
- Modify: `include/SushiAI/autograd/backward.hpp`, `src/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Autograd::GradientTape` (moved verbatim), `Autograd::BackwardRule`, `Autograd::RuleEntry`, and `Autograd::concat_rules`. Task 9 and every rule task include `rule.hpp`.

- [ ] **Step 1: Move `GradientTape` into its own header**

Cut `class GradientTape` (`backward.hpp:89-200`) and the `class Differentiator;` forward declaration into a new `include/SushiAI/autograd/tape.hpp`, unchanged. Its file comment states what the header is for:

```cpp
// The only interface a backward rule sees.
//
// A rule does not name Differentiator: the dependency is inverted, so the
// traversal can change without touching a single rule, and a rule cannot reach
// into the traversal's bookkeeping even by accident. That was already true when
// this class lived in backward.hpp; giving it its own header is what makes it
// visible, and what lets a rules/*.cpp include the interface without also
// including differentiate()'s declaration.
```

`backward.hpp` includes `tape.hpp` and keeps `differentiate()` and `has_backward_rule()`.

- [ ] **Step 2: Write the rule vocabulary**

Create `include/SushiAI/autograd/rule.hpp`:

```cpp
namespace SushiAI
{
    namespace Autograd
    {
        /** @brief What every backward rule looks like. */
        using BackwardRule = void (*)(const GradientTape&);

        /**
         * @struct RuleEntry
         * @brief One operation's backward rule, paired with its identity.
         */
        struct RuleEntry
        {
            OpID op;             /**< @brief The forward operation. */
            BackwardRule rule;   /**< @brief What differentiates it. */
        };

        /**
         * @brief Concatenates two rule tables at compile time.
         *
         * The mechanism behind @c rules/all.hpp. Nothing self-registers:
         * @c sushiai is a static library, so a translation unit whose only job
         * is to run a registration constructor resolves no symbol and the
         * linker drops it - silently, and only in a downstream consumer's
         * build, which is the worst possible place to discover a missing
         * gradient. Taking the address of a function declared in a domain
         * header is an address constant expression, so the concatenated table
         * stays @c constexpr and lookup stays a compile-time walk.
         *
         * @tparam N The left table's size.
         * @tparam M The right table's size.
         * @param a The left table.
         * @param b The right table.
         * @return Their concatenation.
         */
        template<std::size_t N, std::size_t M>
        constexpr std::array<RuleEntry, N + M> concat_rules(
            const std::array<RuleEntry, N>& a, const std::array<RuleEntry, M>& b)
        {
            std::array<RuleEntry, N + M> out{};
            for (std::size_t i = 0; i < N; ++i) out[i] = a[i];
            for (std::size_t i = 0; i < M; ++i) out[N + i] = b[i];
            return out;
        }
    }
}
```

- [ ] **Step 3: Move the tape's out-of-line methods**

Create `src/autograd/tape.cpp` and move `GradientTape::node()`, `output_grad()`, `output_is_root()`, `input_requires_grad()`, `contribute()`, `contribute_input()`, `skip()` and `apply_seed_factor()` into it from `backward.cpp`, unchanged. Add `autograd/tape.cpp` to `src/CMakeLists.txt` beside `autograd/backward.cpp`.

- [ ] **Step 4: Build and run everything**

Run: `sa build`
Expected: SUCCESS.

Run: `sa test --suite all`
Expected: PASS, with the same number of tests as before this task. A moved definition that changed behaviour would show here; nothing in this task is allowed to.

---

# Task 9: One translation unit per domain

**Files:**
- Create: `include/SushiAI/autograd/rules/{elementwise,linalg,activation,loss,shape,spatial,norm,all}.hpp`
- Create: `src/autograd/rules/{elementwise,linalg,activation,loss,shape,spatial,norm}.cpp`
- Create: `src/autograd/differentiator.cpp`
- Delete: `src/autograd/backward.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `tests/functional/unit/test_rule_table.cpp` (create), registered in `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 8's `rule.hpp`.
- Produces: `Autograd::RULES` in `rules/all.hpp`; per-domain `inline constexpr std::array<RuleEntry, N> <DOMAIN>_RULES`. Tasks 10–15 each add to exactly one domain array and one `.cpp`.

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_rule_table.cpp`:

```cpp
namespace SushiAI
{
    namespace Test
    {
        TEST(Unit_RuleTable, EveryDomainsRulesReachTheCombinedTable)
        {
            // all.hpp concatenates; a domain left out of the concat compiles
            // fine and simply never differentiates. The count is the cheapest
            // thing that notices.
            constexpr std::size_t domains =
                Autograd::ELEMENTWISE_RULES.size() + Autograd::LINALG_RULES.size()
                + Autograd::ACTIVATION_RULES.size() + Autograd::LOSS_RULES.size()
                + Autograd::SHAPE_RULES.size() + Autograd::SPATIAL_RULES.size()
                + Autograd::NORM_RULES.size();

            EXPECT_EQ(Autograd::RULES.size(), domains);
        }

        TEST(Unit_RuleTable, NoOperationHasTwoRules)
        {
            // Two entries for one identity is not a compile error and the
            // first one silently wins - so which gradient runs would depend on
            // the order two domains happened to be concatenated in.
            for (std::size_t i = 0; i < Autograd::RULES.size(); ++i)
                for (std::size_t j = i + 1; j < Autograd::RULES.size(); ++j)
                    EXPECT_NE(Autograd::RULES[i].op, Autograd::RULES[j].op)
                        << "'" << Ops::name_of(Autograd::RULES[i].op)
                        << "' has two backward rules";
        }

        TEST(Unit_RuleTable, TheRulesThatExistedBeforeTheSplitStillDo)
        {
            // The split is a move, not a rewrite. These fourteen were the
            // table before it; every one must still resolve.
            for (Ops::OpID op : {Ops::MATMUL.id, Ops::BIAS_ADD.id, Ops::ADD.id,
                                 Ops::SUB.id, Ops::MUL.id, Ops::SCALE.id, Ops::EXP.id,
                                 Ops::RELU.id, Ops::GELU.id, Ops::SIGMOID.id,
                                 Ops::TANH.id, Ops::SUM_ROWS.id, Ops::ACCUMULATE.id,
                                 Ops::NLL_LOSS.id})
                EXPECT_TRUE(Autograd::has_backward_rule(op))
                    << "'" << Ops::name_of(op) << "' lost its rule in the split";
        }
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_RuleTable"`
Expected: FAIL to compile.

- [ ] **Step 3: Split the rules by domain**

Move each existing rule from `src/autograd/backward.cpp` into its domain's `.cpp`, **unchanged**, and declare it in that domain's header:

| Domain | Rules moved from `backward.cpp` |
|---|---|
| `linalg` | `matmul_backward` (:103), `bias_add_backward` (:139), `sum_rows_backward` (:239) |
| `elementwise` | `add_backward` (:151), `sub_backward` (:160), `mul_backward` (:169), `scale_backward` (:182), `exp_backward` (:191), `accumulate_backward` (:255) |
| `activation` | `activation_backward` (:215) |
| `loss` | `log_softmax_backward` (:264), `nll_loss_backward` (:289) |
| `shape` | *(empty until Task 10)* |
| `spatial` | *(empty until Tasks 11–13)* |
| `norm` | *(empty until Task 14)* |

Each header follows this shape — `elementwise.hpp` shown in full, the rest identical in structure:

```cpp
namespace SushiAI
{
    namespace Autograd
    {
        /** @brief d(a+b) = [dy, dy]. */
        void add_backward(const GradientTape& tape);
        /** @brief d(a-b) = [dy, -dy]. */
        void sub_backward(const GradientTape& tape);
        /** @brief d(a*b) = [dy*b, dy*a]. */
        void mul_backward(const GradientTape& tape);
        /** @brief d(alpha*x) = alpha*dy. */
        void scale_backward(const GradientTape& tape);
        /** @brief d(exp x) = dy * exp(x), reading the forward output. */
        void exp_backward(const GradientTape& tape);
        /** @brief An in-graph accumulation passes its gradient to both operands. */
        void accumulate_backward(const GradientTape& tape);

        /** @brief Elementwise operations' backward rules. */
        inline constexpr std::array<RuleEntry, 6> ELEMENTWISE_RULES = {{
            {Ops::ADD.id,        &add_backward},
            {Ops::SUB.id,        &sub_backward},
            {Ops::MUL.id,        &mul_backward},
            {Ops::SCALE.id,      &scale_backward},
            {Ops::EXP.id,        &exp_backward},
            {Ops::ACCUMULATE.id, &accumulate_backward},
        }};
    }
}
```

The rules stop being `static`/anonymous-namespace functions and gain external linkage, which is what lets a header take their address.

- [ ] **Step 4: Concatenate them, and prove CV-9**

Create `include/SushiAI/autograd/rules/all.hpp`:

```cpp
namespace SushiAI
{
    namespace Autograd
    {
        /**
         * @brief Every backward rule, as one flat compile-time table.
         *
         * Adding an operation to a domain changes only that domain's file;
         * adding a domain is one line here.
         */
        inline constexpr auto RULES = concat_rules(
            concat_rules(
                concat_rules(ELEMENTWISE_RULES, LINALG_RULES),
                concat_rules(ACTIVATION_RULES, LOSS_RULES)),
            concat_rules(
                concat_rules(SHAPE_RULES, SPATIAL_RULES),
                NORM_RULES));

        /** @brief Whether @p op has a rule in @ref RULES. */
        constexpr bool has_rule(OpID op) noexcept
        {
            for (const RuleEntry& e : RULES)
                if (e.op == op) return true;
            return false;
        }

        /**
         * @brief CV-9: every operation declaring itself differentiable has a rule.
         *
         * The flag lives in @c core/ops.hpp, where the spec puts it; the proof
         * lives here, because here is the only place both the flag and the rule
         * table are visible. Splitting them that way is not a compromise - it
         * is what keeps @c core/ from depending on @c autograd/.
         *
         * @param table The identities to check.
         * @return True when every differentiable entry has a rule.
         */
        constexpr bool every_differentiable_op_has_a_rule(
            SushiRuntime::span<const Ops::OpName> table) noexcept
        {
            for (const Ops::OpName& entry : table)
                if (entry.differentiable && !has_rule(entry.id)) return false;
            return true;
        }

        static_assert(every_differentiable_op_has_a_rule(
                          SushiRuntime::span<const Ops::OpName>(Ops::ALL.data(),
                                                                Ops::ALL.size())),
                      "An operation is declared differentiable in core/ops.hpp but has no "
                      "backward rule. Declaring an identity and forgetting its rule is "
                      "otherwise silent: the graph builds, differentiate() skips the node, "
                      "and the model trains with a missing gradient.");
    }
}
```

This needs `OpName::differentiable`. Add it to `core/ops.hpp`'s `struct OpName` with a default of `false`, and a `constexpr OpName differentiable_op(std::string_view)` factory beside `op()`:

```cpp
            /**
             * @brief Whether a backward rule must exist for this operation.
             *
             * Proved by @c autograd/rules/all.hpp, which is where the rule
             * table is visible; this file only declares the claim.
             */
            bool differentiable = false;
```

Then mark the forward operations that need gradients — `MATMUL`, `BIAS_ADD`, `ADD`, `SUB`, `MUL`, `SCALE`, `EXP`, `SUM_ROWS`, `ACCUMULATE`, `NLL_LOSS`, the five activations, and from this plan `IM2COL`, `COL2IM`, `MAXPOOL2D`, `AVGPOOL2D`, `SPATIAL_MEAN`, `UPSAMPLE2D`, `CONCAT_CHANNELS`, `SLICE_CHANNELS`, `CHANNEL_AFFINE`, `BATCHNORM2D`, `RESHAPE`, `LAYOUT_CONVERT` — by constructing them with `differentiable_op(...)`.

**Mark each one only in the task that writes its rule.** Marking all of them now makes the `static_assert` fire until Task 15, which would leave the tree unbuildable across six tasks. In this task, mark only the fourteen that already have rules.

- [ ] **Step 5: Reduce `backward.cpp` to a traversal**

Move what remains — `Differentiator`, `differentiate()`, `has_backward_rule()` — into `src/autograd/differentiator.cpp`. `has_backward_rule` becomes:

```cpp
        bool has_backward_rule(OpID op) noexcept
        {
            return find_rule(op) != nullptr;
        }
```

with `find_rule` now walking `RULES` from `rules/all.hpp` and keeping the existing `LOG_SOFTMAX_RULE` special case, whose comment ("the extra entry that only exists to produce a good error") moves with it.

Delete `src/autograd/backward.cpp`. In `src/CMakeLists.txt`, replace its entry with:

```cmake
    autograd/differentiator.cpp
    autograd/tape.cpp
    autograd/rules/elementwise.cpp
    autograd/rules/linalg.cpp
    autograd/rules/activation.cpp
    autograd/rules/loss.cpp
    autograd/rules/shape.cpp
    autograd/rules/spatial.cpp
    autograd/rules/norm.cpp
    autograd/gradcheck.cpp
```

- [ ] **Step 6: Prove the move changed nothing**

Run: `sa test --suite all`
Expected: PASS, with the **same test count** as before Task 8. This is the whole gate for Tasks 8–9: a restructuring that changes a number has changed behaviour, and the number is the only thing that would say so.

---

# Task 10: `shape` rules — `RESHAPE` and `LAYOUT_CONVERT`

The two easiest gradients in the plan, done first so that the pattern every later rule task follows is established on operations whose mathematics cannot be got wrong.

**Files:**
- Modify: `include/SushiAI/autograd/rules/shape.hpp`, `src/autograd/rules/shape.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 7, 9.
- Produces: `SHAPE_RULES` sized 2.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/integration/test_autograd_numeric.cpp`:

```cpp
TEST_F(Integration_Autograd, ReshapeMovesItsGradientBackUnchanged)
{
    // A reshape moves no data, so its gradient is its incoming gradient in the
    // original shape. Worth a numeric check anyway: "reshape back to the input
    // shape" and "reshape back to the output shape" are both one line and only
    // one is right.
    Probe probe(engine(), Shape({3, 8}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.reshape(b.reshape(x, Shape({4, 6})), Shape({3, 8})));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, LayoutConvertsGradientIsTheConverseConversion)
{
    // The adjoint of a permutation is the inverse permutation. Getting it
    // wrong gives a gradient of exactly the right magnitude in the wrong
    // place - which no tolerance-based check on the loss would notice.
    Probe probe(engine(), Shape({2, 3, 4, 5}), [](const GraphBuilder& b, ValueId x)
    {
        const ValueId nchw = b.layout_convert(x, Graph::TensorLayout::NCHW);
        const ValueId back = b.layout_convert(nchw, Graph::TensorLayout::NHWC);
        return b.log_softmax(b.spatial_mean(back));
    });

    probe.expect_analytic_matches_numeric();
}
```

Match `Probe`'s real constructor and assertion spellings by reading the top of `test_autograd_numeric.cpp` first; the names above follow its documented shape, and where they differ the file wins.

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.Reshape|Integration_Autograd.LayoutConvert"`
Expected: FAIL — `differentiate()` reaches a `RESHAPE` node with no rule.

- [ ] **Step 3: Write the rules**

`src/autograd/rules/shape.cpp`:

```cpp
        void reshape_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().reshape(dy, tape.graph().values()[x].shape));
        }

        void layout_convert_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The adjoint of a permutation is the inverse permutation, which
            // here is simply the conversion back to the operand's own layout.
            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().layout_convert(
                dy, tape.graph().values()[x].layout));
        }
```

`shape.hpp` declares both and grows its table:

```cpp
        /** @brief Reshapes the incoming gradient back to the operand's shape. */
        void reshape_backward(const GradientTape& tape);
        /** @brief Converts the incoming gradient back to the operand's layout. */
        void layout_convert_backward(const GradientTape& tape);

        /** @brief Structural operations' backward rules. */
        inline constexpr std::array<RuleEntry, 2> SHAPE_RULES = {{
            {Ops::RESHAPE.id,        &reshape_backward},
            {Ops::LAYOUT_CONVERT.id, &layout_convert_backward},
        }};
```

- [ ] **Step 4: Mark them differentiable**

In `core/ops.hpp`, change `RESHAPE` and `LAYOUT_CONVERT` to use `differentiable_op(...)`. CV-9's `static_assert` now covers them.

- [ ] **Step 5: Run the tests**

Run: `sa test --suite integration --filter "Integration_Autograd"`
Expected: PASS, including every case that existed before.

---

# Task 11: `spatial` rules — `IM2COL` and `COL2IM`

Convolution's **only** new rule (spec §6). `matmul_backward` already produces the weight gradient; no `conv2d_backward` is written.

**Files:**
- Modify: `include/SushiAI/autograd/rules/spatial.hpp`, `src/autograd/rules/spatial.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 5, 9, 10.
- Produces: `SPATIAL_RULES` sized 2 after this task; Tasks 12–13 extend it.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(Integration_Autograd, AWholeConvolutionDifferentiatesThroughIm2colAndMatmul)
{
    // CV-1's claim, checked numerically: with no CONV2D identity, a
    // convolution's gradient is im2col_backward composed with the matmul
    // rule that was already there. If this passes, convolution needs no
    // rule of its own - which is the entire reason there isn't one.
    Probe probe(engine(), Shape({2, 5, 5, 3}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = b.im2col(x, g);                 // [2*25, 27]
        const ValueId weights = b.parameter({27, 8}, "w");   // 8 output channels
        const ValueId out = b.matmul(cols, weights);         // [2*25, 8]
        return b.log_softmax(b.spatial_mean(b.reshape(out, Shape({2, 5, 5, 8}))));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, Im2colsGradientAccumulatesEveryWindowThatReadAPixel)
{
    // A stride-1 padded 3x3 has every interior pixel read by nine windows, so
    // its gradient is a sum of nine terms. A col2im that overwrote instead of
    // accumulating would give exactly one of them - a gradient that is the
    // right shape, the right sign, and roughly a ninth of the right size.
    Probe probe(engine(), Shape({1, 4, 4, 2}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.pad_top = 1;  g.pad_left = 1;

        const ValueId cols = b.im2col(x, g);
        const ValueId weights = b.parameter({18, 4}, "w");
        return b.log_softmax(b.spatial_mean(
            b.reshape(b.matmul(cols, weights), Shape({1, 4, 4, 4}))));
    });

    probe.expect_analytic_matches_numeric();
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.AWholeConvolution|Integration_Autograd.Im2cols"`
Expected: FAIL — no rule for `IM2COL`.

- [ ] **Step 3: Write the rules**

`src/autograd/rules/spatial.cpp`:

```cpp
        namespace
        {
            /** @brief Reads CV-5's eight slots back out of a node's attributes. */
            Graph::ConvGeometry geometry_of(const Graph::Node& node) noexcept
            {
                Graph::ConvGeometry g;
                g.kernel_h   = static_cast<int64_t>(node.attributes.get(0));
                g.kernel_w   = static_cast<int64_t>(node.attributes.get(1));
                g.stride_h   = static_cast<int64_t>(node.attributes.get(2));
                g.stride_w   = static_cast<int64_t>(node.attributes.get(3));
                g.dilation_h = static_cast<int64_t>(node.attributes.get(4));
                g.dilation_w = static_cast<int64_t>(node.attributes.get(5));
                g.pad_top    = static_cast<int64_t>(node.attributes.get(6));
                g.pad_left   = static_cast<int64_t>(node.attributes.get(7));
                return g;
            }
        }

        void im2col_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // col2im *is* im2col's adjoint: a pixel read by several windows
            // gets the sum of their gradients, which is what makes this one
            // call rather than a scatter loop. SushiBLAS writes it as a
            // gather, so the sum's order is fixed and the result is
            // bit-reproducible (CV-6).
            const Graph::ValueId x = tape.node().inputs[0];
            tape.contribute_input(0, tape.builder().col2im(
                dy, tape.graph().values()[x].shape, geometry_of(tape.node())));
        }

        void col2im_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The adjoint of the adjoint. The extents come from the column
            // matrix this node consumed rather than from the geometry, because
            // that operand may have been produced by an asymmetrically padded
            // im2col whose extents the geometry cannot reproduce.
            const Graph::Node& node = tape.node();
            const Graph::ConvGeometry g = geometry_of(node);
            const Graph::Shape image = tape.graph().values()[node.outputs[0]].shape;
            const int64_t windows = tape.graph().values()[node.inputs[0]].shape[0];
            const int64_t per_image = windows / image[0];
            const int64_t out_w = Graph::output_extent(image[2], g.pad_left, g.pad_left,
                                                       g.dilation_w, g.kernel_w, g.stride_w);

            tape.contribute_input(0, tape.builder().im2col_with_extents(
                dy, g, per_image / out_w, out_w));
        }
```

Declare both in `spatial.hpp` and give it a two-entry `SPATIAL_RULES`.

- [ ] **Step 4: Mark them differentiable and run**

Change `IM2COL` and `COL2IM` to `differentiable_op(...)`.

Run: `sa test --suite integration --filter "Integration_Autograd"`
Expected: PASS.

---

# Task 12: `spatial` rules — pooling, upsample, spatial mean

Spec §6 is explicit: `maxpool2d_backward` and `avgpool2d_backward` are **separate functions**, not one `pool_backward` behind a mode flag. Their mathematics shares nothing — one routes through a recorded index, the other divides by a constant.

**Files:**
- Modify: `include/SushiAI/autograd/rules/spatial.hpp`, `src/autograd/rules/spatial.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 6, 7, 11.
- Produces: `SPATIAL_RULES` sized 5.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(Integration_Autograd, MaxPoolRoutesEachGradientToTheElementThatWon)
{
    // Sampled with push_off_kinks so no two elements in a window tie: at a tie
    // the max is not differentiable and a numeric check would be comparing
    // against a one-sided derivative.
    Probe probe(engine(), Shape({2, 4, 4, 3}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        g.stride_h = 2; g.stride_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.expect_analytic_matches_numeric_off_kinks();
}

TEST_F(Integration_Autograd, OverlappingMaxPoolWindowsAccumulateOnADominantPixel)
{
    // Stride 1: a pixel that wins several windows receives all their
    // gradients. The separate-functions decision earns itself here - a shared
    // pool_backward would have had to branch on mode inside this loop.
    Probe probe(engine(), Shape({1, 5, 5, 2}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 2; g.kernel_w = 2;
        return b.log_softmax(b.spatial_mean(b.maxpool2d(x, g).output));
    });

    probe.expect_analytic_matches_numeric_off_kinks();
}

TEST_F(Integration_Autograd, AvgPoolSpreadsItsGradientOverTheWholeWindow)
{
    Probe probe(engine(), Shape({2, 6, 6, 3}), [](const GraphBuilder& b, ValueId x)
    {
        Graph::ConvGeometry g;
        g.kernel_h = 3; g.kernel_w = 3;
        g.stride_h = 3; g.stride_w = 3;
        return b.log_softmax(b.spatial_mean(b.avgpool2d(x, g)));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, UpsampleSumsEachSourcePixelsReplicas)
{
    // Scale 2 replicates every pixel four times, so its gradient is the sum of
    // four. Forgetting the sum gives a gradient a quarter of the right size,
    // which trains - just slower, and only in this one layer.
    Probe probe(engine(), Shape({2, 3, 3, 4}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(b.upsample2d(x, 2, 2)));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, SpatialMeanBroadcastsScaledByOneOverHW)
{
    // The rule whose absence was a real gap in an earlier draft of the design.
    // It is a per-(n,c) broadcast: BIAS_ADD's is per-row and CHANNEL_AFFINE's
    // is per-channel, and neither would have produced this.
    Probe probe(engine(), Shape({2, 4, 4, 5}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(x));
    });

    probe.expect_analytic_matches_numeric();
}
```

If `Probe` has no off-kinks variant, add one in this task using `Autograd::push_off_kinks` (`include/SushiAI/autograd/gradcheck.hpp:188`), which exists for exactly this and is already used by `Unit_GradCheck.AReluFamilyFunctionNeedsToBeSampledOffTheKink`.

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.MaxPool|Integration_Autograd.AvgPool|Integration_Autograd.Upsample|Integration_Autograd.SpatialMean"`
Expected: FAIL — no rules.

- [ ] **Step 3: Write the five rules**

```cpp
        void maxpool2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The indices are this node's second *result*, so the rule reads
            // exactly what its own forward wrote - there is no way to pair a
            // gradient with another pool's argmax.
            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().maxpool2d_backward(
                dy, node.outputs[1], tape.graph().values()[node.inputs[0]].shape,
                geometry_of(node)));
        }

        void avgpool2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // No index and no saved forward: every element of a window gets
            // the same share, so the divisor is the window area alone. This is
            // why it is a separate function from the max rule rather than the
            // same one with a flag.
            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().avgpool2d_backward(
                dy, tape.graph().values()[node.inputs[0]].shape, geometry_of(node)));
        }

        void upsample2d_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().upsample2d_backward(
                dy, tape.graph().values()[node.inputs[0]].shape,
                static_cast<int64_t>(node.attributes.get(0)),
                static_cast<int64_t>(node.attributes.get(1))));
        }

        void spatial_mean_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();
            tape.contribute_input(0, tape.builder().spatial_mean_backward(
                dy, tape.graph().values()[node.inputs[0]].shape));
        }
```

Each names a `GraphBuilder` emitter that Task 6 did not write, because a backward-only emitter belongs with its rule. Add `maxpool2d_backward`, `avgpool2d_backward`, `upsample2d_backward` and `spatial_mean_backward` to `builder.hpp`/`builder.cpp` in this task, each emitting its `Ops::*_BACKWARD` identity with the geometry attributes and the stated output shape. `spatial_mean_backward`'s divisor is `1/(H*W)` taken from the shape it is given, never passed in — one source for it, so the forward and backward cannot disagree.

- [ ] **Step 4: Mark differentiable and run**

`MAXPOOL2D`, `AVGPOOL2D`, `UPSAMPLE2D`, `SPATIAL_MEAN` become `differentiable_op(...)`.

Run: `sa test --suite integration --filter "Integration_Autograd"`
Expected: PASS.

---

# Task 13: `spatial` rules — `CONCAT_CHANNELS` and `SLICE_CHANNELS`

**Files:**
- Modify: `include/SushiAI/autograd/rules/spatial.hpp`, `src/autograd/rules/spatial.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 7, 12.
- Produces: `SPATIAL_RULES` sized 7.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(Integration_Autograd, ConcatGivesEachOperandItsOwnChannelRange)
{
    // The gradient of a concat is a slice per operand, at that operand's
    // offset. An off-by-one offset gives every operand a gradient of the
    // right shape belonging to its neighbour.
    Probe probe(engine(), Shape({2, 4, 4, 6}), [](const GraphBuilder& b, ValueId x)
    {
        const ValueId left = b.slice_channels(x, 0, 2);
        const ValueId right = b.slice_channels(x, 2, 4);
        const std::array<ValueId, 2> parts = {right, left};   // deliberately reordered
        return b.log_softmax(b.spatial_mean(b.concat_channels(
            SushiRuntime::span<const ValueId>(parts.data(), parts.size()))));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, SliceScattersItsGradientBackIntoZeros)
{
    // Channels outside the slice receive exactly zero, not stale memory. The
    // reordered concat above is what makes a wrong offset visible; this one
    // makes a missing zero-fill visible.
    Probe probe(engine(), Shape({1, 3, 3, 8}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.spatial_mean(b.slice_channels(x, 3, 2)));
    });

    probe.expect_analytic_matches_numeric();
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.Concat|Integration_Autograd.Slice"`
Expected: FAIL.

- [ ] **Step 3: Write the rules**

```cpp
        void concat_channels_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // One slice per operand, at that operand's running offset. Walked
            // in operand order rather than by looking anything up, because the
            // offset is defined by the order the forward concatenated in.
            const Graph::Node& node = tape.node();
            int64_t offset = 0;
            for (std::size_t i = 0; i < node.inputs.size(); ++i)
            {
                const int64_t channels = tape.graph().values()[node.inputs[i]].shape[3];
                if (tape.input_requires_grad(i))
                    tape.contribute_input(i, tape.builder().slice_channels(dy, offset, channels));
                offset += channels;
            }
        }

        void slice_channels_backward(const GradientTape& tape)
        {
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // The adjoint of a slice is a scatter into zeros. Expressed as a
            // concat of three parts rather than as a new identity: the ranges
            // before and after the slice are zeros, and CONCAT_CHANNELS
            // already places a run of channels at an offset.
            const Graph::Node& node = tape.node();
            const Graph::Shape in = tape.graph().values()[node.inputs[0]].shape;
            const int64_t offset = static_cast<int64_t>(node.attributes.get(0));
            const int64_t taken = tape.graph().values()[node.outputs[0]].shape[3];

            std::vector<Graph::ValueId> parts;
            if (offset > 0)
                parts.push_back(tape.builder().zeros(
                    Graph::Shape({in[0], in[1], in[2], offset}),
                    tape.graph().values()[dy].dtype));
            parts.push_back(dy);
            if (offset + taken < in[3])
                parts.push_back(tape.builder().zeros(
                    Graph::Shape({in[0], in[1], in[2], in[3] - offset - taken}),
                    tape.graph().values()[dy].dtype));

            tape.contribute_input(0, parts.size() == 1
                ? dy
                : tape.builder().concat_channels(
                      SushiRuntime::span<const Graph::ValueId>(parts.data(), parts.size())));
        }
```

- [ ] **Step 4: Mark differentiable and run**

Run: `sa test --suite integration --filter "Integration_Autograd"`
Expected: PASS.

---

# Task 14: `norm` rules — `BATCHNORM2D` and `CHANNEL_AFFINE`

`batchnorm2d_backward` is its own rule and is added to nothing (spec §6). CV-7 keeps the running statistics out of the graph entirely — this task does **not** implement `BATCHNORM_UPDATE_STATS`, which is Plan C's step 11.

**Files:**
- Modify: `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`, `include/SushiAI/autograd/rules/norm.hpp`, `src/autograd/rules/norm.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 7, 9.
- Produces:
  ```cpp
  struct BatchNormResult { ValueId output; ValueId saved_mean; ValueId saved_inv_std; };
  BatchNormResult batchnorm2d(ValueId x, ValueId gamma, ValueId beta, double epsilon) const;
  ```
  `NORM_RULES` sized 2. Plan C's `BatchNorm2d` layer calls the emitter and reads `saved_mean`/`saved_inv_std` for `BATCHNORM_UPDATE_STATS`.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(Integration_Autograd, BatchNormsGradientAccountsForTheStatisticsItComputed)
{
    // The rule that is easy to get subtly wrong and hard to notice: the batch
    // mean and variance are functions of x, so dx has three terms, not one.
    // Treating them as constants gives a gradient that is right on average
    // over the batch and wrong per element - the model still trains, slightly
    // worse, with nothing to point at.
    Probe probe(engine(), Shape({4, 3, 3, 5}), [](const GraphBuilder& b, ValueId x)
    {
        const ValueId gamma = b.parameter({5}, "gamma");
        const ValueId beta = b.parameter({5}, "beta");
        return b.log_softmax(b.spatial_mean(b.batchnorm2d(x, gamma, beta, 1e-5).output));
    });

    probe.expect_analytic_matches_numeric();
}

TEST_F(Integration_Autograd, ChannelAffineDifferentiatesAtItsInputAndBothParameters)
{
    Probe probe(engine(), Shape({2, 4, 4, 6}), [](const GraphBuilder& b, ValueId x)
    {
        const ValueId gamma = b.parameter({6}, "gamma");
        const ValueId beta = b.parameter({6}, "beta");
        return b.log_softmax(b.spatial_mean(b.channel_affine(x, gamma, beta)));
    });

    probe.expect_analytic_matches_numeric();
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.BatchNorm|Integration_Autograd.ChannelAffine"`
Expected: FAIL to compile — `batchnorm2d` has no emitter.

- [ ] **Step 3: Emit the forward, with its saved statistics**

```cpp
        BatchNormResult GraphBuilder::batchnorm2d(ValueId x, ValueId gamma, ValueId beta,
                                                  double epsilon) const
        {
            const Value& in = graph_.values()[x];
            SA_THROW_IF(in.shape.rank() != 4,
                        "batchnorm2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(epsilon <= 0.0,
                        "batchnorm2d's epsilon must be positive; got {}", epsilon);

            for (ValueId p : {gamma, beta})
            {
                const Value& v = graph_.values()[p];
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "batchnorm2d's per-channel parameters must be [{}]",
                            in.shape[3]);
            }

            const Shape per_channel({in.shape[3]});
            const ValueId y = declare(in.shape, in.dtype, Lifetime::ACTIVATION, "bn");
            // Saved rather than recomputed: the backward needs exactly the
            // statistics the forward used, and recomputing them from a value
            // some later node may have overwritten is how a normalisation
            // silently drifts.
            const ValueId mean = declare(per_channel, in.dtype, Lifetime::ACTIVATION,
                                         "bn_saved_mean");
            const ValueId inv_std = declare(per_channel, in.dtype, Lifetime::ACTIVATION,
                                            "bn_saved_inv_std");
            graph_.value(y).layout = in.layout;

            Attributes a;
            a.values[0] = epsilon;
            emit(Ops::BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std}, a);
            return BatchNormResult{y, mean, inv_std};
        }
```

- [ ] **Step 4: Write the rules**

```cpp
        void batchnorm2d_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            // One node producing all three gradients, because all three share
            // the two reductions over the batch: splitting them would compute
            // the same sums three times, and adding it to activation_backward
            // (which is [dy, primal] -> [dx]) would break that rule's shape.
            const Graph::Node& node = tape.node();
            const Graph::BatchNormGrads g = tape.builder().batchnorm2d_backward(
                dy, node.inputs[0], node.inputs[1], node.outputs[1], node.outputs[2]);

            if (tape.input_requires_grad(0)) tape.contribute_input(0, g.dx);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, g.dgamma);
            if (tape.input_requires_grad(2)) tape.contribute_input(2, g.dbeta);
        }

        void channel_affine_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::NO_VALUE) return;

            const Graph::Node& node = tape.node();

            // dx = dy * gamma, broadcast per channel - the same shape of
            // operation as the forward, so it is the forward's own emitter
            // with beta set to zero rather than a new identity.
            if (tape.input_requires_grad(0))
                tape.contribute_input(0, tape.builder().channel_affine(
                    dy, node.inputs[1],
                    tape.builder().zeros(tape.graph().values()[node.inputs[2]].shape,
                                         tape.graph().values()[dy].dtype)));

            // dgamma and dbeta are per-channel sums over N, H and W.
            if (tape.input_requires_grad(1))
                tape.contribute_input(1, tape.builder().sum_channels(
                    tape.builder().mul(dy, node.inputs[0])));
            if (tape.input_requires_grad(2))
                tape.contribute_input(2, tape.builder().sum_channels(dy));
        }
```

`batchnorm2d_backward`'s emitter, `BatchNormGrads`, and `sum_channels` are written in this task. `sum_channels` reduces `[N,H,W,C]` to `[C]`; it is `Ops::SPATIAL_MEAN`'s sibling and lowers to SushiBLAS's segmented axis reduction with `segments2` — the second segment axis Plan A added for exactly this pair of operations.

- [ ] **Step 5: Mark differentiable and run**

`BATCHNORM2D` and `CHANNEL_AFFINE` become `differentiable_op(...)`.

Run: `sa test --suite integration --filter "Integration_Autograd"`
Expected: PASS.

---

# Task 15: `LEAKY_RELU`'s rule

The last rule, and the smallest: `activation_backward` already handles it, because spec §6 calls that sharing *parameterisation, not conflation* — five activations with identical structure differing only in the identity looked up in `ACTIVATIONS`.

**Files:**
- Modify: `include/SushiAI/autograd/rules/activation.hpp`, `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`, `include/SushiAI/core/ops.hpp`
- Test: `tests/functional/integration/test_autograd_numeric.cpp`

**Interfaces:**
- Consumes: Tasks 4, 9.
- Produces: `ACTIVATION_RULES` sized 5; `GraphBuilder::leaky_relu(ValueId, double)`.

- [ ] **Step 1: Write the failing test**

```cpp
TEST_F(Integration_Autograd, LeakyReluHasBothOfItsSlopes)
{
    // Sampled off the kink, and deliberately over a range wide enough that
    // some elements land on each side: a probe that only ever sees positive
    // inputs would pass with the negative slope hard-coded to zero, i.e.
    // with a plain ReLU.
    Probe probe(engine(), Shape({4, 8}), [](const GraphBuilder& b, ValueId x)
    {
        return b.log_softmax(b.leaky_relu(x, 0.1));
    });

    probe.expect_analytic_matches_numeric_off_kinks();
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite integration --filter "Integration_Autograd.LeakyRelu"`
Expected: FAIL to compile — no `leaky_relu` emitter.

- [ ] **Step 3: Emit it, and add the one table row**

```cpp
        ValueId GraphBuilder::leaky_relu(ValueId x, double slope) const
        {
            SA_THROW_IF(slope <= 0.0 || slope >= 1.0,
                        "leaky_relu's slope must be in (0, 1); got {}. At 0 it is a ReLU "
                        "and at 1 it is the identity, and both would make the "
                        "output-reading gradient (ops.hpp's ACTIVATIONS) unsound.",
                        slope);

            const Value& in = graph_.values()[x];
            const ValueId out = declare(in.shape, in.dtype, Lifetime::ACTIVATION, "leaky_relu");
            graph_.value(out).layout = in.layout;

            Attributes a;
            a.values[0] = slope;
            emit(Ops::LEAKY_RELU.id, {x}, {out}, a);
            return out;
        }
```

In `activation.hpp`, add one row — no new function:

```cpp
        inline constexpr std::array<RuleEntry, 5> ACTIVATION_RULES = {{
            {Ops::RELU.id,       &activation_backward},
            {Ops::GELU.id,       &activation_backward},
            {Ops::SIGMOID.id,    &activation_backward},
            {Ops::TANH.id,       &activation_backward},
            {Ops::LEAKY_RELU.id, &activation_backward},
        }};
```

- [ ] **Step 4: Run everything**

Mark `LEAKY_RELU` as `differentiable_op(...)`.

Run: `sa test --suite all`
Expected: PASS. Every identity this plan marked differentiable now has a rule, so CV-9's `static_assert` is fully loaded for the first time — a clean build is that proof.

---

# Task 16: The `IM2COL` rematerialisation pass

CV-10. Deliberately **not** a general rematerialisation framework: a general pass would have one client, no cost model, and would be making a memory decision without memory information.

**Files:**
- Create: `include/SushiAI/graph/rematerialise.hpp`, `src/graph/rematerialise.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `tests/functional/unit/test_rematerialise.cpp` (create), registered in `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 5, 11.
- Produces: `void rematerialise_im2col(Graph::Graph& graph);`, run after `differentiate()` and before memory planning.

- [ ] **Step 1: Write the failing test**

```cpp
        TEST(Unit_Rematerialise, TheColumnBufferStopsBeingAnActivation)
        {
            // The number CV-10 is about: ResNet-18's first block at batch 128
            // holds 128*32*32*64*9*4 bytes = ~302 MB in this one buffer, and
            // there are twenty convolutions. Cloning the gather next to its
            // backward consumer drops the lifetime to SCRATCH, so one is live
            // at a time instead of twenty.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId w = builder.parameter({27, 8}, "w");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId cols = builder.im2col(x, g);
            const ValueId out = builder.matmul(cols, w);
            graph.finalise();

            const std::array<ValueId, 1> wrt = {w};
            (void)Autograd::differentiate(graph, out,
                SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

            EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::ACTIVATION);
            Graph::rematerialise_im2col(graph);
            EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::SCRATCH);
        }

        TEST(Unit_Rematerialise, TheCloneComputesTheSameThingAsTheOriginal)
        {
            // A clone that read a different operand or carried different
            // attributes would produce a plausible column matrix and a wrong
            // weight gradient. Both are checked because the geometry lives in
            // the attributes and is easy to drop when copying a node.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 4, 4, 2}, "x");
            const ValueId w = builder.parameter({18, 4}, "w");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId cols = builder.im2col(x, g);
            (void)builder.matmul(cols, w);
            graph.finalise();

            const std::array<ValueId, 1> wrt = {w};
            (void)Autograd::differentiate(graph, cols,
                SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

            const std::size_t before = graph.nodes().size();
            Graph::rematerialise_im2col(graph);

            EXPECT_GT(graph.nodes().size(), before) << "no clone was inserted";

            const Graph::Node& original = graph.nodes()[graph.values()[cols].producer];
            std::size_t clones = 0;
            for (const Graph::Node& n : graph.nodes())
            {
                if (n.op != Ops::IM2COL.id || &n == &original) continue;
                ++clones;
                EXPECT_EQ(n.inputs, original.inputs);
                for (std::size_t i = 0; i < Graph::Attributes::CAPACITY; ++i)
                    EXPECT_EQ(n.attributes.get(i), original.attributes.get(i))
                        << "the clone lost attribute slot " << i;
            }
            EXPECT_EQ(clones, std::size_t{1});
        }

        TEST(Unit_Rematerialise, AnIm2colWithNoBackwardConsumerIsLeftAlone)
        {
            // Inference: nothing was differentiated, so there is no second
            // consumer and cloning would be a free slowdown.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 4, 4, 2}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;

            const ValueId cols = builder.im2col(x, g);
            graph.finalise();

            const std::size_t before = graph.nodes().size();
            Graph::rematerialise_im2col(graph);

            EXPECT_EQ(graph.nodes().size(), before);
            EXPECT_EQ(graph.values()[cols].lifetime, Graph::Lifetime::ACTIVATION);
        }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `sa test --suite unit --filter "Unit_Rematerialise"`
Expected: FAIL to compile.

- [ ] **Step 3: Write the pass**

`src/graph/rematerialise.cpp`:

```cpp
        void rematerialise_im2col(Graph& graph)
        {
            // Named for the one operation it serves, and deliberately not
            // generalised: a general pass would have exactly this client, no
            // cost model, and would be deciding a memory policy without any
            // memory information. IM2COL is exempt from needing that model
            // because its buffer is always enormous relative to the cost of
            // recomputing it - one extra gather beside the GEMM it feeds.
            const std::size_t original_count = graph.nodes().size();

            for (std::size_t i = 0; i < original_count; ++i)
            {
                if (graph.nodes()[i].op != Ops::IM2COL.id) continue;

                const ValueId cols = graph.nodes()[i].outputs[0];

                // Consumers appended after the original count are the backward
                // pass's; the forward's are the ones that were already there.
                std::vector<std::size_t> backward_consumers;
                for (std::size_t j = original_count; j < graph.nodes().size(); ++j)
                    for (ValueId in : graph.nodes()[j].inputs)
                        if (in == cols) { backward_consumers.push_back(j); break; }

                if (backward_consumers.empty()) continue;

                // One clone, shared by every backward consumer: they all run
                // adjacently, so a clone each would buy nothing and cost a
                // gather each.
                const Node source = graph.nodes()[i];
                const Value& original = graph.values()[cols];
                const ValueId clone = graph.declare_value(
                    original.shape, original.dtype, Lifetime::SCRATCH, "cols_remat");
                graph.value(clone).layout = original.layout;
                graph.append_node(Node{source.op, source.inputs, {clone}, source.attributes});

                for (std::size_t j : backward_consumers)
                    for (ValueId& in : graph.mutable_node(j).inputs)
                        if (in == cols) in = clone;

                // The original now dies at the end of the forward pass.
                graph.value(cols).lifetime = Lifetime::SCRATCH;
            }
        }
```

`declare_value`, `append_node` and `mutable_node` are the mutating entry points this pass needs. Add whichever of them `Graph` does not already have, beside the existing checked `value(ValueId)` accessor (see amendment A-1), each with Doxygen saying it exists for graph-to-graph passes. Each must guard its index the way `value()` does — a pass writing past the end of `nodes_` is the same defect A-1 removed, in the other container.

- [ ] **Step 4: Run the tests**

Run: `sa test --suite unit --filter "Unit_Rematerialise"`
Expected: PASS, all three.

- [ ] **Step 5: Prove the gradients are unchanged**

Run: `sa test --suite all`
Expected: PASS. `Integration_Autograd.AWholeConvolutionDifferentiatesThroughIm2colAndMatmul` is the case that matters: a rematerialisation that rewired a consumer wrongly would change that gradient, and the numeric check is what would say so.

---

# Task 17: The final gate

**Files:**
- Modify: `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`

- [ ] **Step 1: Document the autograd split**

In `docs/ARCHITECTURE.md`, update the autograd section: `backward.cpp` no longer exists, and the file map is `differentiator.cpp` + `tape.cpp` + `rules/<domain>.cpp`. State the rule that makes it worth having — adding an operation to a domain changes only that domain's file, adding a domain is one line in `all.hpp` — and record why nothing self-registers (static-library linker garbage collection would drop a registration-only translation unit silently, and only in a consumer's build).

- [ ] **Step 2: Document the invariants**

Add to the same document: `TensorLayout` and what `finalise()` proves (CV-4); the `differentiable` flag's split between `core/ops.hpp` (the claim) and `autograd/rules/all.hpp` (the proof), with the layering reason; and `rematerialise_im2col`'s deliberate lack of generality (CV-10).

- [ ] **Step 3: Write the CHANGELOG entries**

Under `[Unreleased]`: **Added** for the twenty identities, `TensorLayout`, the eight attribute slots and the rematerialisation pass; **Changed** for the autograd split and the `finalise()` rename, naming all three moved call sites.

- [ ] **Step 4: Clean build**

Run: `sa build --clean`
Expected: SUCCESS with no warnings. Every `static_assert` in this plan — `Attributes::CAPACITY` against `K_MAX_PARAMS`, `all_ids_unique` over the widened `ALL`, `every_fused_activation_survives` with `LEAKY_RELU` in it, and CV-9's `every_differentiable_op_has_a_rule` — is proved by this command and by nothing else.

- [ ] **Step 5: Full suite**

Run: `sa test --suite all`
Expected: PASS.

- [ ] **Step 6: The package gate**

Run: `sa test --suite package`

This plan adds installed headers (`autograd/tape.hpp`, `autograd/rule.hpp`, `autograd/rules/*.hpp`, `graph/rematerialise.hpp`). If `sa` has no `package` suite, port the one added to `sb` in this session (`cli/sushiblas/services/project.py`'s `test_package`, `Suite.package`, and `_toolchain_args`) — it is the same CLI shape and the same gap. Report it as a finding either way.

- [ ] **Step 7: Report honestly**

State which of the seven gates passed and which did not. A rule whose numeric check was skipped, a `static_assert` disabled to get a build through, or a test left filtered out is reported as not done — not folded into a summary that says the plan is complete.

---

## Self-review

**Spec coverage.** Build-order steps 5–9, each to a task:

| Spec step | Tasks |
|---|---|
| 5 — `CAPACITY` 4→8; `Value` carries a layout; `finalise()` proves uniformity | 1, 2 |
| 6 — identities, `differentiable` + its `static_assert`, shape inference | 3, 4, 5, 6, 7 |
| 7 — the autograd restructuring, **before** any new rule | 8, 9 |
| 8 — the new backward rules | 10, 11, 12, 13, 14, 15 |
| 9 — the `IM2COL` rematerialisation pass | 16 |

Every identity in spec §5 is declared in Task 3 and reaches `Ops::ALL`. `BATCHNORM_UPDATE_STATS` is declared there and deliberately given no rule and no `differentiable` flag — CV-7 keeps it out of the differentiated graph, and it is emitted by Plan C's step 11.

**Gaps found while reviewing, and closed:**

- Task 12's rules call four `GraphBuilder` emitters (`maxpool2d_backward`, `avgpool2d_backward`, `upsample2d_backward`, `spatial_mean_backward`) that Task 6 does not declare. Rather than move them forward into a shape-inference task, Task 12 writes them, because a backward-only emitter has no caller until its rule exists. Task 12 step 3 says so outright.
- Task 14 does the same for `batchnorm2d_backward`, `BatchNormGrads` and `sum_channels`.
- Task 2 cannot compile before Task 3, because `prove_layout_uniformity` names `Ops::LAYOUT_CONVERT`. Task 2 step 6 sends the implementer to Task 3 and back rather than leaving them to discover it.
- CV-9's `static_assert` cannot live in `core/ops.hpp` — that file cannot see the rule table without `core/` depending on `autograd/`. The flag stays where the spec puts it and the proof moves to `rules/all.hpp`; Task 9 step 4 states the reasoning.
- Marking every operation `differentiable` in Task 9 would leave the tree unbuildable from Task 9 to Task 15. Each task marks only what it just gave a rule to; Task 9 step 4 says so explicitly.

**Type consistency.** `ConvGeometry` is declared once (Task 5) and read back once (`geometry_of`, Task 11); its eight slots are written by `geometry_attributes` and read in the same order. `PoolResult` (Task 6) and `BatchNormResult` (Task 14) are the only multi-result returns, and both are named structs rather than `std::pair`, so a caller cannot swap the members. `output_extent` has one definition in SushiAI (Task 5) and matches SushiBLAS's `Spatial::output_extent` argument for argument.

**Names to verify against the tree before writing code**, called out at their point of use rather than assumed: `Ops::folds_without_spill` (Task 4), `GraphBuilder::declare` / `emit` (Task 5), `Shape::element_count` (Task 7), and `Probe`'s constructor and assertion methods (Task 10). Each step says the file wins if it disagrees with the plan.


