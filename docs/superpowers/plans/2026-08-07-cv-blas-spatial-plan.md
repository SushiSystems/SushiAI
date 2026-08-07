# Plan A — SushiBLAS: the `spatial` domain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give SushiBLAS the windowed-movement, channel-broadcast and segmented-reduction primitives a convolutional network needs, verified by SushiBLAS's own suite, so that when SushiAI's autograd is wrong later nobody has to ask whether the kernels are wrong too.

**Architecture:** A new engine domain `spatial`, built exactly like the five that exist: a public `SpatialOps` class reached through `Engine::spatial()`, one `.cpp` per operation, a reference kernel and a SYCL kernel per operation so `ExecutionMode::VERIFY` has a ground truth, and one host-side descriptor — `Spatial::WindowSpan` — that resolves layout, axis order and strides into element counts before the device sees anything. Plus two additions outside the new domain: a second segment axis on `Reduce::AxisSpan`, and a layout-parametric channel-bias GEMM epilogue.

**Tech Stack:** C++17, SYCL (intel-llvm), CMake ≥ 3.22, GoogleTest, the `sb` CLI.

**Repository:** `D:\Projects\sushiblas` — **not** `sushiai`. Every path in this plan is relative to the SushiBLAS checkout unless it says otherwise.

**Source spec:** `sushiai/docs/superpowers/specs/2026-08-07-cv-pipeline-layers-design.md`, build-order steps 1–4.

---

## Global Constraints

- **C++17 throughout.** No concepts, no `consteval`, no `std::span`, no `std::format`. Use `SushiRuntime::span`, `SushiBLAS::Core::format`, `constexpr bool` variable templates / `std::void_t` detectors, and `constexpr` where you want `consteval`. This is an ABI constraint (`SushiRuntime::span` switches implementation on `__cplusplus`), not a style preference.
- **All builds, tests and runs go through the `sb` CLI.** Never invoke ninja or cmake directly. `sb build`, `sb test --suite unit`, `sb run …`.
- **Every file starts with the SushiBLAS license header** (see `docs/CONTRIBUTING.md` §4 — the SushiBLAS variant, naming SushiBLAS and its URL, not SushiAI's).
- **Naming:** `PascalCase` types/namespaces, `snake_case` functions and variables, trailing underscore on private members, 4-space indent, Allman braces for multi-statement blocks, no braces on a single-statement `if`/loop.
- **Doxygen on every public API:** `@brief`, `@param` per parameter, `@return`.
- **Build-system names are prefixed `sushiblas_`.** Never bare `sushi_`.
- **The GTest suite-name prefix is the only thing that assigns a CTest label.** `Unit_*`, `Integration_*`, `Regression_*`. A suite named without one is invisible rather than failing.
- **Every new symbol ships with a test.**
- **One logical change per commit.** Commit header `type(scope): short summary`, lowercase, imperative, no trailing period. Body bullets start with a past-tense verb.
- **All strides are element counts and non-negative.** `Reduce::AxisSpan` documents this and the new `Spatial::WindowSpan` inherits the rule; a negative stride (a flip) is not expressible and must not be made expressible here.
- **Determinism is a hard contract.** SushiBLAS links `sushiruntime_deterministic_fp` PUBLIC. No floating-point atomic add, no `sycl::reduction`, no `sycl::reduce_over_group`, anywhere in this plan. Every backward is a gather (spec CV-6).
- **Activations are NHWC** (spec CV-3). `WindowSpan` names no layout — it carries stride numbers — but every caller in this plan builds it from an NHWC tensor.

---

## Recorded deviations from the spec

These are decisions this plan makes that the spec does not state. Each is here so it is reviewed rather than discovered in the code.

**D-A1 — Three operations the spec's SushiAI identity list needs and its SushiBLAS table omits.**
Spec §5 declares the identities `SPATIAL_MEAN`, `SPATIAL_MEAN_BACKWARD` and `LAYOUT_CONVERT`, and spec §4's operation table lists no SushiBLAS operation any of them could execute. §4 mentions global average pooling only as the *reason* `AxisSpan` gains a second segment axis, and names no entry point. This plan adds three:

| Added | Where | Serves |
|---|---|---|
| `ReductionOps::mean_spatial` | reductions domain | `SPATIAL_MEAN` |
| `SpatialOps::broadcast_spatial` | spatial domain | `SPATIAL_MEAN_BACKWARD` |
| `SpatialOps::layout_convert` | spatial domain | `LAYOUT_CONVERT` (spec CV-3's boundary transpose) |

`mean_spatial` goes in *reductions*, not *spatial*, because spec §4 is explicit: "Global average pooling is **not** here — it is an axis reduction, not a sliding window." Its backward is a broadcast rather than a reduction, so it goes the other way.

**D-A2 — `Spatial::ConvGeometry` and `Spatial::WindowSpan` live in a tensor-free header.**
Spec §4 gives one entry point, `window_span_for(const Tensor&, const ConvGeometry&, const Tensor&)`. This plan splits the declarations from that function: `engine/spatial/geometry.hpp` holds `ConvGeometry`, `WindowSpan`, `ColumnSpan` and `output_extent()` and includes nothing but `<cstdint>`; `engine/spatial/window.hpp` holds `window_span_for` and includes `tensor.hpp`. The reason is downstream: SushiAI's shape inference is host-lane (it reaches no SYCL and is tested in the device-free `Unit_` tier), and CV-5's `H_o` formula must be *the same formula* on both sides of the repo boundary rather than two copies that can drift. The split lets SushiAI include the arithmetic without inheriting the device stack.

**D-A3 — `Internal::execute_blas_task` is generalised rather than copied.**
Its diagnostics say "BLAS operation '{}'", which would be a lie for `spatial.im2col`. Rather than duplicate ~50 lines of dtype switch, this plan moves the body to `Internal::execute_typed_task` in a new `engine/task.hpp`, taking the family string as a parameter, and leaves `execute_blas_task` as a one-line forwarder with `"BLAS operation"`. No existing call site changes. `Internal::dispatch_with_mode` moves to the same header for the same reason — the spatial ops need the `FORCE_REFERENCE` / normal / `VERIFY` triad verbatim.

**D-A4 — `pool_avg` divides by the window area, not by the count of in-bounds elements.**
The spec does not say. This is PyTorch's `count_include_pad=True` default, and matching it is what makes spec §10's parity claim true for a network that average-pools with padding. Recorded because the alternative is defensible and the choice is invisible in any test that does not pad.

**D-A5 — `pool_max` breaks ties towards the first element in scan order** (`>`, not `>=`), and records the **logical** flat input index `((n·H + h)·W + w)·C + c`, not a strided byte or element offset. Logical, because the backward gather recomputes a candidate pixel's own index and compares; a strided offset would not be comparable across a view. First-in-scan-order, because it is deterministic and it is PyTorch's convention.

**D-A6 — `concat_channels` and `slice_channels` are two public methods over one kernel.**
Spec §4 says "One strided-copy kernel serves both", and this plan does that — but exposes both names, because a call site reading `slice_channels(dy, offset, operand_grad)` says what it means and `copy_channels(dy, offset, operand_grad, 0, count)` does not.

**D-A7 — `LEAKY_RELU` needs no work in this plan.** `NonLinearOps::leaky_relu` and `leaky_relu_backward` already exist (`include/SushiBLAS/engine/math/nonlinear.hpp:75,86`). Spec §5's argument that reading the output is a bit-identical substitution holds against that kernel's actual form. Noted so nobody adds a second one.

---

## File structure

**Created:**

| File | Responsibility |
|---|---|
| `include/SushiBLAS/engine/task.hpp` | `Internal::execute_typed_task`, `Internal::dispatch_with_mode` — the domain-agnostic task recorder (D-A3) |
| `include/SushiBLAS/engine/spatial/geometry.hpp` | `ConvGeometry`, `WindowSpan`, `ColumnSpan`, `output_extent()` — pure arithmetic, no tensor, no SYCL (D-A2) |
| `include/SushiBLAS/engine/spatial/window.hpp` | `window_span_for`, `column_span_for` — the single place a layout is interpreted |
| `include/SushiBLAS/engine/spatial.hpp` | `SpatialOps` — the public API |
| `include/SushiBLAS/kernels/blas/level3/channel_bias_epilogue.hpp` | `Kernels::ChannelBiasEpilogue` — installed, because a consumer instantiates it |
| `kernels/spatial.hpp` | aggregator for the private spatial kernels |
| `kernels/reference/spatial/*.hpp` | the pure-C++ specification of each operation |
| `kernels/sycl/spatial/*.hpp` | the device kernel for each operation |
| `src/engine/spatial/*.cpp` | one file per operation: shape checks, access set, metadata, dispatch |
| `src/engine/math/reductions/mean_spatial.cpp` | the segmented reduction over the two middle axes |
| `tests/functional/unit/test_spatial_*.cpp` | per-operation correctness |
| `tests/functional/regression/test_spatial_determinism.cpp` | CV-6: byte-for-byte across two runs |

**Modified:**

| File | Change |
|---|---|
| `include/SushiBLAS/engine/reduce_axis.hpp` | `AxisSpan` gains `segments2` / `segment_stride2`; `reduce_axis` folds over the product |
| `include/SushiBLAS/engine/math/reductions/dispatch.hpp` | `plan_axis` call sees the product; a `channel_span_for` builder |
| `include/SushiBLAS/engine/math/reductions.hpp` | `ReductionOps::mean_spatial` declaration |
| `include/SushiBLAS/engine/blas/utils.hpp` | `execute_blas_task` becomes a forwarder; `dispatch_with_mode` moves out |
| `include/SushiBLAS/engine.hpp` | `Engine::spatial()` accessor, `#include <SushiBLAS/engine/spatial.hpp>` |
| `src/CMakeLists.txt` | the new sources in `SUSHIBLAS_SOURCES` |
| `tests/CMakeLists.txt` | the new test sources |
| `docs/ARCHITECTURE.md` | §2 layer diagram gains the domain; §7 gains the spatial family and CV-6's determinism argument |
| `docs/CHANGELOG.md` | one `[Unreleased]` entry |

Headers install automatically: `src/CMakeLists.txt:215` installs `include/` as a directory. The private `kernels/` tree is reachable from this library's own `.cpp` files through the PRIVATE repo-root include path and is never installed — which is why `ChannelBiasEpilogue` is the one kernel-adjacent file that goes under `include/`, exactly as `gemm.hpp` did and for the same reason (`kernels/blas.hpp:24-34` states the rule).

---

## Task 1: The geometry

**Files:**
- Create: `include/SushiBLAS/engine/spatial/geometry.hpp`
- Test: `tests/functional/unit/test_spatial_geometry.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `SushiBLAS::Spatial::ConvGeometry`, `SushiBLAS::Spatial::WindowSpan`, `SushiBLAS::Spatial::ColumnSpan`, `constexpr int64_t SushiBLAS::Spatial::output_extent(int64_t in, int64_t pad_before, int64_t pad_after, int64_t dilation, int64_t kernel, int64_t stride)`.

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_spatial_geometry.cpp` (license header, then):

```cpp
#include <gtest/gtest.h>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Spatial::output_extent;

        TEST(Unit_SpatialGeometry, TheOrdinaryCasesMatchTheFormula)
        {
            // ResNet-18's stem: 224 -> 112, 7x7 stride 2 pad 3.
            EXPECT_EQ(output_extent(224, 3, 3, 1, 7, 2), 112);
            // A 3x3 stride-1 "same" convolution keeps its extent.
            EXPECT_EQ(output_extent(32, 1, 1, 1, 3, 1), 32);
            // Stride 2 halves it, rounding down.
            EXPECT_EQ(output_extent(32, 1, 1, 1, 3, 2), 16);
            // No padding shrinks by the kernel.
            EXPECT_EQ(output_extent(32, 0, 0, 1, 3, 1), 30);
        }

        TEST(Unit_SpatialGeometry, DilationWidensTheKernelWithoutWideningTheOutput)
        {
            // A dilated 3x3 covers 5 rows, so "same" needs 2 either side.
            EXPECT_EQ(output_extent(32, 2, 2, 2, 3, 1), 32);
            EXPECT_EQ(output_extent(32, 0, 0, 2, 3, 1), 28);
            EXPECT_EQ(output_extent(32, 0, 0, 4, 3, 1), 24);
        }

        TEST(Unit_SpatialGeometry, AsymmetricPaddingIsWhatHoldsYoloTinysLastPool)
        {
            // YOLOv3-tiny's 2x2 stride-1 max-pool, padded on the bottom and right
            // only, so 13x13 stays 13x13. This is the case CV-5 exists to make
            // ordinary, and it is the reason pad_bottom is a host-side number
            // rather than a ninth attribute slot.
            EXPECT_EQ(output_extent(13, 0, 1, 1, 2, 1), 13);
            // Symmetric padding of the same total would be a different answer,
            // which is the point: the two are not interchangeable.
            EXPECT_EQ(output_extent(13, 1, 0, 1, 2, 1), 13);
            EXPECT_EQ(output_extent(13, 0, 0, 1, 2, 1), 12);
        }

        TEST(Unit_SpatialGeometry, NonSquareAndAnisotropicGeometryIsExpressible)
        {
            // Height and width are independent all the way down, so a 1x7 kernel
            // with stride (1,2) is not a special case of anything.
            EXPECT_EQ(output_extent(20, 0, 0, 1, 1, 1), 20);
            EXPECT_EQ(output_extent(20, 3, 3, 1, 7, 2), 10);
        }

        TEST(Unit_SpatialGeometry, AWindowWiderThanThePaddedInputYieldsNoOutputRatherThanANegativeOne)
        {
            // The formula's numerator goes negative here. Integer division of a
            // negative numerator is implementation-flavoured and would produce a
            // garbage extent that later multiplies into an allocation size, so
            // it is answered explicitly.
            EXPECT_EQ(output_extent(3, 0, 0, 1, 5, 1), 0);
            EXPECT_EQ(output_extent(3, 0, 0, 4, 3, 1), 0);
        }

        TEST(Unit_SpatialGeometry, WindowSpanDerivesItsOwnCounts)
        {
            Spatial::WindowSpan span;
            span.batch = 2;
            span.channels = 3;
            span.out_h = 4;
            span.out_w = 5;
            span.geometry.kernel_h = 3;
            span.geometry.kernel_w = 3;

            EXPECT_EQ(span.windows(), 2 * 4 * 5);
            EXPECT_EQ(span.patch_elements(), 3 * 3 * 3);
            EXPECT_EQ(span.column_elements(), 2 * 4 * 5 * 3 * 3 * 3);
        }

        TEST(Unit_SpatialGeometry, TheDefaultGeometryIsTheIdentityWindow)
        {
            // A default-constructed geometry is 1x1, stride 1, no dilation, no
            // padding - so it maps every extent to itself. Anything that reads a
            // half-built geometry therefore gets a shape it can recognise as
            // wrong rather than a plausible one.
            const Spatial::ConvGeometry g;
            EXPECT_EQ(output_extent(17, 0, 0, g.dilation_h, g.kernel_h, g.stride_h), 17);
        }
    }
}
```

- [ ] **Step 2: Register the test source and run it to verify it fails**

In `tests/CMakeLists.txt`, add a group after the `# --- Level 3 ---` block:

```cmake
    # --- Spatial (windowed movement over an N-D grid) ---
    functional/unit/test_spatial_geometry.cpp
```

Run: `sb build`
Expected: FAIL — `SushiBLAS/engine/spatial/geometry.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

Create `include/SushiBLAS/engine/spatial/geometry.hpp` (license header, then):

```cpp
#pragma once

/**
 * @file geometry.hpp
 * @brief The window geometry of a spatial operation, and the extents it implies.
 *
 * Separate from `window.hpp` and free of `Tensor` on purpose. The output-extent
 * formula below is the *only* statement of how an input extent, a padding pair,
 * a dilation, a kernel and a stride combine, and it has two consumers on either
 * side of a repository boundary: this library's kernels, and SushiAI's host-lane
 * shape inference. That inference is device-free by design - it is exercised in
 * a unit tier that links no SYCL - so the arithmetic has to be reachable without
 * dragging the device stack behind it. Two copies of one formula that must agree
 * is the alternative, and they would agree until the day they did not.
 *
 * Includes `<cstdint>` and nothing else. Keep it that way.
 */

#include <cstdint>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @struct ConvGeometry
         * @brief A 2-D sliding window: its size, its step, its spacing, its origin.
         *
         * Exactly eight numbers, and the count is load-bearing rather than tidy.
         * SushiAI carries this into `Graph::Attributes`, whose capacity is
         * bounded by `SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS` - eight.
         *
         * **`pad_bottom` and `pad_right` are deliberately absent.** A kernel
         * addresses its input as `h_in = h_o * stride_h - pad_top + r * dilation_h`
         * and bounds-checks against `[0, H)`. The trailing pads never enter that
         * expression; they affect only *how many output rows exist*, which is
         * @ref output_extent's answer and which the host has already written into
         * the output tensor's shape by the time any of this reaches a device. So
         * asymmetric padding costs zero slots over symmetric padding, and every
         * geometry is expressible: non-square kernels, anisotropic stride and
         * dilation, and independent padding on all four sides.
         */
        struct ConvGeometry
        {
            /** @brief Window height, in input rows. */
            int64_t kernel_h = 1;
            /** @brief Window width, in input columns. */
            int64_t kernel_w = 1;
            /** @brief Rows between the origins of vertically adjacent windows. */
            int64_t stride_h = 1;
            /** @brief Columns between the origins of horizontally adjacent windows. */
            int64_t stride_w = 1;
            /** @brief Rows between vertically adjacent taps of one window. */
            int64_t dilation_h = 1;
            /** @brief Columns between horizontally adjacent taps of one window. */
            int64_t dilation_w = 1;
            /** @brief Implicit zero rows above row 0. */
            int64_t pad_top = 0;
            /** @brief Implicit zero columns left of column 0. */
            int64_t pad_left = 0;
        };

        /**
         * @brief How many windows fit along one axis.
         *
         * `floor((in + pad_before + pad_after - dilation*(kernel - 1) - 1) / stride) + 1`,
         * which is the same expression PyTorch, Darknet and cuDNN all use.
         *
         * A window wider than the padded input answers **0** rather than letting
         * the numerator go negative. That is not defensiveness: the result is
         * multiplied into an allocation size two layers up, and a negative
         * numerator under integer division would produce a plausible-looking
         * positive extent for some operand combinations.
         *
         * @param in The input extent along this axis.
         * @param pad_before Implicit zeros before index 0.
         * @param pad_after Implicit zeros after the last index.
         * @param dilation Spacing between taps; 1 for an undilated window.
         * @param kernel The window's extent along this axis.
         * @param stride The step between consecutive windows.
         * @return The output extent, or 0 when no window fits.
         */
        [[nodiscard]] constexpr int64_t output_extent(int64_t in,
                                                      int64_t pad_before, int64_t pad_after,
                                                      int64_t dilation, int64_t kernel,
                                                      int64_t stride) noexcept
        {
            if (stride <= 0 || kernel <= 0 || dilation <= 0) return 0;

            const int64_t covered = dilation * (kernel - 1) + 1;
            const int64_t padded = in + pad_before + pad_after;
            if (padded < covered) return 0;

            return (padded - covered) / stride + 1;
        }

        /**
         * @struct WindowSpan
         * @brief Where one spatial operation's input and output live, in element counts.
         *
         * Modelled on @ref SushiBLAS::Reduce::AxisSpan, and for the same reason:
         * it carries counts and strides only, so the layout, the axis order and
         * any non-unit stride a view inherited are resolved once on the host and
         * the device sees none of them. One kernel per operation, no layout
         * branch in the inner loop, no second instantiation.
         *
         * The struct names no layout. Every caller in the convolutional stack
         * builds it from an NHWC tensor (see the spec's CV-3), but adding NCHW
         * later is filling in a different stride assignment in
         * @ref window_span_for, not re-cutting this abstraction.
         *
         * All strides are element counts and all are non-negative, which
         * `Tensor`'s stride-explicit constructor already guarantees.
         */
        struct WindowSpan
        {
            /** @brief How many independent images. */
            int64_t batch = 0;
            /** @brief The input's row count. */
            int64_t in_h = 0;
            /** @brief The input's column count. */
            int64_t in_w = 0;
            /** @brief The channel count, shared by input and output. */
            int64_t channels = 0;
            /** @brief The output's row count; @ref output_extent's answer. */
            int64_t out_h = 0;
            /** @brief The output's column count; @ref output_extent's answer. */
            int64_t out_w = 0;

            /** @brief Input elements between consecutive images. */
            int64_t in_batch_stride = 0;
            /** @brief Input elements between vertically adjacent pixels. */
            int64_t in_row_stride = 0;
            /** @brief Input elements between horizontally adjacent pixels. */
            int64_t in_col_stride = 0;
            /** @brief Input elements between adjacent channels of one pixel. */
            int64_t in_channel_stride = 1;

            /** @brief Output elements between consecutive images. */
            int64_t out_batch_stride = 0;
            /** @brief Output elements between vertically adjacent pixels. */
            int64_t out_row_stride = 0;
            /** @brief Output elements between horizontally adjacent pixels. */
            int64_t out_col_stride = 0;
            /** @brief Output elements between adjacent channels of one pixel. */
            int64_t out_channel_stride = 1;

            /** @brief The window itself. */
            ConvGeometry geometry{};

            /** @brief How many windows the operation visits: `batch * out_h * out_w`. */
            [[nodiscard]] constexpr int64_t windows() const noexcept
            {
                return batch * out_h * out_w;
            }

            /** @brief How many elements one window contributes: `kernel_h * kernel_w * channels`. */
            [[nodiscard]] constexpr int64_t patch_elements() const noexcept
            {
                return geometry.kernel_h * geometry.kernel_w * channels;
            }

            /** @brief How many elements an im2col matrix holds: `windows() * patch_elements()`. */
            [[nodiscard]] constexpr int64_t column_elements() const noexcept
            {
                return windows() * patch_elements();
            }

            /** @brief How many elements the input holds: `batch * in_h * in_w * channels`. */
            [[nodiscard]] constexpr int64_t input_elements() const noexcept
            {
                return batch * in_h * in_w * channels;
            }

            /** @brief How many elements the output holds: `windows() * channels`. */
            [[nodiscard]] constexpr int64_t output_elements() const noexcept
            {
                return windows() * channels;
            }
        };

        /**
         * @struct ColumnSpan
         * @brief How an im2col matrix is addressed: a row stride and an element stride.
         *
         * Kept out of @ref WindowSpan because the column matrix is rank-2 and
         * everything else in this family is rank-4. Folding two unrelated
         * addressing schemes into one struct would leave four fields meaningless
         * for every operation but two.
         *
         * Entry `(row, element)` lives at `data[row * row_stride + element * element_stride]`,
         * where `row` indexes a window in `(n, h_o, w_o)` order and `element`
         * indexes a patch position in `(r, s, c)` order - channel innermost, so
         * an NHWC gather copies contiguous runs of `channels` elements.
         */
        struct ColumnSpan
        {
            /** @brief Elements between the starts of consecutive rows. */
            int64_t row_stride = 0;
            /** @brief Elements between consecutive entries of one row. */
            int64_t element_stride = 1;
        };

    } // namespace Spatial
} // namespace SushiBLAS
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `sb build` then `sb test --suite unit --filter "Unit_SpatialGeometry.*"`
Expected: 7 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add include/SushiBLAS/engine/spatial/geometry.hpp tests/functional/unit/test_spatial_geometry.cpp tests/CMakeLists.txt
git commit -m "feat(spatial): add the window geometry and its output-extent formula"
```

---

## Task 2: The domain-agnostic task recorder

Refactor only — no behaviour changes and no new operation. It is its own task because it touches every BLAS op's execution path, and a reviewer should be able to reject it without rejecting `im2col`.

**Files:**
- Create: `include/SushiBLAS/engine/task.hpp`
- Modify: `include/SushiBLAS/engine/blas/utils.hpp:67-181`
- Test: `tests/functional/unit/test_dispatch_seam.cpp` (append)

**Interfaces:**
- Consumes: `Dispatch::Domain`, `Dispatch::require_domain` from `engine/dispatch.hpp`.
- Produces: `template<Dispatch::Domain Dom, typename Func> sycl::event Internal::execute_typed_task(Engine&, const SushiRuntime::Graph::TaskMetadata&, const std::vector<SushiRuntime::Core::ResourceRegion>& reads, const std::vector<SushiRuntime::Core::ResourceRegion>& writes, Core::DataType, const char* family, Func&& op_func)`; and `Internal::dispatch_with_mode` unchanged, now declared here.

- [ ] **Step 1: Write the failing test**

Append to `tests/functional/unit/test_dispatch_seam.cpp`, inside the existing `SushiBLAS::Test` namespace:

```cpp
        // The family string is what a caller outside BLAS needs from this seam:
        // an op recorded by the spatial domain must not be reported as a BLAS
        // operation when its dtype is refused. Pinned here rather than left to
        // the spatial tests, because the guarantee belongs to the seam.
        TEST_F(Unit_DispatchSeamTest, ATypedTaskReportsTheFamilyItsCallerDeclared)
        {
            auto t = engine->create_tensor({4}, Core::DataType::COMPLEX32);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = "spatial.im2col";
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;

            try
            {
                (void)Internal::execute_typed_task<Dispatch::Domain::REAL>(
                    *engine, meta, {}, {}, t.dtype, "Spatial operation",
                    [](auto, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event
                    {
                        return sycl::event();
                    });
                FAIL() << "a complex dtype must be refused by a REAL-only task";
            }
            catch (const std::exception& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("Spatial operation"), std::string::npos) << message;
                EXPECT_EQ(message.find("BLAS operation"), std::string::npos) << message;
                EXPECT_NE(message.find("spatial.im2col"), std::string::npos) << message;
            }
        }
```

If the existing fixture in that file is not named `Unit_DispatchSeamTest`, use whatever name it already declares; do not rename it.

- [ ] **Step 2: Run it to verify it fails**

Run: `sb build`
Expected: FAIL — `no member named 'execute_typed_task' in namespace 'SushiBLAS::Internal'`.

- [ ] **Step 3: Move the body into the new header**

Create `include/SushiBLAS/engine/task.hpp` (license header, then):

```cpp
#pragma once

/**
 * @file task.hpp
 * @brief Recording one dispatched task on the engine's graph, for any domain.
 *
 * This is `engine/blas/utils.hpp`'s `execute_blas_task` with the one thing that
 * was BLAS-specific about it - the word "BLAS" in its diagnostics - lifted into
 * a parameter. Nothing else changed, and `execute_blas_task` still exists and
 * still says "BLAS operation"; it forwards here.
 *
 * The move happened because a second domain arrived. `spatial.im2col` reported
 * as a "BLAS operation" when its dtype is refused sends a reader to the wrong
 * half of the library, and the alternative - a second copy of the seven-arm
 * dtype switch - is seven arms to forget when a `DataType` is added.
 */

#include <vector>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    class Engine;
    enum class ExecutionMode;

    namespace Internal
    {
        /**
         * @brief Registers one generically-dispatched task on @p engine's graph.
         *
         * One `add_task`, one `switch (dtype)` dispatching a generically-templated
         * closure. An unsupported dtype fails loudly at record time, never
         * silently no-ops.
         *
         * @tparam Dom Which element types to instantiate @p op_func for.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param meta The task metadata to record.
         * @param reads Regions the task reads.
         * @param writes Regions the task writes.
         * @param dtype The operands' runtime element type.
         * @param family What kind of operation this is, for diagnostics -
         *        "BLAS operation", "Spatial operation".
         * @param op_func A callable
         *        `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`,
         *        generic over the runtime dtype via `decltype(scalar_type)`.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         * @throws std::runtime_error at record time if @p dtype falls outside @p Dom.
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_typed_task(Engine& engine,
                                       const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, const char* family, Func&& op_func);

        /**
         * @brief Collapses the FORCE_REFERENCE / normal / VERIFY execution-mode triad.
         *
         * @tparam ReferenceFn The reference callable's type, deduced.
         * @tparam SyclFn The device callable's type, deduced.
         * @param mode The engine's current execution mode.
         * @param q The queue to submit to.
         * @param deps Events this submission must depend on.
         * @param reference_fn A callable `() -> void` running the pure-C++ reference kernel.
         * @param sycl_fn A callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`
         *        running the SYCL kernel.
         * @return The submitted event, or an empty one under FORCE_REFERENCE.
         */
        template<typename ReferenceFn, typename SyclFn>
        sycl::event dispatch_with_mode(ExecutionMode mode, sycl::queue& q,
                                       const std::vector<sycl::event>& deps,
                                       ReferenceFn&& reference_fn, SyclFn&& sycl_fn);
    }
}
```

**Note on the definitions.** `execute_typed_task` calls `engine.get_graph()` and `dispatch_with_mode` compares against `ExecutionMode`, so both need `Engine` complete. `blas/utils.hpp` gets away with this today because it is only ever included after `engine.hpp`. Preserve that: put the *definitions* at the bottom of this same header, after an `#include <SushiBLAS/engine.hpp>` guarded the way `blas/utils.hpp` currently arranges its own includes. Check what `blas/utils.hpp` does at its top and mirror it exactly — if it relies on the includer having pulled `engine.hpp` first, this header does the same and says so in a comment. Do not introduce a new inclusion order that only works by luck.

The definitions are the current bodies from `blas/utils.hpp:93-179`, with two edits and no others:

1. `Dispatch::require_domain<Dom>(dtype, "BLAS operation", meta.name);` becomes `Dispatch::require_domain<Dom>(dtype, family, meta.name);`
2. the closure captures `family` alongside `dtype`, `op_func` and `name`, and its trailing throw becomes
   `SB_THROW_IF(true, "{} '{}' does not support dtype {}.", family, name, Core::to_string(dtype));`

- [ ] **Step 4: Make `blas/utils.hpp` forward**

In `include/SushiBLAS/engine/blas/utils.hpp`, add `#include <SushiBLAS/engine/task.hpp>`, delete the `dispatch_with_mode` definition entirely (it now lives in `task.hpp` and reaches every existing caller through this include), and replace `execute_blas_task`'s body with:

```cpp
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_blas_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, Func&& op_func)
        {
            // The family string is the whole of what this wrapper adds. Kept as a
            // named function rather than replaced at its twenty-odd call sites:
            // "BLAS operation" is the right words for every one of them, and
            // spelling it twenty times is twenty chances to spell it differently.
            return execute_typed_task<Dom>(engine, meta, reads, writes, dtype,
                                           "BLAS operation", std::forward<Func>(op_func));
        }
```

Keep the long explanatory comment about why the switch is spelled out by hand rather than routed through `with_element_type` — move it to `task.hpp` with the code it explains. It documents a live clang bug and losing it would cost someone the rediscovery.

- [ ] **Step 5: Run the whole unit suite**

Run: `sb build` then `sb test --suite unit`
Expected: every previously passing test still passes, plus `ATypedTaskReportsTheFamilyItsCallerDeclared`. A refactor that changes a result is not a refactor — if anything else moves, stop and find out why before continuing.

- [ ] **Step 6: Commit**

```bash
git add include/SushiBLAS/engine/task.hpp include/SushiBLAS/engine/blas/utils.hpp tests/functional/unit/test_dispatch_seam.cpp
git commit -m "refactor(engine): lift the task recorder out of the BLAS namespace"
```

---

## Task 3: `im2col`, and the domain it arrives in

The first operation carries the domain's scaffolding: `window_span_for`, `SpatialOps`, `Engine::spatial()`, the CMake wiring and the kernel directory layout. Folded into one task because none of it is independently testable — a `SpatialOps` with no operations proves nothing.

**Files:**
- Create: `include/SushiBLAS/engine/spatial/window.hpp`, `include/SushiBLAS/engine/spatial.hpp`, `kernels/spatial.hpp`, `kernels/reference/spatial/im2col.hpp`, `kernels/sycl/spatial/im2col.hpp`, `src/engine/spatial/im2col.cpp`
- Modify: `include/SushiBLAS/engine.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/functional/unit/test_spatial_im2col.cpp`

**Interfaces:**
- Consumes: Task 1's `Spatial::WindowSpan` / `ColumnSpan` / `ConvGeometry` / `output_extent`; Task 2's `Internal::execute_typed_task` / `Internal::dispatch_with_mode`.
- Produces:
  - `Spatial::WindowSpan Spatial::window_span_for(const Tensor& in, const ConvGeometry& g, const Tensor& out, const char* name)` — the rank-4-output form, which reads `out_h`/`out_w` off `out`
  - `Spatial::WindowSpan Spatial::window_span_for(const Tensor& in, const ConvGeometry& g, int64_t out_h, int64_t out_w, const char* name)` — the extents form, for operations whose output is not a rank-4 image (`im2col`, `col2im`)
  - `Spatial::ColumnSpan Spatial::column_span_for(const Tensor& cols, const WindowSpan& span, const char* name)`
  - `class SpatialOps` with `explicit SpatialOps(Engine&)`, `sycl::event im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g)` and its explicit-extent overload `sycl::event im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g, int64_t out_h, int64_t out_w)`
  - `SpatialOps Engine::spatial()`
  - `template<typename T> void Kernels::Reference::im2col(const Spatial::WindowSpan&, const Spatial::ColumnSpan&, const T* in, T* out)`
  - `template<typename T> sycl::event Kernels::SYCL::im2col(sycl::queue&, const Spatial::WindowSpan&, const Spatial::ColumnSpan&, const T* in, T* out, const std::vector<sycl::event>&)`

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_spatial_im2col.cpp` (license header, then):

```cpp
#include <vector>
#include <numeric>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_SpatialIm2ColTest : public SushiBLASTest
        {
            protected:
                /** @brief An NHWC tensor holding 0, 1, 2, ... in memory order. */
                Tensor ramp(int64_t n, int64_t h, int64_t w, int64_t c)
                {
                    auto t = engine->create_tensor({n, h, w, c}, Core::DataType::FLOAT32);
                    float* p = t.data_as<float>();
                    for (int64_t i = 0; i < t.num_elements; ++i)
                        p[i] = static_cast<float>(i);
                    return t;
                }
        };

        TEST_F(Unit_SpatialIm2ColTest, AUnitWindowIsTheIdentityGather)
        {
            // 1x1, stride 1, no padding: the column matrix is the input reshaped,
            // element for element and in the same order. If this fails, the row
            // and element index arithmetic disagree with each other and nothing
            // further is worth reading.
            auto in = ramp(2, 3, 4, 5);
            auto cols = engine->create_tensor({2 * 3 * 4, 5}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;   // 1x1, stride 1, dilation 1, no pad
            engine->spatial().im2col(in, cols, g);
            engine->execute();

            const float* out = cols.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                EXPECT_FLOAT_EQ(out[i], static_cast<float>(i)) << "element " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, EachRowIsOneWindowWithChannelInnermost)
        {
            // 2x2 stride 1 over a 1x3x3x2 input: 4 windows, 8 entries each.
            // Written out by hand rather than derived, so the test is a
            // specification and not a second implementation.
            auto in = ramp(1, 3, 3, 2);
            auto cols = engine->create_tensor({4, 8}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // Input element (h, w, c) holds (h*3 + w)*2 + c.
            // Row 0 is the window at (0,0): taps (0,0) (0,1) (1,0) (1,1).
            const std::vector<float> row0 = {0, 1, 2, 3, 6, 7, 8, 9};
            // Row 1 is the window at (0,1): taps (0,1) (0,2) (1,1) (1,2).
            const std::vector<float> row1 = {2, 3, 4, 5, 8, 9, 10, 11};
            // Row 3 is the window at (1,1): taps (1,1) (1,2) (2,1) (2,2).
            const std::vector<float> row3 = {8, 9, 10, 11, 14, 15, 16, 17};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < row0.size(); ++i)
            {
                EXPECT_FLOAT_EQ(out[0 * 8 + i], row0[i]) << "row 0 entry " << i;
                EXPECT_FLOAT_EQ(out[1 * 8 + i], row1[i]) << "row 1 entry " << i;
                EXPECT_FLOAT_EQ(out[3 * 8 + i], row3[i]) << "row 3 entry " << i;
            }
        }

        TEST_F(Unit_SpatialIm2ColTest, OutOfBoundsTapsReadZeroRatherThanNeighbouringMemory)
        {
            // Padding is a parameter, not a separate zero-padded copy - which is
            // what saves a full-tensor copy forward and a crop backward. The
            // property that makes that safe is exactly this one.
            auto in = ramp(1, 2, 2, 1);
            auto cols = engine->create_tensor({4, 9}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // Row 0 is the window centred on pixel (0,0); its first tap sits at
            // (-1,-1) and its last at (1,1). Only the four in-bounds taps carry
            // data: positions 4, 5, 7, 8 of the 3x3.
            const std::vector<float> row0 = {0, 0, 0, 0, 0, 1, 0, 2, 3};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < row0.size(); ++i)
                EXPECT_FLOAT_EQ(out[i], row0[i]) << "entry " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, DilationSpacesTheTapsWithoutMovingTheWindows)
        {
            auto in = ramp(1, 5, 5, 1);
            auto cols = engine->create_tensor({1, 9}, Core::DataType::FLOAT32);

            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.dilation_h = 2;
            g.dilation_w = 2;

            engine->spatial().im2col(in, cols, g);
            engine->execute();

            // The single window's taps are rows {0,2,4} x columns {0,2,4}.
            const std::vector<float> expected = {0, 2, 4, 10, 12, 14, 20, 22, 24};

            const float* out = cols.data_as<float>();
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_FLOAT_EQ(out[i], expected[i]) << "entry " << i;
        }

        TEST_F(Unit_SpatialIm2ColTest, TheSyclKernelAgreesWithTheReferenceKernel)
        {
            // VERIFY runs both and compares, which is what makes the reference
            // kernel this operation's specification rather than a second guess.
            engine->set_mode(ExecutionMode::VERIFY);

            auto in = ramp(2, 7, 5, 3);
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 1;
            g.dilation_h = 2;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t out_h = Spatial::output_extent(7, 1, 1, 2, 3, 2);
            const int64_t out_w = Spatial::output_extent(5, 1, 1, 1, 2, 1);
            auto cols = engine->create_tensor({2 * out_h * out_w, 3 * 2 * 3},
                                              Core::DataType::FLOAT32);

            EXPECT_NO_THROW({
                engine->spatial().im2col(in, cols, g);
                engine->execute();
            });
        }

        TEST_F(Unit_SpatialIm2ColTest, AColumnMatrixOfTheWrongShapeIsRefusedByName)
        {
            auto in = ramp(1, 4, 4, 2);
            auto cols = engine->create_tensor({16, 7}, Core::DataType::FLOAT32);   // 7 != 2*2*2

            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;

            try
            {
                engine->spatial().im2col(in, cols, g);
                FAIL() << "a column matrix whose width is not kernel_h*kernel_w*C must be refused";
            }
            catch (const std::exception& e)
            {
                const std::string message = e.what();
                EXPECT_NE(message.find("spatial.im2col"), std::string::npos) << message;
                EXPECT_NE(message.find("7"), std::string::npos) << message;
                EXPECT_NE(message.find("8"), std::string::npos) << message;
            }
        }

        TEST_F(Unit_SpatialIm2ColTest, ARankThatIsNotFourIsRefusedAtRecordTime)
        {
            auto in = engine->create_tensor({4, 4, 2}, Core::DataType::FLOAT32);
            auto cols = engine->create_tensor({16, 8}, Core::DataType::FLOAT32);

            EXPECT_THROW(engine->spatial().im2col(in, cols, Spatial::ConvGeometry{}),
                         std::exception);
        }
    }
}
```

- [ ] **Step 2: Register the test and run it to verify it fails**

Add `functional/unit/test_spatial_im2col.cpp` to the spatial group in `tests/CMakeLists.txt`.

Run: `sb build`
Expected: FAIL — `no member named 'spatial' in 'SushiBLAS::Engine'`.

- [ ] **Step 3: Write `window.hpp`**

Create `include/SushiBLAS/engine/spatial/window.hpp` (license header, then):

```cpp
#pragma once

/**
 * @file window.hpp
 * @brief Building a @ref SushiBLAS::Spatial::WindowSpan from tensors.
 *
 * The single place a layout is interpreted. Everything downstream of this
 * function sees element counts, so adding NCHW later is a second stride
 * assignment here and nothing else anywhere.
 */

#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Spatial
    {
        /**
         * @brief Describes a windowed operation whose output extents are stated.
         *
         * The form `im2col` and `col2im` take: their other operand is a rank-2
         * column matrix, so there is no output image to read `H_o` and `W_o`
         * off, and the caller states them.
         *
         * Reads the strides off @p in rather than deriving them from its shape,
         * so a view with a non-unit stride - a channel slice, a
         * `storage_offset` into a shared allocation - is addressed correctly
         * without any caller knowing it was one.
         *
         * **What it can and cannot check.** The extents are taken, not
         * recomputed, because @ref output_extent needs `pad_bottom` and
         * `pad_right` and @ref ConvGeometry deliberately does not carry them
         * (see its documentation). What it can prove is that every window's
         * *origin* lands inside the padded input: an output row whose window
         * starts at or past `in_h` reads nothing but padding, which is always a
         * caller's arithmetic error and never a legitimate request.
         *
         * @param in The rank-4 NHWC input.
         * @param g The window geometry.
         * @param out_h The output's row count.
         * @param out_w The output's column count.
         * @param name The operation's name, for diagnostics.
         * @return The span, in element counts; the `out_*_stride` fields are
         *         left at zero, since there is no output image to describe.
         * @throws std::runtime_error when @p in is not rank 4, or a window
         *         origin falls outside the padded input.
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        int64_t out_h, int64_t out_w,
                                                        const char* name)
        {
            SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                        name, in.rank);

            WindowSpan span;
            span.batch = in.shape[0];
            span.in_h = in.shape[1];
            span.in_w = in.shape[2];
            span.channels = in.shape[3];
            span.out_h = out_h;
            span.out_w = out_w;

            span.in_batch_stride = in.strides[0];
            span.in_row_stride = in.strides[1];
            span.in_col_stride = in.strides[2];
            span.in_channel_stride = in.strides[3];

            span.geometry = g;

            // A window whose origin is at or past the last input row covers only
            // padding. Legal arithmetic, never a legal request - and the shape it
            // produces is plausible enough to survive every other check here.
            SB_THROW_IF(out_h > 0 && (out_h - 1) * g.stride_h - g.pad_top >= span.in_h,
                        "Spatial operation '{}' asks for {} output rows, but window {} starts at "
                        "input row {}, which is outside a {}-row input padded by {} above.",
                        name, out_h, out_h - 1, (out_h - 1) * g.stride_h - g.pad_top,
                        span.in_h, g.pad_top);
            SB_THROW_IF(out_w > 0 && (out_w - 1) * g.stride_w - g.pad_left >= span.in_w,
                        "Spatial operation '{}' asks for {} output columns, but window {} starts at "
                        "input column {}, which is outside a {}-column input padded by {} on the left.",
                        name, out_w, out_w - 1, (out_w - 1) * g.stride_w - g.pad_left,
                        span.in_w, g.pad_left);

            return span;
        }

        /**
         * @brief Describes a windowed operation from its two rank-4 NHWC operands.
         *
         * Spec §4's signature, and the form every operation with an image
         * output takes - pooling, upsampling, the channel copies, the affine.
         * `H_o` and `W_o` come from @p out, which the caller sized with
         * @ref output_extent while the trailing padding it needs was still in
         * scope.
         *
         * @param in The rank-4 NHWC input.
         * @param g The window geometry.
         * @param out The rank-4 NHWC output.
         * @param name The operation's name, for diagnostics.
         * @return The span, in element counts, with both operands' strides filled in.
         * @throws std::runtime_error when a rank is not 4, the batch or channel
         *         counts disagree, or a window origin falls outside the padded input.
         */
        [[nodiscard]] inline WindowSpan window_span_for(const Tensor& in, const ConvGeometry& g,
                                                        const Tensor& out, const char* name)
        {
            SB_THROW_IF(out.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC output; got rank {}.",
                        name, out.rank);

            WindowSpan span = window_span_for(in, g, out.shape[1], out.shape[2], name);

            SB_THROW_IF(out.shape[0] != span.batch,
                        "Spatial operation '{}' has {} input images and {} output images.",
                        name, span.batch, out.shape[0]);
            SB_THROW_IF(out.shape[3] != span.channels,
                        "Spatial operation '{}' has {} input channels and {} output channels.",
                        name, span.channels, out.shape[3]);

            span.out_batch_stride = out.strides[0];
            span.out_row_stride = out.strides[1];
            span.out_col_stride = out.strides[2];
            span.out_channel_stride = out.strides[3];

            return span;
        }

        /**
         * @brief Describes an im2col matrix's addressing, and checks its width.
         *
         * @param cols The rank-2 column matrix.
         * @param span The window span the matrix must match.
         * @param name The operation's name, for diagnostics.
         * @return The column addressing, in element counts.
         * @throws std::runtime_error when @p cols is not rank 2, or its extents
         *         are not `span.windows()` by `span.patch_elements()`.
         */
        [[nodiscard]] inline ColumnSpan column_span_for(const Tensor& cols, const WindowSpan& span,
                                                        const char* name)
        {
            SB_THROW_IF(cols.rank != 2,
                        "Spatial operation '{}' needs a rank-2 column matrix; got rank {}.",
                        name, cols.rank);
            SB_THROW_IF(cols.shape[0] != span.windows(),
                        "Spatial operation '{}' produces {} windows but the column matrix has {} rows.",
                        name, span.windows(), cols.shape[0]);
            SB_THROW_IF(cols.shape[1] != span.patch_elements(),
                        "Spatial operation '{}' produces patches of {} elements "
                        "(kernel_h {} * kernel_w {} * channels {}) but the column matrix is {} wide.",
                        name, span.patch_elements(), span.geometry.kernel_h,
                        span.geometry.kernel_w, span.channels, cols.shape[1]);

            return ColumnSpan{cols.strides[0], cols.strides[1]};
        }

    } // namespace Spatial
} // namespace SushiBLAS
```

- [ ] **Step 4: Write the reference kernel**

Create `kernels/reference/spatial/im2col.hpp` (license header, then):

```cpp
#pragma once

#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Reference
        {
            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             *
             * This is `im2col`'s specification: `ExecutionMode::VERIFY` runs it
             * against the SYCL kernel and compares. Written as the obvious nest
             * of loops, in the obvious order, because a reference kernel that is
             * clever is a second implementation to be wrong rather than a
             * ground truth.
             *
             * Out-of-bounds taps write zero. Padding never materialises as a
             * copy of the input, which is what saves a full-tensor pass here and
             * a crop in `col2im`.
             *
             * @tparam T The element type.
             * @param span The window layout, in element counts.
             * @param cols The column matrix's addressing.
             * @param in The input's first element.
             * @param out The column matrix's first element.
             */
            template<typename T>
            void im2col(const Spatial::WindowSpan& span, const Spatial::ColumnSpan& cols,
                        const T* in, T* out)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                {
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                    {
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                        {
                            const int64_t row = (n * span.out_h + h_o) * span.out_w + w_o;

                            for (int64_t r = 0; r < g.kernel_h; ++r)
                            {
                                const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                const bool row_ok = h_in >= 0 && h_in < span.in_h;

                                for (int64_t s = 0; s < g.kernel_w; ++s)
                                {
                                    const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                    const bool ok = row_ok && w_in >= 0 && w_in < span.in_w;

                                    const int64_t patch = (r * g.kernel_w + s) * span.channels;
                                    const int64_t base = n * span.in_batch_stride
                                                       + h_in * span.in_row_stride
                                                       + w_in * span.in_col_stride;

                                    for (int64_t c = 0; c < span.channels; ++c)
                                    {
                                        out[row * cols.row_stride + (patch + c) * cols.element_stride] =
                                            ok ? in[base + c * span.in_channel_stride] : T(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 5: Write the SYCL kernel**

Create `kernels/sycl/spatial/im2col.hpp` (license header, then):

```cpp
#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             *
             * One work-item per *output* element, which is what makes this a pure
             * gather: every item writes exactly one entry and reads exactly one
             * input element (or none, for a padded tap). No two items write the
             * same address, so there is nothing to synchronise and nothing whose
             * combination order could vary between runs.
             *
             * Consecutive items differ in `c` first, so a warp reads a
             * contiguous run of channels - which is the NHWC half of the
             * argument for choosing that layout.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The window layout, in element counts.
             * @param cols The column matrix's addressing.
             * @param in The input's first element.
             * @param out The column matrix's first element.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event im2col(sycl::queue& q, const Spatial::WindowSpan& span,
                               const Spatial::ColumnSpan& cols,
                               const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const Spatial::ConvGeometry g = span.geometry;
                const int64_t patch_width = span.patch_elements();
                const int64_t total = span.column_elements();
                if (total == 0) return sycl::event();

                // Copied out by value: the closure below is device code and may
                // not chase a host reference.
                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t row_stride = cols.row_stride;
                const int64_t element_stride = cols.element_stride;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0);
                             flat < static_cast<std::size_t>(total); flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const int64_t col = i % patch_width;
                            const int64_t row = i / patch_width;

                            const int64_t c = col % channels;
                            const int64_t rs = col / channels;
                            const int64_t s = rs % g.kernel_w;
                            const int64_t r = rs / g.kernel_w;

                            const int64_t w_o = row % out_w;
                            const int64_t nh = row / out_w;
                            const int64_t h_o = nh % out_h;
                            const int64_t n = nh / out_h;

                            const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                            const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;

                            const bool ok = h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w;

                            out[row * row_stride + col * element_stride] = ok
                                ? in[n * in_batch_stride + h_in * in_row_stride
                                     + w_in * in_col_stride + c * in_channel_stride]
                                : T(0);
                        }
                    });
                });
            }
        }
    }
}
```

Confirm the exact spellings of `Device::Grid1D`, `Device::plan_1d`, `Device::profile_for`, `grid.stride()` and `grid.nd_range()` against `include/SushiBLAS/engine/launch_1d.hpp` and `reduce_axis.hpp:276-283` before building — the call in `reduce_axis` is the working example to copy.

- [ ] **Step 6: Write the kernel aggregator**

Create `kernels/spatial.hpp` (license header, then):

```cpp
#pragma once

// Private to this tree, spelled with quoted repo-root-relative paths, for the
// reason kernels/blas.hpp states: a kernel a *public* seam instantiates in a
// consumer's translation unit has to live under include/, and nothing here is
// one. The spatial operations are reached only through SpatialOps, whose
// methods are compiled into this library.

#include "kernels/reference/spatial/im2col.hpp"
#include "kernels/sycl/spatial/im2col.hpp"
```

- [ ] **Step 7: Write `SpatialOps`**

Create `include/SushiBLAS/engine/spatial.hpp` (license header, then):

```cpp
#pragma once

/**
 * @file spatial.hpp
 * @brief Windowed and strided movement over an N-D grid.
 *
 * The word "convolution" does not appear in this domain and should not be added
 * to it. `im2col` is a gather, `pool_max` is a windowed reduction,
 * `channel_affine` is a broadcast elementwise. Parameter ownership, training
 * mode and differentiation belong to the layer above; what belongs here is
 * moving numbers around a grid.
 *
 * Every operation in this domain has both a pure-C++ reference kernel and a
 * SYCL one, so `ExecutionMode::VERIFY` has a ground truth.
 *
 * **Every backward here is a gather, never an atomic scatter.** SushiBLAS links
 * `sushiruntime_deterministic_fp` PUBLIC, and a floating-point atomic add has a
 * combination order that varies between runs - so `col2im` written as "each
 * column entry atomically adds into its pixel" would silently break the
 * library's bit-reproducibility guarantee. Each output element gathers its own
 * contributions in a fixed order instead. This is the same argument
 * `docs/ARCHITECTURE.md` §7 gives for rejecting `sycl::reduction` in the folding
 * reductions, and `tests/functional/regression/test_spatial_determinism.cpp` is
 * what keeps it true.
 */

#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    class Engine;

    /**
     * @class SpatialOps
     * @brief Windowed gathers, windowed reductions and channel-wise broadcasts.
     */
    class SpatialOps
    {
        public:
            /**
             * @brief Construct SpatialOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit SpatialOps(Engine& e) : engine_(e) {}

            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             *
             * `[N, H, W, C]` to `[N * H_o * W_o, C * R * S]`, with the patch
             * index running `(r, s, c)` - channel innermost. Out-of-bounds taps
             * are zero; padding is a parameter and never a materialised copy.
             *
             * The output's extents are read from @p cols rather than recomputed,
             * because @ref Spatial::ConvGeometry carries no trailing padding
             * (see @ref Spatial::output_extent for why) - so `H_o` is the
             * caller's arithmetic and this checks it for consistency instead of
             * repeating it.
             *
             * @param in The rank-4 NHWC input.
             * @param cols The rank-2 column matrix to fill.
             * @param g The window geometry.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
```

- [ ] **Step 8: Write the operation**

Create `src/engine/spatial/im2col.cpp` (license header, then):

```cpp
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event SpatialOps::im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.im2col";

        SB_THROW_IF(in.dtype != cols.dtype,
                    "Spatial operation '{}' has a {} input and a {} column matrix.",
                    NAME, Core::to_string(in.dtype), Core::to_string(cols.dtype));
        SB_THROW_IF(cols.rank != 2, "Spatial operation '{}' needs a rank-2 column matrix; got rank {}.",
                    NAME, cols.rank);

        // The window count comes from the column matrix, which the caller sized
        // from Spatial::output_extent - the one formula, on the host, where the
        // trailing padding it needs is still in scope.
        SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                    NAME, in.rank);
        const int64_t batch = in.shape[0];
        SB_THROW_IF(batch <= 0 || cols.shape[0] % batch != 0,
                    "Spatial operation '{}' cannot split {} column rows across {} images.",
                    NAME, cols.shape[0], batch);
        const int64_t windows_per_image = cols.shape[0] / batch;

        // H_o and W_o are not separately recoverable from the row count alone,
        // so they are carried on the geometry's own terms: the caller states the
        // output width and the height follows. Passing both explicitly is what
        // the SushiAI layer will do; here the column matrix plus the geometry
        // pins one of them and the division gives the other.
        const int64_t out_w = Spatial::output_extent(in.shape[2], g.pad_left,
                                                     g.pad_left, g.dilation_w,
                                                     g.kernel_w, g.stride_w);
        SB_THROW_IF(out_w <= 0 || windows_per_image % out_w != 0,
                    "Spatial operation '{}' derived {} output columns, which does not divide the "
                    "{} windows per image the column matrix asks for.",
                    NAME, out_w, windows_per_image);
        const int64_t out_h = windows_per_image / out_w;

        // The extents overload, not the two-tensor one: the other operand here
        // is a rank-2 column matrix, so there is no output image whose shape
        // could supply H_o and W_o.
        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out_h, out_w, NAME);
        const Spatial::ColumnSpan column = Spatial::column_span_for(cols, span, NAME);

        void* read_in = in.data_or_null();
        void* write_cols = cols.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_cols) writes.push_back(cols.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.im2col"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_cols);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::im2col<T>(span, column, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::im2col<T>(q2, span, column, source, destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
```

**Note on the `out_w` derivation.** Deriving the output extents from the column matrix's row count means symmetric padding is assumed for the width. That is wrong for YOLOv3-tiny's asymmetric pool — but `im2col` is never the operation that pools, and every convolution in both models in scope pads symmetrically. Rather than let that be a silent assumption, **add an explicit overload** in the same file taking `out_h` and `out_w`:

```cpp
    sycl::event SpatialOps::im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g,
                                   int64_t out_h, int64_t out_w);
```

and make the two-operand form call it with the derived numbers. Declare both in `spatial.hpp`, document that the explicit form is the one a caller with asymmetric padding must use, and cover it in the test file with one case where the derived and explicit answers differ. SushiAI will always call the explicit form, because its shape inference has already computed both extents.

- [ ] **Step 9: Wire the engine and the build**

In `include/SushiBLAS/engine.hpp`: add `#include <SushiBLAS/engine/spatial.hpp>` to the include block, and next to `reductions()`:

```cpp
            /**
             * @brief Windowed and strided movement over an N-D grid.
             * @return A SpatialOps object providing access to the spatial kernels.
             */
            inline SpatialOps spatial() { return SpatialOps(*this); }
```

In `src/CMakeLists.txt`, add to `SUSHIBLAS_SOURCES`, after the `engine/math/reductions/` group:

```cmake
    # --- Spatial ---
    engine/spatial/im2col.cpp
```

- [ ] **Step 10: Run the tests**

Run: `sb build` then `sb test --suite unit --filter "Unit_SpatialIm2Col*"`
Expected: all PASS.

Then `sb test --suite unit` — the whole tier, to prove the `engine.hpp` change broke nothing.

- [ ] **Step 11: Commit**

```bash
git add include/SushiBLAS/engine/spatial.hpp include/SushiBLAS/engine/spatial/window.hpp \
        include/SushiBLAS/engine.hpp kernels/spatial.hpp kernels/reference/spatial/im2col.hpp \
        kernels/sycl/spatial/im2col.hpp src/engine/spatial/im2col.cpp \
        src/CMakeLists.txt tests/CMakeLists.txt tests/functional/unit/test_spatial_im2col.cpp
git commit -m "feat(spatial): add the spatial domain and its im2col gather"
```

---

## Task 4: `col2im`

**Files:**
- Create: `kernels/reference/spatial/col2im.hpp`, `kernels/sycl/spatial/col2im.hpp`, `src/engine/spatial/col2im.cpp`
- Modify: `include/SushiBLAS/engine/spatial.hpp`, `kernels/spatial.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/functional/unit/test_spatial_col2im.cpp`

**Interfaces:**
- Consumes: Task 3's `window_span_for`, `column_span_for`, `SpatialOps`.
- Produces: `sycl::event SpatialOps::col2im(const Tensor& cols, Tensor& in_grad, const Spatial::ConvGeometry& g, int64_t out_h, int64_t out_w)`; the two kernels of the same name.

- [ ] **Step 1: Write the failing test**

Create `tests/functional/unit/test_spatial_col2im.cpp`. The two tests that matter:

```cpp
        TEST_F(Unit_SpatialCol2ImTest, EachPixelReceivesOncePerWindowThatReadIt)
        {
            // col2im of an all-ones column matrix is the *overlap count* of each
            // pixel: how many windows read it. That is a closed-form answer this
            // test computes independently, and it is the strongest cheap check
            // that the gather visits exactly the right set - a rule that drops
            // contributions or double-counts fails it, where a round-trip
            // against im2col would not necessarily.
            Spatial::ConvGeometry g;
            g.kernel_h = 3;
            g.kernel_w = 3;
            g.pad_top = 1;
            g.pad_left = 1;

            const int64_t H = 4, W = 5, C = 2, N = 2;
            const int64_t out_h = Spatial::output_extent(H, 1, 1, 1, 3, 1);
            const int64_t out_w = Spatial::output_extent(W, 1, 1, 1, 3, 1);

            auto cols = engine->create_tensor({N * out_h * out_w, C * 3 * 3},
                                              Core::DataType::FLOAT32);
            float* c = cols.data_as<float>();
            for (int64_t i = 0; i < cols.num_elements; ++i) c[i] = 1.0f;

            auto dx = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            engine->spatial().col2im(cols, dx, g, out_h, out_w);
            engine->execute();

            const float* got = dx.data_as<float>();
            for (int64_t n = 0; n < N; ++n)
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t w = 0; w < W; ++w)
                    {
                        // With a 3x3 stride-1 window padded by 1, a pixel is read
                        // by every window whose origin is within one step of it and
                        // inside the image: 3x3 in the interior, fewer at the edges.
                        // Counted by brute force rather than by a closed form, so
                        // this expectation cannot share a mistake with the kernel.
                        int64_t count = 0;
                        for (int64_t ho = 0; ho < out_h; ++ho)
                            for (int64_t wo = 0; wo < out_w; ++wo)
                                for (int64_t r = 0; r < 3; ++r)
                                    for (int64_t s = 0; s < 3; ++s)
                                        if (ho - 1 + r == h && wo - 1 + s == w) ++count;

                        for (int64_t ch = 0; ch < C; ++ch)
                            EXPECT_FLOAT_EQ(got[((n * H + h) * W + w) * C + ch],
                                            static_cast<float>(count))
                                << "pixel (" << n << "," << h << "," << w << "," << ch << ")";
                    }
        }

        TEST_F(Unit_SpatialCol2ImTest, StrideAndDilationSelectTheRightContributors)
        {
            // A stride-2 window means most pixels are read by exactly one window
            // and some by none. Getting the divisibility test wrong - dropping
            // the `% stride != 0` guard - turns this into a uniform count and is
            // otherwise invisible.
            Spatial::ConvGeometry g;
            g.kernel_h = 2;
            g.kernel_w = 2;
            g.stride_h = 2;
            g.stride_w = 2;

            const int64_t H = 4, W = 4, C = 1;
            const int64_t out_h = Spatial::output_extent(H, 0, 0, 1, 2, 2);   // 2
            const int64_t out_w = Spatial::output_extent(W, 0, 0, 1, 2, 2);   // 2

            auto cols = engine->create_tensor({out_h * out_w, C * 4}, Core::DataType::FLOAT32);
            float* c = cols.data_as<float>();
            for (int64_t i = 0; i < cols.num_elements; ++i) c[i] = 1.0f;

            auto dx = engine->create_tensor({1, H, W, C}, Core::DataType::FLOAT32);
            engine->spatial().col2im(cols, dx, g, out_h, out_w);
            engine->execute();

            // Non-overlapping 2x2 tiles: every pixel is read exactly once.
            const float* got = dx.data_as<float>();
            for (int64_t i = 0; i < dx.num_elements; ++i)
                EXPECT_FLOAT_EQ(got[i], 1.0f) << "element " << i;
        }
```

Also write, in the same file: a `TheSyclKernelAgreesWithTheReferenceKernel` VERIFY case mirroring Task 3's, and a test that `col2im` writes every element of its destination (fill `dx` with a sentinel first and assert no sentinel survives) — the gather form writes each pixel exactly once, so a pixel no window reads must be written **zero**, not left alone.

- [ ] **Step 2: Run it to verify it fails**

Run: `sb build`
Expected: FAIL — `no member named 'col2im' in 'SushiBLAS::SpatialOps'`.

- [ ] **Step 3: Write the reference kernel**

`kernels/reference/spatial/col2im.hpp`, the inverse addressing of Task 3's — **a gather, one output pixel per iteration**:

```cpp
            template<typename T>
            void col2im(const Spatial::WindowSpan& span, const Spatial::ColumnSpan& cols,
                        const T* in, T* out)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                    for (int64_t h = 0; h < span.in_h; ++h)
                        for (int64_t w = 0; w < span.in_w; ++w)
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                T total = T(0);

                                // Ascending r, then ascending s. Which terms are
                                // summed and the order they are summed in are
                                // pure functions of the geometry, so two runs
                                // perform the same additions on the same values
                                // in the same order. That is the whole of CV-6.
                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_num = h + g.pad_top - r * g.dilation_h;
                                    if (h_num < 0 || h_num % g.stride_h != 0) continue;
                                    const int64_t h_o = h_num / g.stride_h;
                                    if (h_o >= span.out_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                        if (w_num < 0 || w_num % g.stride_w != 0) continue;
                                        const int64_t w_o = w_num / g.stride_w;
                                        if (w_o >= span.out_w) continue;

                                        const int64_t row = (n * span.out_h + h_o) * span.out_w + w_o;
                                        const int64_t patch = (r * g.kernel_w + s) * span.channels + c;

                                        total += in[row * cols.row_stride + patch * cols.element_stride];
                                    }
                                }

                                out[n * span.in_batch_stride + h * span.in_row_stride
                                    + w * span.in_col_stride + c * span.in_channel_stride] = total;
                            }
            }
```

- [ ] **Step 4: Write the SYCL kernel**

`kernels/sycl/spatial/col2im.hpp` — one work-item per **input** element (`span.input_elements()` of them), the same body as Step 3's innermost block, decomposing the flat id as `c` innermost then `w`, `h`, `n`. Copy every `span` field out by value before the closure, exactly as Task 3's kernel does. **A `sycl::atomic_ref` must not appear in this file**; the whole point of the gather form is that it does not need one.

- [ ] **Step 5: Write the operation and wire it**

`src/engine/spatial/col2im.cpp`, mirroring Task 3's structure: dtype check, `window_span_for(in_grad, g, out_h, out_w, "spatial.col2im")` — the extents overload, since `in_grad` plays the *input* role in the span and the other operand is a column matrix — then `column_span_for(cols, span, …)`, reads `{cols}`, writes `{in_grad}`, `meta.op_id = "spatial.col2im"_op`, `Domain::REAL`.

Add the declaration to `spatial.hpp`, the two includes to `kernels/spatial.hpp`, and `engine/spatial/col2im.cpp` to `SUSHIBLAS_SOURCES`.

- [ ] **Step 6: Run the tests**

Run: `sb build` then `sb test --suite unit --filter "Unit_SpatialCol2Im*"`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add kernels/reference/spatial/col2im.hpp kernels/sycl/spatial/col2im.hpp \
        src/engine/spatial/col2im.cpp include/SushiBLAS/engine/spatial.hpp \
        kernels/spatial.hpp src/CMakeLists.txt tests/CMakeLists.txt \
        tests/functional/unit/test_spatial_col2im.cpp
git commit -m "feat(spatial): add col2im as a fixed-order gather"
```

---

## Task 5: `pool_max` and `pool_max_backward`

**Files:**
- Create: `kernels/reference/spatial/pool_max.hpp`, `kernels/sycl/spatial/pool_max.hpp`, `src/engine/spatial/pool_max.cpp`, `src/engine/spatial/pool_max_backward.cpp`
- Modify: `include/SushiBLAS/engine/spatial.hpp`, `kernels/spatial.hpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/functional/unit/test_spatial_pool_max.cpp`

**Interfaces:**
- Produces:
  - `sycl::event SpatialOps::pool_max(const Tensor& in, Tensor& out, Tensor& indices, const Spatial::ConvGeometry& g)`
  - `sycl::event SpatialOps::pool_max_backward(const Tensor& dy, const Tensor& indices, Tensor& dx, const Spatial::ConvGeometry& g)`
  - `template<typename T> void Kernels::Reference::pool_max(const Spatial::WindowSpan&, const T* in, T* out, int64_t* indices)` and its SYCL twin
  - `template<typename T> void Kernels::Reference::pool_max_backward(const Spatial::WindowSpan&, const T* dy, const int64_t* indices, T* dx)` and its SYCL twin

`indices` is **INT64**, for the reason `execute_index_reduction` exists: an index is not a number, and is exact in fp32 only below 2²⁴. A 224×224×64 feature map passes that at batch 6.

- [ ] **Step 1: Write the failing tests**

The cases that must be present:

1. **`ThePoolTakesTheWindowMaximum`** — a hand-built 1×4×4×1 input with known values, 2×2 stride 2, expected `[2,2]` output written out literally.
2. **`TheIndexNamesTheElementTheMaximumCameFrom`** — assert `indices[k] == ((n*H + h)*W + w)*C + c` for the known argmax of each window, using the **logical** flat index (D-A5).
3. **`ATieGoesToTheFirstElementInScanOrder`** — an all-equal window; assert the index is the window's first tap. Deterministic tie-breaking is what makes the backward gather's equality test single-valued.
4. **`PaddingNeverWinsAWindow`** — a window whose only in-bounds elements are negative, with padding. The implicit zeros must **not** be candidates: a max-pool over padding is `-inf`-initialised, not zero-initialised, or every negative feature map becomes zero at its border and nothing throws. Assert the output equals the largest *real* element.
5. **`TheBackwardRoutesEachGradientToItsArgmaxAndNowhereElse`** — non-overlapping windows: `dx` must hold `dy` at the argmax positions and exactly `0` everywhere else.
6. **`OverlappingWindowsAccumulateAtASharedArgmax`** — 2×2 stride 1 (YOLOv3-tiny's pool), a single dominant pixel selected by four windows: assert `dx` at that pixel is the sum of the four `dy` values.
7. **`TheSyclKernelAgreesWithTheReferenceKernel`** — VERIFY over a padded, strided, non-square case.

Write test 4 first and run it before writing any kernel; a zero-initialised max is the single likeliest defect here and it should be the failure you see.

- [ ] **Step 2: Run to verify failure**

Run: `sb build` — FAIL, `no member named 'pool_max'`.

- [ ] **Step 3: Write the forward kernels**

Reference, one output element per iteration:

```cpp
            template<typename T>
            void pool_max(const Spatial::WindowSpan& span, const T* in, T* out, int64_t* indices)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                    for (int64_t h_o = 0; h_o < span.out_h; ++h_o)
                        for (int64_t w_o = 0; w_o < span.out_w; ++w_o)
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                // Lowest, not zero. Padding is absence, not a
                                // zero-valued candidate - a zero-initialised
                                // maximum turns every all-negative border window
                                // into 0 and reports no error at all.
                                T best = std::numeric_limits<T>::lowest();
                                int64_t best_index = -1;

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_in = h_o * g.stride_h - g.pad_top + r * g.dilation_h;
                                    if (h_in < 0 || h_in >= span.in_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_in = w_o * g.stride_w - g.pad_left + s * g.dilation_w;
                                        if (w_in < 0 || w_in >= span.in_w) continue;

                                        const T candidate = in[n * span.in_batch_stride
                                                             + h_in * span.in_row_stride
                                                             + w_in * span.in_col_stride
                                                             + c * span.in_channel_stride];

                                        // Strictly greater: a tie keeps the first
                                        // element in scan order, so the recorded
                                        // index is single-valued and the backward
                                        // gather's equality test cannot match twice.
                                        if (candidate > best)
                                        {
                                            best = candidate;
                                            best_index = ((n * span.in_h + h_in) * span.in_w + w_in)
                                                       * span.channels + c;
                                        }
                                    }
                                }

                                const int64_t at = n * span.out_batch_stride
                                                 + h_o * span.out_row_stride
                                                 + w_o * span.out_col_stride
                                                 + c * span.out_channel_stride;
                                out[at] = best;
                                indices[at] = best_index;
                            }
            }
```

`std::numeric_limits<T>::lowest()` is not available in device code for `sycl::half` in every configuration — check what the existing reductions use for `RunningMax`'s identity (`include/SushiBLAS/engine/reduce_1d.hpp`) and use the same construct in the SYCL twin.

SYCL twin: one work-item per output element (`span.output_elements()`), same body, flat id decomposed `c`, `w_o`, `h_o`, `n`.

- [ ] **Step 4: Write the backward kernels**

**A gather over the windows that could have selected each input pixel** — the same divisibility walk `col2im` uses, with an equality test on the recorded index:

```cpp
            template<typename T>
            void pool_max_backward(const Spatial::WindowSpan& span, const T* dy,
                                   const int64_t* indices, T* dx)
            {
                const Spatial::ConvGeometry& g = span.geometry;

                for (int64_t n = 0; n < span.batch; ++n)
                    for (int64_t h = 0; h < span.in_h; ++h)
                        for (int64_t w = 0; w < span.in_w; ++w)
                            for (int64_t c = 0; c < span.channels; ++c)
                            {
                                const int64_t mine =
                                    ((n * span.in_h + h) * span.in_w + w) * span.channels + c;

                                T total = T(0);

                                for (int64_t r = 0; r < g.kernel_h; ++r)
                                {
                                    const int64_t h_num = h + g.pad_top - r * g.dilation_h;
                                    if (h_num < 0 || h_num % g.stride_h != 0) continue;
                                    const int64_t h_o = h_num / g.stride_h;
                                    if (h_o >= span.out_h) continue;

                                    for (int64_t s = 0; s < g.kernel_w; ++s)
                                    {
                                        const int64_t w_num = w + g.pad_left - s * g.dilation_w;
                                        if (w_num < 0 || w_num % g.stride_w != 0) continue;
                                        const int64_t w_o = w_num / g.stride_w;
                                        if (w_o >= span.out_w) continue;

                                        const int64_t at = n * span.out_batch_stride
                                                         + h_o * span.out_row_stride
                                                         + w_o * span.out_col_stride
                                                         + c * span.out_channel_stride;

                                        if (indices[at] == mine) total += dy[at];
                                    }
                                }

                                dx[n * span.in_batch_stride + h * span.in_row_stride
                                   + w * span.in_col_stride + c * span.in_channel_stride] = total;
                            }
            }
```

- [ ] **Step 5: Write the two operations and wire them**

`src/engine/spatial/pool_max.cpp` records `"spatial.pool_max"_op` with reads `{in}` and writes `{out, indices}`; `pool_max_backward.cpp` records `"spatial.pool_max_backward"_op` with reads `{dy, indices}` and writes `{dx}`.

Both must refuse an `indices` tensor whose dtype is not `INT64`, by name, at record time. The dtype passed to `execute_typed_task` is the **value** tensor's, not the index tensor's — the index pointer is captured as `int64_t*` outside the dtype switch.

- [ ] **Step 6: Run and commit**

Run: `sb build` then `sb test --suite unit --filter "Unit_SpatialPoolMax*"` — all PASS.

```bash
git commit -m "feat(spatial): add max pooling and its gather-form backward"
```

---

## Task 6: `pool_avg` and `pool_avg_backward`

**Files:** as Task 5, with `pool_avg` in the names.

**Interfaces:**
- Produces: `sycl::event SpatialOps::pool_avg(const Tensor& in, Tensor& out, const Spatial::ConvGeometry& g)` and `sycl::event SpatialOps::pool_avg_backward(const Tensor& dy, Tensor& dx, const Spatial::ConvGeometry& g)`, plus the four kernels.

Forward: the same window walk as `pool_max`, summing in ascending `(r, s)` order, then dividing by `kernel_h * kernel_w` — **the window area, including padded taps** (D-A4). Record that in the Doxygen with the words "count_include_pad", so a reader looking for the PyTorch flag finds the answer.

Backward: the same divisibility gather as `pool_max_backward`, adding `dy[at] / (kernel_h * kernel_w)` for every window that covers the pixel, with no index test — the divisor is a constant, which is why this is a separate function from `pool_max_backward` and not one `pool_backward` behind a mode flag. Their mathematics shares nothing: one selects through a recorded index, the other broadcasts a constant.

**Tests** (`tests/functional/unit/test_spatial_pool_avg.cpp`):

1. `TheAverageIsOverTheWholeWindowIncludingPaddedTaps` — a padded corner window with one real element `v`: assert the output is `v / (R*S)`, not `v`. This is D-A4's pin; without it the convention is invisible.
2. `AnUnpaddedPoolIsThePlainMean` — hand-computed 2×2 stride 2.
3. `TheBackwardSpreadsEachGradientOverItsWholeWindow` — non-overlapping: every pixel gets `dy/(R*S)`.
4. `OverlappingWindowsSumTheirShares` — 2×2 stride 1: an interior pixel covered by four windows gets the sum of four shares.
5. `TheBackwardOfAConstantForwardIsUniform` — a stride-1 padded pool over an all-ones `dy`: the interior must be exactly `1.0`, which is the closed-form check that the divisor and the window set agree.
6. `TheSyclKernelAgreesWithTheReferenceKernel` — VERIFY.

Commit: `feat(spatial): add average pooling and its backward`

---

## Task 7: `upsample_nearest` and its backward

**Interfaces:**
- Produces: `sycl::event SpatialOps::upsample_nearest(const Tensor& in, Tensor& out, int64_t scale_h, int64_t scale_w)` and `sycl::event SpatialOps::upsample_nearest_backward(const Tensor& dy, Tensor& dx, int64_t scale_h, int64_t scale_w)`, plus four kernels.

This operation carries no `ConvGeometry` — it has two numbers, not eight, and giving it a geometry it does not use would be eight fields a reader has to rule out. It builds a `WindowSpan` only for the strides, with `geometry` left default.

Forward, one work-item per output element: `out[n, h_o, w_o, c] = in[n, h_o / scale_h, w_o / scale_w, c]`.

Backward, one work-item per **input** element, summing the fixed `scale_h * scale_w` outputs that read it, in ascending order:

```cpp
                                T total = T(0);
                                for (int64_t dh = 0; dh < scale_h; ++dh)
                                    for (int64_t dw = 0; dw < scale_w; ++dw)
                                        total += dy[n * span.out_batch_stride
                                                  + (h * scale_h + dh) * span.out_row_stride
                                                  + (w * scale_w + dw) * span.out_col_stride
                                                  + c * span.out_channel_stride];
```

**Tests** (`tests/functional/unit/test_spatial_upsample.cpp`):

1. `EachInputPixelBecomesAScaleByScaleBlock` — a 1×2×2×1 ramp at scale 2, expected 4×4 written out literally.
2. `AnisotropicScalesAreIndependent` — `scale_h = 2, scale_w = 3`.
3. `AUnitScaleIsACopy` — scale 1 in both axes reproduces the input exactly.
4. `TheBackwardSumsTheBlockThatReadEachInput` — `dy` a ramp, `dx[i]` asserted equal to the sum of its block.
5. `AScaleOfZeroOrNegativeIsRefused` — by name, at record time.
6. `TheSyclKernelAgreesWithTheReferenceKernel` — VERIFY.

Commit: `feat(spatial): add nearest-neighbour upsampling and its backward`

---

## Task 8: `concat_channels` and `slice_channels`

**Interfaces:**
- Produces:
  - `sycl::event SpatialOps::concat_channels(const Tensor& src, Tensor& dst, int64_t dst_channel_offset)`
  - `sycl::event SpatialOps::slice_channels(const Tensor& src, int64_t src_channel_offset, Tensor& dst)`
  - one kernel pair, `copy_channel_range`, serving both (D-A6)

The kernel copies `count` channels from `src` starting at `src_channel_offset` into `dst` starting at `dst_channel_offset`, over a shared `[N, H, W]` grid. `count` is the *smaller* channel count and is derived, not passed: `concat_channels` takes all of `src`'s channels, `slice_channels` fills all of `dst`'s. Both refuse, by name, a range that runs past either tensor's channel count.

**Why not a zero-copy `Tensor::slice()` view:** the gradient a slice produces has to be a materialised operand for the next node, and a graph whose values are sometimes views and sometimes buffers is a distinction the memory planner would have to carry everywhere. Put that sentence in the Doxygen — it is the question every reader will ask.

**Record the known conservatism in the file's own comment**, from spec §4: in NHWC a channel range is strided, not a contiguous byte interval, so two `concat_channels` writes into one destination produce overlapping `Tensor::region()`s and SushiRuntime's dependency tracker serialises them. That is correct and possibly slower than necessary. Measure before addressing.

**Tests** (`tests/functional/unit/test_spatial_channels.cpp`):

1. `ConcatPlacesEachSourceAtItsOffset` — two sources of 2 and 3 channels into a 5-channel destination; assert every element, including that neither wrote outside its range (pre-fill the destination with a sentinel).
2. `SliceIsConcatsInverse` — concat two tensors, slice them back out, compare bit-for-bit against the originals.
3. `TheyAreEachOthersBackward` — concat A and B into C, then slice C at A's offset for A's width: the result is A. This is the property the autograd rule will rely on and it is worth naming.
4. `ARangeThatRunsPastTheChannelCountIsRefused` — by name, at record time, both directions.
5. `AStridedSourceIsAddressedThroughItsStrides` — build the source as a view with a non-unit channel stride and assert the copy is still correct. This is what proves the operation reads strides rather than assuming contiguity.
6. `TheSyclKernelAgreesWithTheReferenceKernel` — VERIFY.

Commit: `feat(spatial): add channel concatenation and slicing`

---

## Task 9: `channel_affine`, `broadcast_spatial` and `layout_convert`

Three broadcasts, grouped because each is a few lines and none is independently interesting. All three write one output element per work-item and read one input element plus at most one per-channel scalar — no reduction, no accumulation, nothing whose order could vary.

**Interfaces:**
- `sycl::event SpatialOps::channel_affine(const Tensor& in, const Tensor& gamma, const Tensor& beta, Tensor& out)`
  `out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. BatchNorm's apply half, and the *whole* of BatchNorm in evaluation mode. `gamma` and `beta` are rank-1 `[C]` and are addressed through their own strides.
- `sycl::event SpatialOps::broadcast_spatial(const Tensor& in, Tensor& out, double scale)` (D-A1)
  `out[n,h,w,c] = in[n,c] * scale`. Global average pooling's backward: an `[N,C]` gradient spread over `[N,H,W,C]` scaled by `1/(H*W)`. A **per-(n,c)** broadcast — `BIAS_ADD` broadcasts a row vector and `channel_affine` broadcasts per channel, and neither is this.
- `sycl::event SpatialOps::layout_convert(const Tensor& in, Tensor& out)` (D-A1)
  A rank-4 permuting copy. Its direction is not a flag: it reads both tensors' strides and copies element by logical position, so NCHW→NHWC and NHWC→NCHW are the same call with the operands' strides swapped. Two O(N·C·H·W) passes per step, immaterial beside the convolutions, and exactly what PyTorch's `channels_last` does. It buys **interoperability** — NCHW data and NCHW checkpoints in — and is not an NCHW implementation.

**Tests** (`tests/functional/unit/test_spatial_broadcast.cpp`):

1. `ChannelAffineAppliesThePerChannelPair` — hand-computed over `[1,2,2,3]`.
2. `ChannelAffineWithUnitGammaAndZeroBetaIsACopy`.
3. `BroadcastSpatialSpreadsAndScales` — `[2,3]` over `[2,4,5,3]`, every element asserted.
4. `BroadcastSpatialAtOneOverHwIsTheMeanBackward` — feed an all-ones `[N,C]` at `scale = 1/(H*W)` and assert every output element is exactly `1/(H*W)`; then assert that summing each `(n,c)` plane gives `1.0` to within one ulp of the accumulated error. This is the property `SPATIAL_MEAN_BACKWARD` needs and gradcheck will otherwise find much later.
5. `LayoutConvertRoundTripsToItself` — NHWC → NCHW → NHWC, bit-for-bit.
6. `LayoutConvertMovesTheChannelIndex` — hand-computed on a small ramp, so a round-trip that permuted nothing at all cannot pass.
7. `TheSyclKernelsAgreeWithTheReferenceKernels` — VERIFY over all three.

Commit: `feat(spatial): add the channel affine and the two broadcast copies`

---

## Task 10: `AxisSpan`'s second segment axis, and `mean_spatial`

**Files:**
- Modify: `include/SushiBLAS/engine/reduce_axis.hpp`, `include/SushiBLAS/engine/math/reductions/dispatch.hpp`, `include/SushiBLAS/engine/math/reductions.hpp`
- Create: `src/engine/math/reductions/mean_spatial.cpp`, `tests/functional/unit/test_mean_spatial.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `Reduce::AxisSpan::segments2`, `Reduce::AxisSpan::segment_stride2`; `sycl::event ReductionOps::mean_spatial(const Tensor& t, Tensor& result)`.

**Why a second *segment* axis and not a second *folded* axis** — the distinction decides which loop changes, and it is worth restating in the header:

- BatchNorm over NHWC needs **nothing new**: `[N·H·W, C]` is a zero-copy rank-2 view and folding axis 0 gives `[C]` through the existing five numbers.
- Global average pooling **does** need it: `[N,H,W,C] → [N,C]` has segments indexed by the *pair* `(n, c)`, whose offset is `n·HWC + c`, and no single `segment_stride` expresses that.

- [ ] **Step 1: Write the failing test**

`tests/functional/unit/test_mean_spatial.cpp`:

```cpp
        TEST_F(Unit_MeanSpatialTest, FoldsTheTwoMiddleAxesOfAnNhwcTensor)
        {
            // [2,3,4,5] -> [2,5]. Each (n,c) result is the mean of the 12 pixels
            // of that image and channel, computed here independently.
            const int64_t N = 2, H = 3, W = 4, C = 5;
            auto in = engine->create_tensor({N, H, W, C}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i)
                p[i] = static_cast<float>(i % 17) - 8.0f;

            auto out = engine->create_tensor({N, C}, Core::DataType::FLOAT32);
            engine->reductions().mean_spatial(in, out);
            engine->execute();

            const float* got = out.data_as<float>();
            for (int64_t n = 0; n < N; ++n)
                for (int64_t c = 0; c < C; ++c)
                {
                    double expected = 0.0;
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t w = 0; w < W; ++w)
                            expected += p[((n * H + h) * W + w) * C + c];
                    expected /= static_cast<double>(H * W);

                    EXPECT_NEAR(got[n * C + c], static_cast<float>(expected), 1e-5f)
                        << "(" << n << ", " << c << ")";
                }
        }

        TEST_F(Unit_MeanSpatialTest, ASingleSegmentPairIsStillCorrect)
        {
            // N = 1 and C = 1 collapses segments2 and segments both to one,
            // which is the shape a "generalisation" most easily breaks.
            auto in = engine->create_tensor({1, 2, 2, 1}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;

            auto out = engine->create_tensor({1, 1}, Core::DataType::FLOAT32);
            engine->reductions().mean_spatial(in, out);
            engine->execute();

            EXPECT_FLOAT_EQ(out.data_as<float>()[0], 2.5f);
        }
```

Add, in the **same file**, a regression-tier guard that the existing five-number callers are untouched. It needs its own fixture — `class Regression_AxisSpanTest : public SushiBLASTest {};` — because the CTest label comes from the *suite* name and a `Regression_` case declared against a `Unit_` fixture would carry the wrong one:

```cpp
        TEST_F(Regression_AxisSpanTest, ASpanWithNoSecondAxisReducesExactlyAsItAlwaysDid)
        {
            // Every existing caller aggregate-initialises AxisSpan with five
            // values and now gets segments2 = 1, segment_stride2 = 0 by default.
            // If the new outer loop is wrong at segments2 == 1, every row-wise
            // reduction in the library is wrong and this is the cheapest place
            // to see it.
            const int64_t rows = 6, cols = 7;
            auto in = engine->create_tensor({rows, cols}, Core::DataType::FLOAT32);
            float* p = in.data_as<float>();
            for (int64_t i = 0; i < in.num_elements; ++i) p[i] = static_cast<float>(i);

            auto out = engine->create_tensor({rows}, Core::DataType::FLOAT32);
            engine->reductions().sum_along(in, 1, out);
            engine->execute();

            const float* got = out.data_as<float>();
            for (int64_t r = 0; r < rows; ++r)
            {
                float expected = 0.0f;
                for (int64_t c = 0; c < cols; ++c) expected += p[r * cols + c];
                EXPECT_FLOAT_EQ(got[r], expected) << "row " << r;
            }
        }
```

- [ ] **Step 2: Run to verify failure**

Run: `sb build` — FAIL, `no member named 'mean_spatial'`.

- [ ] **Step 3: Extend `AxisSpan`**

In `include/SushiBLAS/engine/reduce_axis.hpp`, add two fields **at the end** of `AxisSpan` (so every existing five-value aggregate initialisation still compiles and means what it meant):

```cpp
            /**
             * @brief How many *groups* of segments; 1 when there is only one axis
             * of segmentation.
             *
             * The second **segment** axis, not a second folded one - and which
             * of the two it is decides which loop changes. Folding two axes at
             * once needs nothing new: `[N*H*W, C]` is a zero-copy rank-2 view and
             * the existing five numbers fold axis 0 to `[C]`. What the five
             * numbers cannot express is a segment identified by a *pair*:
             * `[N,H,W,C] -> [N,C]` has one result per `(n, c)`, whose input
             * offset is `n * H*W*C + c`, and no single `segment_stride` produces
             * that from a single index.
             *
             * Segment `(s2, s)` starts at `in[s2 * segment_stride2 + s * segment_stride]`
             * and its result goes to `out[(s2 * segments + s) * out_stride]`.
             */
            std::size_t segments2 = 1;

            /** @brief Input elements between the starts of consecutive segment groups. */
            std::size_t segment_stride2 = 0;

            /** @brief How many folds in total: `segments * segments2`. */
            [[nodiscard]] std::size_t total_segments() const noexcept
            {
                return segments * (segments2 == 0 ? 0 : segments2);
            }
```

In `reduce_axis()`, replace the two uses of `segments` that name the *launch* with `span.total_segments()`, and replace both `const T* base = in + s * segment_stride;` with the pair decomposition:

```cpp
                            // One integer division per fold, against an inner
                            // loop of `extent` elements. Not worth a specialised
                            // segments2 == 1 kernel, which would be a second
                            // instantiation of every reduction in the library.
                            const std::size_t s2 = s / segments;
                            const std::size_t s1 = s % segments;
                            const T* base = in + s2 * segment_stride2 + s1 * segment_stride;
```

where `s` is now the flat fold index (`it.get_global_id(0)` grid-strided in the item-per-segment shape, `it.get_group(0)` in the cooperative one) and the output index becomes `s * out_stride` — which is already what both shapes write, because the flat index and `(s2 * segments + s1)` are the same number.

**Determinism is unchanged and must be stated as such in the header's determinism paragraph:** the fold order within a segment did not move, and no cross-segment combination was introduced.

- [ ] **Step 4: Extend `plan_axis`'s caller**

Every `plan_axis(profile, span.segments, span.extent)` in `engine/math/reductions/dispatch.hpp` becomes `plan_axis(profile, span.total_segments(), span.extent)`. Grep for it — there is more than one call site.

- [ ] **Step 5: Add the span builder and the operation**

In `engine/math/reductions/dispatch.hpp`, next to `axis_span_for`, add:

```cpp
        /**
         * @brief The span that folds an NHWC tensor's two middle axes.
         *
         * `axis_span_for`'s documented rank-2 restriction is lifted only this
         * far: this is the one shape a single `segment_stride` cannot describe,
         * and generalising the operand validation past it would be inventing a
         * requirement nothing has.
         *
         * @param t The rank-4 `[N, H, W, C]` input.
         * @param result The rank-2 `[N, C]` output.
         * @param name The operation's name, for diagnostics.
         * @return The span, in element counts.
         * @throws std::runtime_error when the ranks or the N/C extents disagree.
         */
        [[nodiscard]] inline Reduce::AxisSpan spatial_span_for(const Tensor& t, const Tensor& result,
                                                               const char* name)
        {
            SB_THROW_IF(t.rank != 4, "Reduction '{}' needs a rank-4 NHWC input; got rank {}.",
                        name, t.rank);
            SB_THROW_IF(result.rank != 2, "Reduction '{}' needs a rank-2 [N, C] result; got rank {}.",
                        name, result.rank);
            SB_THROW_IF(result.shape[0] != t.shape[0] || result.shape[1] != t.shape[3],
                        "Reduction '{}' folds [{}, {}, {}, {}] to [{}, {}] but was given a [{}, {}] result.",
                        name, t.shape[0], t.shape[1], t.shape[2], t.shape[3],
                        t.shape[0], t.shape[3], result.shape[0], result.shape[1]);

            Reduce::AxisSpan span;
            span.segments  = static_cast<std::size_t>(t.shape[3]);              // c
            span.segments2 = static_cast<std::size_t>(t.shape[0]);              // n
            span.extent    = static_cast<std::size_t>(t.shape[1] * t.shape[2]); // h * w
            span.segment_stride  = static_cast<std::size_t>(t.strides[3]);
            span.segment_stride2 = static_cast<std::size_t>(t.strides[0]);
            span.element_stride  = static_cast<std::size_t>(t.strides[2]);
            span.out_stride      = static_cast<std::size_t>(result.strides[1]);
            return span;
        }
```

**A constraint this makes explicit:** the fold walks `h` and `w` as one run of `extent` elements at `element_stride`, which is exact only when `strides[1] == strides[2] * in_w` — true for any contiguous NHWC tensor and false for a row-sliced view. Assert it:

```cpp
            SB_THROW_IF(t.strides[1] != t.strides[2] * t.shape[2],
                        "Reduction '{}' folds H and W as one run, which needs the row stride ({}) "
                        "to be the column stride ({}) times the width ({}). This tensor is a "
                        "row-sliced view; materialise it first.",
                        name, t.strides[1], t.strides[2], t.shape[2]);
```

Then `src/engine/math/reductions/mean_spatial.cpp`, following `mean.cpp`'s shape exactly — `Reduce::CompensatedSum`, identity transform, `value / n` finalizer, `"math.mean_spatial"_op`.

Declare `mean_spatial` in `engine/math/reductions.hpp` with full Doxygen.

- [ ] **Step 6: Run the whole suite**

Run: `sb build` then `sb test --suite all`.
Expected: everything that passed before still passes. This task changed a kernel every reduction in the library goes through; the unit tier alone is not enough of a gate for it.

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(reductions): add a second segment axis and the spatial mean"
```

---

## Task 11: `ChannelBiasEpilogue`

**Files:**
- Create: `include/SushiBLAS/kernels/blas/level3/channel_bias_epilogue.hpp`
- Test: `tests/functional/unit/test_channel_bias_epilogue.cpp`
- Modify: `tests/CMakeLists.txt`

It lives under `include/`, not `kernels/`, because `Dispatch::execute_gemm` instantiates an epilogue in the **consumer's** translation unit — the rule `kernels/blas.hpp:24-34` states and `gemm_epilogue.hpp` follows.

**Interfaces:**
- Produces: `struct Kernels::ChannelBiasEpilogue` with `const void* bias`, `int64_t row_stride`, `int64_t col_stride`, and `template<typename T> T operator()(T value, int64_t row, int64_t col) const noexcept`.

**Why it is stride-parametric rather than a "bias per column" epilogue:** in NHWC the convolution GEMM is `[N·H·W, C_in·R·S] × [C_in·R·S, C_out]`, so the channel index is the **column** and the existing `GEMM_BIAS` family applies to convolution verbatim. In NCHW the channel would be the **row**. Indexing `bias[row * row_stride + col * col_stride]` with host-set strides means adding NCHW later is filling in `(1, 0)` instead of `(0, 1)`, not writing a second epilogue — which is exactly what spec CV-3 means by "the seam is still built layout-parametric".

```cpp
        /**
         * @struct ChannelBiasEpilogue
         * @brief Adds a per-channel bias to each GEMM output element, in either layout.
         *
         * `value + bias[row * row_stride + col * col_stride]`, with the stride
         * pair set on the host. `(0, 1)` makes the **column** the channel, which
         * is what an NHWC convolution GEMM needs and what
         * `Ops::GEMM_BIAS`'s existing row-vector broadcast already does; `(1, 0)`
         * makes the **row** the channel, which is what an NCHW one would need.
         * Neither is baked in, so a second layout costs two numbers rather than
         * a second epilogue and a second fused lowering recorder.
         *
         * @warning The pointer is captured into a SYCL kernel. It must be
         *          device-accessible, must outlive the submission, and the
         *          access set that reads it has to be declared on the task -
         *          `Dispatch::gemm_epilogue(f).reads(bias_tensor)` is how, and
         *          `execute_gemm` cannot discover it (see its documentation).
         */
        struct ChannelBiasEpilogue
        {
            /** @brief The bias vector's first element, as the GEMM's own element type. */
            const void* bias = nullptr;
            /** @brief Bias elements to advance per output row; 0 when the channel is the column. */
            int64_t row_stride = 0;
            /** @brief Bias elements to advance per output column; 0 when the channel is the row. */
            int64_t col_stride = 1;

            /**
             * @brief Returns @p value plus the bias for its position.
             * @tparam T The element type of C.
             * @param value The fully formed C element.
             * @param row The element's row in C.
             * @param col The element's column in C.
             * @return @p value plus the corresponding bias element.
             */
            template<typename T>
            [[nodiscard]] T operator()(T value, int64_t row, int64_t col) const noexcept
            {
                return value + static_cast<const T*>(bias)[row * row_stride + col * col_stride];
            }
        };
```

**Tests:**

1. `AColumnChannelBiasMatchesAPlainRowVectorBroadcast` — run `execute_gemm` with `(0, 1)` and compare element-for-element against a hand-computed `A*B + bias[col]`.
2. `ARowChannelBiasBroadcastsDownTheOtherAxis` — `(1, 0)` against `A*B + bias[row]`. Both directions tested, because a stride pair that only ever ran one way is a parameter nobody proved.
3. `AZeroBiasIsTheIdentityEpilogue` — bit-for-bit against `GemmIdentityEpilogue`, which is the check that the fusion hook costs nothing when it does nothing.
4. `TheEpiloguesAccessSetReachesTheTask` — declare the bias through `Dispatch::gemm_epilogue(f).reads(bias)`, write the bias from a prior recorded task, and assert the GEMM sees the written values. Without the declared read the dependency edge does not exist and this is a race that passes by luck; check how `test_gemm.cpp` exercises `gemm_epilogue` today and follow it.

Commit: `feat(blas): add a layout-parametric channel bias epilogue`

---

## Task 12: Determinism, the regression gate

**Files:**
- Create: `tests/functional/regression/test_spatial_determinism.cpp`
- Modify: `tests/CMakeLists.txt`

CV-6's guarantee has a **silent** failure mode: an atomic add gives an answer that is correct to every tolerance any other test uses, and differs in the last bits between runs. Nothing else in this plan catches it, which is why it is in the regression tier.

- [ ] **Step 1: Write the test**

```cpp
        // Every backward in the spatial domain is written as a gather so that
        // each output element sums its own contributions in a fixed order (spec
        // CV-6). The failure mode this guards is silent: `sycl::atomic_ref`'s
        // floating-point add is correct to any tolerance and non-reproducible in
        // the last bits, so a tolerance-based test passes and the library's
        // bit-reproducibility contract - inherited from the PUBLIC link against
        // `sushiruntime_deterministic_fp` - is quietly gone.
        //
        // Comparison is on the raw bit patterns. EXPECT_FLOAT_EQ would accept
        // the very difference being looked for.
        template<typename Run>
        void expect_bit_identical_across_runs(Run&& run, int64_t elements)
        {
            std::vector<uint32_t> first(static_cast<std::size_t>(elements));
            std::vector<uint32_t> second(static_cast<std::size_t>(elements));

            run(first);
            run(second);

            for (std::size_t i = 0; i < first.size(); ++i)
                ASSERT_EQ(first[i], second[i])
                    << "element " << i << " differs between two runs of the same kernel";
        }
```

Cover, each with **overlapping windows** so there is something to accumulate — a non-overlapping geometry has one contribution per output and would pass with an atomic add in place:

1. `Regression_SpatialDeterminism.Col2ImIsBitReproducible` — 3×3 stride 1 padded, so each interior pixel gathers 9 terms.
2. `Regression_SpatialDeterminism.MaxPoolBackwardIsBitReproducible` — 2×2 stride 1.
3. `Regression_SpatialDeterminism.AvgPoolBackwardIsBitReproducible` — 3×3 stride 1.
4. `Regression_SpatialDeterminism.UpsampleBackwardIsBitReproducible` — scale 3, so each input gathers 9 terms.
5. `Regression_SpatialDeterminism.MeanSpatialIsBitReproducible` — Task 10's reduction, over a long enough extent to reach the cooperative shape.

Use input values whose sum is order-sensitive — a mix of large and small magnitudes, e.g. `{1e8, 1.0, -1e8, 1.0, …}` — so a reordering actually changes the result. A ramp of similar magnitudes can reorder without changing a bit and would make this test pass vacuously.

- [ ] **Step 2: Run, then prove the test can fail**

Run: `sb build` then `sb test --suite regression --filter "Regression_SpatialDeterminism.*"` — all PASS.

Then **temporarily** rewrite one backward's inner accumulation as a `sycl::atomic_ref` scatter, rebuild, and confirm the test fails. Revert. A determinism test nobody has seen fail is a test nobody knows works — and this one is guarding against code that does not exist yet, so its only chance to be validated is now.

- [ ] **Step 3: Commit**

```bash
git commit -m "test(spatial): pin the gather form's bit reproducibility"
```

---

## Task 13: Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`

- [ ] **Step 1: `ARCHITECTURE.md` §2, the layer diagram**

Add the `spatial` domain alongside `blas`, `logic`, `math/*`. Every class, CMake function and file the document names must still exist — check that the additions do.

- [ ] **Step 2: `ARCHITECTURE.md` §7, the kernels section**

Add a `spatial` subsection under "Notable kernel designs" carrying:
- what the domain is for, in the vocabulary of §3's table (gathers, windowed reductions, broadcasts — not "convolution");
- `WindowSpan` as the host-side resolution of layout and strides, and its kinship with `Reduce::AxisSpan`;
- **CV-6's determinism argument in full**: why every backward is a gather, what a floating-point atomic add would cost, and that `tests/functional/regression/test_spatial_determinism.cpp` is the guard. This is the same argument §7 already gives for rejecting `sycl::reduction`, and it should read as the same argument applied again rather than as a new one;
- `AxisSpan`'s second segment axis and the segment/fold distinction.

- [ ] **Step 3: `ARCHITECTURE.md` §6**

`engine/blas/utils.hpp` is named there. Update it for Task 2's move: `execute_typed_task` is in `engine/task.hpp` and `execute_blas_task` is the BLAS-named forwarder.

- [ ] **Step 4: `CHANGELOG.md`**

One `[Unreleased]` entry naming the domain, the operations, the `AxisSpan` extension and the epilogue.

- [ ] **Step 5: Commit**

```bash
git commit -m "docs: record the spatial domain and its determinism argument"
```

---

## Final gate

- [ ] `sb build --clean` — a clean configure, because this plan added directories and CMake source-list entries and a warm build tree can hide a missing one.
- [ ] `sb test --suite all` — every tier.
- [ ] `sb test` under `ExecutionMode::VERIFY` for the spatial tier specifically, if the CLI exposes a mode switch; otherwise confirm each operation's VERIFY test is present and ran.
- [ ] `cmake --install` + `tests/package/` against the installed prefix — this plan added installed headers (`engine/spatial/*.hpp`, `engine/spatial.hpp`, `engine/task.hpp`, `kernels/blas/level3/channel_bias_epilogue.hpp`), and `docs/CONTRIBUTING.md` §3 requires a packaging change to be verified end to end.
- [ ] Then `superpowers:finishing-a-development-branch`.

---

## What comes after this plan

The remaining three plans, from the spec's §13 build order. Each is written when its turn comes, not now — Plan B's task boundaries depend on the API this plan actually ships.

| Plan | Spec steps | Deliverable |
|---|---|---|
| **B — SushiAI: IR and autograd** | 5–9 | `Attributes::CAPACITY` 4→8; `Value` carries `Layout` and `Graph::finalise()` proves uniformity; twenty operation identities with the `differentiable` flag and its `static_assert`; shape inference in `GraphBuilder`; **the `autograd/backward.cpp` split into per-domain translation units, before any new rule**; the new backward rules; the `IM2COL` rematerialisation pass. Gate: gradcheck green over every new rule. |
| **C — SushiAI: modules and layers** | 10–12 | `Buffer` and `for_each_buffer`; `visit_named_children`; `TrainStep`/`EvalStep` and `BATCHNORM_UPDATE_STATS`; the ten layers and `BasicBlock`. Gate: a small CNN trains, and BatchNorm's running statistics provably move. |
| **D — Models, data and parity** | 13–16 | ResNet-18 trained; YOLOv3-tiny body forward against a PyTorch/Darknet reference dump (obtaining the dump is a task, not an assumption); CIFAR-10 with host-side augmentation and a download script plus synthetic fallback; checkpoint naming/layout parity and the `state_dict` → SACP converter. |

**Note for Plan B, recorded now while the evidence is in hand:** spec CV-4 says "`Graph::finalise()` proves it". There is no `finalise()` on `Graph` today — `include/SushiAI/graph/ir.hpp:360` has `propagate_requires_grad()` and nothing else. Plan B either adds `finalise()` (and moves the propagation into it) or renames the spec's reference. It is a naming decision, not a gap, but it should be made deliberately rather than discovered.
