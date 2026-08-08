/**************************************************************************/
/* builder.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file builder.hpp
 * @brief The one place that knows an operation's output shape.
 *
 * Graph::add_node takes ids and asks no questions: it enforces
 * single-assignment and nothing else, because it is the substrate every pass
 * rewrites and a pass that already knows its own shapes should not pay for a
 * re-derivation. That leaves shape inference homeless, and shape inference is
 * exactly the rule that must be stated once - a layer, a backward rule and a
 * fusion pass all emit MATMUL, and three copies of "the output is [m, n]"
 * would be three chances to disagree.
 *
 * So every emitter lives here, each one validating its operands and declaring
 * its result. Nothing in the library calls @c Graph::add_node directly with a
 * hand-computed shape.
 *
 * Host lane: this header pulls in no SushiBLAS and no SYCL.
 */

#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
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

        /**
         * @struct BatchNormResult
         * @brief A batch normalisation's output and the statistics it computed.
         *
         * The two statistics are returned rather than recomputed for the same
         * reason @ref PoolResult returns its indices: the backward pass needs
         * exactly the numbers *this* forward normalised by. Recomputing them
         * from an input a later node may have overwritten is how a
         * normalisation drifts without anything failing.
         *
         * Plan C's @c BatchNorm2d layer reads @c saved_mean and
         * @c saved_inv_std to update its running statistics, which CV-7 keeps
         * out of the graph entirely.
         */
        struct BatchNormResult
        {
            ValueId output;         /**< @brief The normalised [N,H,W,C] value. */
            ValueId saved_mean;     /**< @brief The batch mean, [C]. */
            ValueId saved_inv_std;  /**< @brief 1/sqrt(var + eps), [C]. */
        };

        /**
         * @struct BatchNormGrads
         * @brief A batch normalisation's three gradients, produced together.
         *
         * One node rather than three, because all three share the two
         * reductions over the batch: splitting them would compute the same
         * sums three times.
         */
        struct BatchNormGrads
        {
            ValueId dx;      /**< @brief The [N,H,W,C] input gradient. */
            ValueId dgamma;  /**< @brief The [C] scale gradient. */
            ValueId dbeta;   /**< @brief The [C] shift gradient. */
        };

        /**
         * @class GraphBuilder
         * @brief Emits shape-checked IR nodes into a Graph.
         *
         * A non-owning view over a Graph. Copyable and cheap; hold one for as
         * long as you are appending, and never past the graph's lifetime.
         */
        class GraphBuilder
        {
            public:
                /**
                 * @brief Wraps @p graph for emission.
                 * @param graph The graph to append to; must outlive this builder.
                 */
                explicit GraphBuilder(Graph& graph) noexcept : graph_(&graph) {}

                /** @brief The graph being built. */
                Graph& graph() const noexcept { return *graph_; }

                // --- Leaves -----------------------------------------------------

                /**
                 * @brief Declares a minibatch input: a leaf rewritten every step.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId input(Shape shape, std::string_view debug_name,
                              DType dtype = DType::FLOAT32) const;

                /**
                 * @brief Declares a learnable parameter: a leaf that needs a gradient.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId parameter(Shape shape, std::string_view debug_name,
                                  DType dtype = DType::FLOAT32) const;

                // --- Linear algebra ---------------------------------------------

                /**
                 * @brief Emits @c MATMUL: C = op(A) * op(B).
                 *
                 * Transposition is a flag, never a view: SushiBLAS derives a
                 * matrix's leading dimension from its @c shape rather than its
                 * @c strides, so a transposed Tensor view is silently
                 * mis-indexed. The flag reaches gemm's own @c transA / @c transB,
                 * which is the only correct route.
                 *
                 * @param a Left operand.
                 * @param b Right operand.
                 * @param transpose_a Whether to transpose @p a.
                 * @param transpose_b Whether to transpose @p b.
                 * @return The product's id.
                 * @throws Error If either operand is not rank 2, the dtypes
                 *         differ, or the contracted extents disagree.
                 */
                ValueId matmul(ValueId a, ValueId b,
                               bool transpose_a = false, bool transpose_b = false) const;

                /**
                 * @brief Emits @c BIAS_ADD: Y = X + 1 * bias^T, over an [N, C] X.
                 * @param x The [N, C] matrix.
                 * @param bias The [C] row added to every row of @p x.
                 * @return The sum's id.
                 * @throws Error If the ranks or extents do not line up.
                 */
                ValueId bias_add(ValueId x, ValueId bias) const;

                /**
                 * @brief Emits @c SUM_ROWS: the column sums of an [N, C] matrix.
                 * @param x The [N, C] matrix.
                 * @return The [C] result's id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueId sum_rows(ValueId x) const;

                // --- Elementwise -------------------------------------------------

                /** @brief Emits @c ADD: elementwise A + B. @throws Error on mismatch. */
                ValueId add(ValueId a, ValueId b) const;

                /** @brief Emits @c SUB: elementwise A - B. @throws Error on mismatch. */
                ValueId sub(ValueId a, ValueId b) const;

                /** @brief Emits @c MUL: elementwise A * B. @throws Error on mismatch. */
                ValueId mul(ValueId a, ValueId b) const;

                /**
                 * @brief Emits @c SCALE: Y = alpha * X.
                 * @param x The operand.
                 * @param alpha The host-side constant factor.
                 * @return The scaled value's id.
                 */
                ValueId scale(ValueId x, double alpha) const;

                /**
                 * @brief Emits @c DYNAMIC_SCALE: Y = X * S[0], with S in device memory.
                 *
                 * The one emitter in this builder whose two operands are
                 * deliberately *not* @c require_same: @p factor is a 1-element
                 * value and @p x is whatever shape it is. They must still agree
                 * on element type, because the kernel multiplies in one type and
                 * a widening operand would be a silent precision change.
                 *
                 * @param x The tensor to scale.
                 * @param factor A 1-element value holding the factor; the host
                 *        is free to rewrite it between replays, which is the
                 *        entire reason this operation exists.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The scaled value's id.
                 * @throws Error If @p factor is not a single element, or the two
                 *         operands disagree on element type.
                 */
                ValueId dynamic_scale(ValueId x, ValueId factor,
                                      std::string_view debug_name = "dynamic_scale") const;

                /** @brief Emits @c EXP: elementwise exp(X), out of place. */
                ValueId exp(ValueId x) const;

                /**
                 * @brief Emits @c ZERO: a fresh all-zero tensor with no operands.
                 * @param shape The value's extents.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueId zeros(Shape shape, DType dtype = DType::FLOAT32) const;

                // --- Activations ---------------------------------------------------

                /** @brief Emits @c RELU. */
                ValueId relu(ValueId x) const;

                /** @brief Emits @c GELU (tanh approximation). */
                ValueId gelu(ValueId x) const;

                /** @brief Emits @c SIGMOID. */
                ValueId sigmoid(ValueId x) const;

                /** @brief Emits @c TANH. */
                ValueId tanh(ValueId x) const;

                /**
                 * @brief Emits @c LEAKY_RELU.
                 * @param x The input.
                 * @param slope The negative-side slope; must lie in (0, 1) -
                 *        at 0 it is a ReLU and at 1 it is the identity, and
                 *        both would make the output-reading gradient (see
                 *        Ops::ACTIVATIONS) unsound.
                 * @throws Error If @p slope is outside (0, 1).
                 */
                ValueId leaky_relu(ValueId x, double slope) const;

                /**
                 * @brief Emits one of the activation-gradient operations.
                 *
                 * @param op @c RELU_BACKWARD, @c GELU_BACKWARD,
                 *        @c SIGMOID_BACKWARD, @c TANH_BACKWARD or
                 *        @c LEAKY_RELU_BACKWARD.
                 * @param dy The incoming gradient.
                 * @param primal The forward input (relu/gelu) or the forward
                 *        output (sigmoid/tanh/leaky_relu) - see ops.hpp.
                 * @param attributes Passed through unchanged; only
                 *        @c LEAKY_RELU_BACKWARD reads from it (the slope, in
                 *        slot 0). There is no default: an absent or
                 *        out-of-range slope on that op would silently emit a
                 *        plain-ReLU gradient rather than the requested one,
                 *        so it is checked against the same (0, 1) range
                 *        @c leaky_relu() enforces on the way in.
                 * @return The incoming-side gradient's id.
                 * @throws Error If @p op is not an activation gradient, the
                 *         operands disagree in shape or dtype, or @p op is
                 *         @c LEAKY_RELU_BACKWARD and slot 0 is outside (0, 1).
                 */
                ValueId activation_backward(OpID op, ValueId dy, ValueId primal,
                                            const Attributes& attributes = {}) const;

                // --- Losses ----------------------------------------------------------

                /**
                 * @brief Emits @c LOG_SOFTMAX over the rows of an [N, C] matrix.
                 * @param x The logits.
                 * @return The [N, C] log-probabilities' id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueId log_softmax(ValueId x) const;

                /**
                 * @brief Emits @c NLL_LOSS: the mean negative log-likelihood.
                 *
                 * @param log_probs An [N, C] value, normally a @c LOG_SOFTMAX result.
                 * @param one_hot An [N, C] one-hot target matrix.
                 * @return A 1-element loss value's id.
                 * @throws Error If the two operands disagree in shape or dtype.
                 */
                ValueId nll_loss(ValueId log_probs, ValueId one_hot) const;

                // --- Structural --------------------------------------------------------

                /**
                 * @brief Emits @c ACCUMULATE: acc_out = acc + delta, in @p acc's buffer.
                 *
                 * The output aliases @p acc at lowering time, so this is the
                 * ordered, allocation-free way to sum gradient contributions at
                 * a fan-out. It is ordered by the graph's own edges rather than
                 * by atomics, which is what keeps replay byte-identical under
                 * SushiRuntime's @c sushiruntime_deterministic_fp contract.
                 *
                 * @param acc The buffer accumulated into.
                 * @param delta The contribution added.
                 * @return The accumulated value's id.
                 * @throws Error If the operands disagree in shape or dtype.
                 */
                ValueId accumulate(ValueId acc, ValueId delta) const;

                /**
                 * @brief Emits @c LAYOUT_CONVERT: the one node CV-4 exempts from
                 *        preserving a value's TensorLayout.
                 *
                 * Permutes a rank-4 value's extents to match @p layout - NHWC's
                 * [N,H,W,C] becomes NCHW's [N,C,H,W] and back - and stamps the
                 * result with the requested layout.
                 *
                 * @param x The rank-4 value to convert.
                 * @param layout The result's layout.
                 * @return The converted value's id.
                 * @throws Error If @p x is not rank 4.
                 */
                ValueId layout_convert(ValueId x, TensorLayout layout) const;

                /**
                 * @brief Emits @c CAST: a copy of @p x in a different element type.
                 *
                 * The one emitter here whose result deliberately does *not*
                 * share its operand's dtype, which is why it is the only route
                 * to one - every other entry point runs @c require_same or an
                 * equivalent check, and that stays true now that casts exist
                 * (see @c graph/cast_insertion.hpp). A conversion that is not
                 * spelled as a node is a conversion the memory planner cannot
                 * size, the dependency tracker cannot order and a reader of the
                 * IR cannot see.
                 *
                 * @param x The value to convert.
                 * @param dtype The result's element type; may equal @p x's, in
                 *        which case this is an ordinary copy.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The converted value's id, with an activation lifetime.
                 * @throws Error If either element type is not a real float -
                 *         SushiBLAS's converting copy is bounded to HALF,
                 *         FLOAT32 and FLOAT64, and refuses the rest by name.
                 */
                ValueId cast(ValueId x, DType dtype, std::string_view debug_name = "cast") const;

                /**
                 * @brief Emits @c CAST writing an *existing* value.
                 *
                 * The result-side half of a cast. @ref cast declares its own
                 * result, which is what an operand conversion needs; widening a
                 * low-precision result into a buffer that already exists - a
                 * gradient the optimizer reads, a logit the training loop reads
                 * - has to write the value the caller already holds the id of,
                 * or the id would stop naming the number.
                 *
                 * @param source The value to read.
                 * @param destination The value to write; must have no producer yet.
                 * @throws Error If the shapes disagree, either element type is
                 *         not a real float, or @p destination is already written.
                 */
                void cast_into(ValueId source, ValueId destination) const;

                /**
                 * @brief Declares a value with the same shape, dtype and lifetime as @p like.
                 * @param like The value to copy the description from.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueId like(ValueId like, std::string_view debug_name = {}) const;

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

                // --- Spatial -----------------------------------------------------

                /**
                 * @brief Emits @c IM2COL: each window becomes one row of a matrix.
                 *
                 * The result is @c [N*H_o*W_o, kernel_h*kernel_w*C], which is
                 * exactly the left operand of the GEMM a convolution is (CV-1).
                 * This overload derives the output extents assuming symmetric
                 * padding; a caller that needs otherwise states its own
                 * extents through @ref im2col_with_extents.
                 *
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The column matrix's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or the
                 *         window does not fit the padded input.
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
                 * @throws Error If @p x is not rank 4, not NHWC, or either
                 *         extent is not positive.
                 */
                ValueId im2col_with_extents(ValueId x, const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c COL2IM: the adjoint of @ref im2col.
                 *
                 * A gather over the input pixels, never an atomic scatter
                 * (CV-6), which is what makes it bit-reproducible. This
                 * overload re-derives @p cols's expected row count from @p g
                 * assuming symmetric padding, which is wrong whenever the
                 * forward pass padded asymmetrically (bottom/right only, as
                 * @c ConvGeometry's own doc warns). A caller that knows the
                 * forward's actual window counts - which a backward rule
                 * always does, because the forward already wrote them into
                 * the graph - states them through @ref col2im_with_extents
                 * instead.
                 *
                 * @param cols The column matrix.
                 * @param image_shape The @c [N,H,W,C] shape to return to.
                 * @param g The window geometry @p cols was produced with.
                 * @return The image's id.
                 * @throws Error If @p image_shape is not rank 4, @p cols is
                 *         not rank 2 or not NHWC, or @p cols does not have
                 *         the row and column counts that geometry implies
                 *         under symmetric padding.
                 */
                ValueId col2im(ValueId cols, Shape image_shape, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c COL2IM with the window counts stated outright.
                 *
                 * The overload a backward rule uses, because the forward's extents
                 * are already written down in the graph and need not be re-derived.
                 *
                 * @param cols The column matrix.
                 * @param image_shape The @c [N,H,W,C] shape to return to.
                 * @param g The window geometry @p cols was produced with.
                 * @param out_h The number of window rows @p cols holds.
                 * @param out_w The number of window columns @p cols holds.
                 * @return The image's id.
                 * @throws Error If @p image_shape is not rank 4, @p cols is
                 *         not rank 2 or not NHWC, or @p cols does not have
                 *         `N*out_h*out_w` rows and `kernel_h*kernel_w*C` columns.
                 */
                ValueId col2im_with_extents(ValueId cols, Shape image_shape,
                                            const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c MAXPOOL2D, with symmetric padding assumed.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value and its argmax indices.
                 * @throws Error If @p x is not rank 4, not NHWC, or the
                 *         window does not fit.
                 */
                PoolResult maxpool2d(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c MAXPOOL2D with the output extents stated outright.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @param out_h The pooled height.
                 * @param out_w The pooled width.
                 * @return The pooled value and its argmax indices.
                 * @throws Error If @p x is not rank 4, not NHWC, or either
                 *         extent is not positive.
                 */
                PoolResult maxpool2d_with_extents(ValueId x, const ConvGeometry& g,
                                                  int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c AVGPOOL2D. Divides by the full window area.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or the
                 *         window does not fit.
                 */
                ValueId avgpool2d(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits @c UPSAMPLE2D: nearest-neighbour, integer scale.
                 * @param x The @c [N,H,W,C] input.
                 * @param scale_h The vertical replication factor.
                 * @param scale_w The horizontal replication factor.
                 * @return The upsampled value's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or either
                 *         scale is below one.
                 */
                ValueId upsample2d(ValueId x, int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits @c SPATIAL_MEAN: the mean of every [H,W] plane.
                 * @param x The @c [N,H,W,C] input.
                 * @return The @c [N,C] result's id.
                 * @throws Error If @p x is not rank 4 or not NHWC.
                 */
                ValueId spatial_mean(ValueId x) const;

                /**
                 * @brief Emits @c MAXPOOL2D_BACKWARD: routes each gradient through the argmax.
                 *
                 * A backward-only emitter, added beside @c maxpool2d_backward's rule
                 * rather than with the forward, because nothing else in the graph
                 * ever has reason to emit it.
                 *
                 * @param dy The @c [N,H_o,W_o,C] output gradient.
                 * @param indices The INT64 argmax @c maxpool2d wrote, same shape as @p dy.
                 * @param input_shape The @c [N,H,W,C] shape of the pooled input.
                 * @param g The window geometry the forward used.
                 * @return The @c [N,H,W,C] input gradient's id.
                 * @throws Error If @p dy is not rank 4, not NHWC, or @p indices
                 *         is not INT64 and shaped like @p dy.
                 */
                ValueId maxpool2d_backward(ValueId dy, ValueId indices, Shape input_shape,
                                          const ConvGeometry& g) const;

                /**
                 * @brief Emits @c AVGPOOL2D_BACKWARD: spreads each gradient over its window.
                 * @param dy The @c [N,H_o,W_o,C] output gradient.
                 * @param input_shape The @c [N,H,W,C] shape of the pooled input.
                 * @param g The window geometry the forward used.
                 * @return The @c [N,H,W,C] input gradient's id.
                 * @throws Error If @p dy is not rank 4 or not NHWC.
                 */
                ValueId avgpool2d_backward(ValueId dy, Shape input_shape,
                                          const ConvGeometry& g) const;

                /**
                 * @brief Emits @c UPSAMPLE2D_BACKWARD: sums each source pixel's replicas.
                 * @param dy The @c [N,H*scale_h,W*scale_w,C] output gradient.
                 * @param input_shape The @c [N,H,W,C] shape of the upsampled input.
                 * @param scale_h The vertical replication factor the forward used.
                 * @param scale_w The horizontal replication factor the forward used.
                 * @return The @c [N,H,W,C] input gradient's id.
                 * @throws Error If @p dy is not rank 4 or not NHWC.
                 */
                ValueId upsample2d_backward(ValueId dy, Shape input_shape,
                                           int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits @c SPATIAL_MEAN_BACKWARD: broadcasts by 1/(H*W).
                 *
                 * The divisor is taken from @p input_shape alone, never passed
                 * separately, so the forward and this backward cannot disagree
                 * about it.
                 *
                 * @param dy The @c [N,C] output gradient.
                 * @param input_shape The @c [N,H,W,C] shape averaged over.
                 * @return The @c [N,H,W,C] input gradient's id.
                 * @throws Error If @p dy is not rank 2 or @p input_shape is not rank 4.
                 */
                ValueId spatial_mean_backward(ValueId dy, Shape input_shape) const;

                /**
                 * @brief Emits @c CONCAT_CHANNELS: joins operands along the channel axis.
                 * @param parts The @c [N,H,W,C_i] operands, at least one, sharing
                 *        N, H, W and dtype.
                 * @return The @c [N,H,W,sum(C_i)] result's id.
                 * @throws Error If @p parts is empty, any operand is not rank 4
                 *         or not NHWC, or the operands disagree on N, H, W or dtype.
                 */
                ValueId concat_channels(SushiRuntime::span<const ValueId> parts) const;

                /**
                 * @brief Emits @c SLICE_CHANNELS: a contiguous channel range.
                 * @param x The @c [N,H,W,C] input.
                 * @param offset The first channel taken.
                 * @param count The number of channels taken.
                 * @return The @c [N,H,W,count] result's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or the range
                 *         @c [offset, offset+count) does not fit @p x's channels.
                 */
                ValueId slice_channels(ValueId x, int64_t offset, int64_t count) const;

                /**
                 * @brief Emits @c CHANNEL_AFFINE: per-channel Y = gamma * X + beta.
                 * @param x The @c [N,H,W,C] input.
                 * @param gamma The @c [C] per-channel scale.
                 * @param beta The @c [C] per-channel shift.
                 * @return The @c [N,H,W,C] result's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or either
                 *         parameter is not a rank-1 @c [C] value.
                 */
                ValueId channel_affine(ValueId x, ValueId gamma, ValueId beta) const;

                /**
                 * @brief Sums an @c [N,H,W,C] value down to @c [C].
                 *
                 * @c CHANNEL_AFFINE's parameter gradients, and nothing else so
                 * far. Deliberately *not* a new identity: in NHWC the fold is
                 * over axis 0 of the zero-copy rank-2 @c [N*H*W, C] view, which
                 * is the case the design says "needs nothing new" - so this is
                 * the two identities that already spell it, @c RESHAPE followed
                 * by @c SUM_ROWS. A third op for a composition that already has
                 * a lowering would be one more thing to keep in step with the
                 * precision policy and the identity table for no new capability.
                 *
                 * Not @c SPATIAL_MEAN's sibling either, despite the shape
                 * similarity: @c SPATIAL_MEAN's surviving position is the pair
                 * @c (n,c) and needs the second segment axis; this one's is
                 * @c c alone.
                 *
                 * @param x The @c [N,H,W,C] input.
                 * @return The @c [C] per-channel sum's id.
                 * @throws Error If @p x is not rank 4 or not NHWC.
                 */
                ValueId sum_channels(ValueId x) const;

                /**
                 * @brief Emits @c BATCHNORM2D: normalise over N, H and W, then scale and shift.
                 *
                 * @param x The @c [N,H,W,C] input.
                 * @param gamma The @c [C] per-channel scale.
                 * @param beta The @c [C] per-channel shift.
                 * @param epsilon Added to the variance before the square root;
                 *        must be positive.
                 * @return The normalised value and the two saved statistics.
                 * @throws Error If @p x is not rank 4, not NHWC, either
                 *         parameter is not a rank-1 @c [C] value, or @p epsilon
                 *         is not positive.
                 */
                BatchNormResult batchnorm2d(ValueId x, ValueId gamma, ValueId beta,
                                            double epsilon) const;

                /**
                 * @brief Emits @c BATCHNORM2D_BACKWARD: all three gradients at once.
                 *
                 * The statistics come in as values rather than being recomputed,
                 * so this node differentiates the normalisation the forward
                 * actually performed. @p gamma is read because @c dx is scaled
                 * by it; the running statistics are not operands at all (CV-7).
                 *
                 * @param dy The @c [N,H,W,C] output gradient.
                 * @param x The forward's input.
                 * @param gamma The @c [C] scale the forward applied.
                 * @param saved_mean The @c [C] batch mean the forward computed.
                 * @param saved_inv_std The @c [C] inverse standard deviation it computed.
                 * @return The input, scale and shift gradients.
                 * @throws Error If @p dy or @p x is not rank 4 or not NHWC, they
                 *         disagree in shape or dtype, or any of the three
                 *         per-channel operands is not a rank-1 @c [C] value.
                 */
                BatchNormGrads batchnorm2d_backward(ValueId dy, ValueId x, ValueId gamma,
                                                    ValueId saved_mean,
                                                    ValueId saved_inv_std) const;

                /**
                 * @brief Emits @c RESHAPE: a view of @p x with different extents.
                 * @param x The value reshaped.
                 * @param shape The result's extents.
                 * @return The reshaped value's id.
                 * @throws Error If @p shape's element count differs from @p x's.
                 */
                ValueId reshape(ValueId x, Shape shape) const;

            private:
                ValueId declare(Shape shape, DType dtype, Lifetime lifetime,
                                std::string_view debug_name = {}) const;

                ValueId elementwise(OpID op, ValueId a, ValueId b, std::string_view what) const;

                ValueId unary(OpID op, ValueId x) const;

                void require_same(ValueId a, ValueId b, std::string_view what) const;

                Graph* graph_;
        };

    } // namespace Graph
} // namespace SushiAI
