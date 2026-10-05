/****************************************************************/
/* builder.hpp                                                  */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file builder.hpp
 * @brief Declares the GraphBuilder class and shape-inferring node emitters.
 * @author Mustafa Garip
 */

#pragma once

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
         * @brief Defines a 2-D window geometry in eight configuration slots.
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
         * @brief Computes the output extent along one convolution axis.
         * @param stride Step between windows; must be positive.
         * @param kernel Number of taps; must be positive.
         * @param dilation Spacing between taps; must be positive.
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
         * @brief Holds pooled tensor output and argmax indices.
         */
        struct PoolResult
        {
            ValueId output;   /**< @brief The pooled @c [N,H_o,W_o,C] value. */
            ValueId indices;  /**< @brief The INT64 argmax, same shape. */
        };

        /**
         * @struct BatchNormResult
         * @brief Holds normalised output tensor and saved batch statistics.
         */
        struct BatchNormResult
        {
            ValueId output;         /**< @brief The normalised [N,H,W,C] value. */
            ValueId saved_mean;     /**< @brief The batch mean, [C]. */
            ValueId saved_inv_std;  /**< @brief 1/sqrt(var + eps), [C]. */
        };

        /**
         * @struct BatchNormGrads
         * @brief Holds the three gradients produced by batch normalisation backward.
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

                /**
                 * @brief Emits MATMUL: C = op(A) * op(B).
                 * @param a Left operand matrix.
                 * @param b Right operand matrix.
                 * @param transpose_a Whether to transpose @p a.
                 * @param transpose_b Whether to transpose @p b.
                 * @return The product value id.
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
                 * @brief Emits DYNAMIC_SCALE: Y = X * S[0] with S in device memory.
                 * @param x The tensor to scale.
                 * @param factor A 1-element value holding the factor.
                 * @param debug_name Diagnostic label for the operation.
                 * @return The scaled value id.
                 * @throws Error If factor is not 1 element or dtypes mismatch.
                 */
                ValueId dynamic_scale(ValueId x, ValueId factor,
                                      std::string_view debug_name = "dynamic_scale") const;

                /** @brief Emits @c EXP: elementwise exp(X), out of place. */
                ValueId exp(ValueId x) const;

                /**
                 * @brief Emits ZERO: a fresh all-zero tensor with no operands.
                 * @param shape The tensor extents.
                 * @param dtype The element data type.
                 * @return The new value id.
                 */
                ValueId zeros(Shape shape, DType dtype = DType::FLOAT32) const;

                /**
                 * @brief Emits ZERO adopting the layout of @p like.
                 * @param like Value whose layout the zeros adopt.
                 * @param shape The tensor extents.
                 * @param dtype The element data type.
                 * @return The new value id.
                 */
                ValueId zeros_like(ValueId like, Shape shape,
                                   DType dtype = DType::FLOAT32) const;

                /** @brief Emits @c RELU. */
                ValueId relu(ValueId x) const;

                /** @brief Emits @c GELU (tanh approximation). */
                ValueId gelu(ValueId x) const;

                /** @brief Emits @c SIGMOID. */
                ValueId sigmoid(ValueId x) const;

                /** @brief Emits @c TANH. */
                ValueId tanh(ValueId x) const;

                /**
                 * @brief Emits LEAKY_RELU with the given negative slope.
                 * @param x The input tensor.
                 * @param slope The negative-side slope; must lie in (0, 1).
                 * @return The activated value id.
                 * @throws Error If @p slope is outside (0, 1).
                 */
                ValueId leaky_relu(ValueId x, double slope) const;

                /**
                 * @brief Emits an activation gradient operation.
                 * @param op The activation backward OpID.
                 * @param dy The incoming gradient tensor.
                 * @param primal The forward input or forward output tensor.
                 * @param attributes Optional attributes such as leaky ReLU slope.
                 * @return The gradient value id.
                 */
                ValueId activation_backward(OpID op, ValueId dy, ValueId primal,
                                            const Attributes& attributes = {}) const;

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

                /**
                 * @brief Emits ACCUMULATE: acc_out = acc + delta into buffer of @p acc.
                 * @param acc The buffer accumulated into.
                 * @param delta The contribution added.
                 * @return The accumulated value id.
                 * @throws Error If operands disagree in shape or dtype.
                 */
                ValueId accumulate(ValueId acc, ValueId delta) const;

                /**
                 * @brief Emits LAYOUT_CONVERT to permute between NHWC and NCHW layouts.
                 * @param x The rank-4 value to convert.
                 * @param layout The target tensor layout.
                 * @return The converted value id.
                 * @throws Error If @p x is not rank 4.
                 */
                ValueId layout_convert(ValueId x, TensorLayout layout) const;

                /**
                 * @brief Emits CAST: a copy of @p x converted to @p dtype.
                 * @param x The value to convert.
                 * @param dtype The target element type.
                 * @param debug_name Diagnostic label for the operation.
                 * @return The converted value id.
                 * @throws Error If either element type is not a real float.
                 */
                ValueId cast(ValueId x, DType dtype, std::string_view debug_name = "cast") const;

                /**
                 * @brief Emits CAST writing directly into an existing value buffer.
                 * @param source The value to read.
                 * @param destination The value to write; must have no producer yet.
                 * @throws Error If shapes disagree, non-float dtype, or destination exists.
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
                 * @brief Declares a value inheriting the layout of @p like.
                 * @param like Value whose layout the result adopts.
                 * @param shape Result extents.
                 * @param dtype Result element type.
                 * @param lifetime Storage arena lifetime.
                 * @return The new value id.
                 */
                ValueId declare_like(ValueId like, Shape shape, DType dtype,
                                     Lifetime lifetime, std::string_view debug_name) const;

                /**
                 * @brief Emits IM2COL: expands convolution windows into matrix rows.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @return The column matrix value id.
                 * @throws Error If @p x is not rank 4 NHWC or window does not fit.
                 */
                ValueId im2col(ValueId x, const ConvGeometry& g) const;

                /**
                 * @brief Emits IM2COL with explicit output extents.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @param out_h Number of window rows; @p out_w Number of window columns.
                 * @return The column matrix value id.
                 */
                ValueId im2col_with_extents(ValueId x, const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits COL2IM: the adjoint operation of IM2COL.
                 * @param cols The column matrix.
                 * @param image_shape The [N,H,W,C] image shape to reconstruct.
                 * @param g The window geometry used during production.
                 * @return The image value id.
                 * @throws Error If shapes, ranks, or implied dimensions mismatch.
                 */
                ValueId col2im(ValueId cols, Shape image_shape, const ConvGeometry& g) const;

                /**
                 * @brief Emits COL2IM with explicit window counts.
                 * @param cols The column matrix.
                 * @param image_shape Reconstructed [N,H,W,C] image shape.
                 * @param g Window geometry.
                 * @param out_h Window rows; @p out_w Window columns.
                 * @return The reconstructed image value id.
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
                 * @brief Emits MAXPOOL2D with explicit output extents.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @param out_h Pooled height; @p out_w Pooled width.
                 * @return The pooled value and argmax indices.
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
                 * @brief Emits UPSAMPLE2D using nearest-neighbour integer scaling.
                 * @param x The [N,H,W,C] input tensor.
                 * @param scale_h Vertical replication factor; must be positive.
                 * @param scale_w Horizontal replication factor; must be positive.
                 * @return The upsampled value id.
                 * @throws Error If @p x is not rank 4 NHWC or scales are below 1.
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
                 * @brief Emits MAXPOOL2D_BACKWARD routing gradients through argmax indices.
                 * @param dy The [N,H_o,W_o,C] output gradient.
                 * @param indices The INT64 argmax indices tensor.
                 * @param input_shape The [N,H,W,C] shape of the pooled input.
                 * @param g The window geometry.
                 * @return The input gradient value id.
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
                 * @brief Emits UPSAMPLE2D_BACKWARD summing replicas into source pixels.
                 * @param dy The output gradient tensor.
                 * @param input_shape The input tensor shape.
                 * @param scale_h Vertical factor; @p scale_w Horizontal factor.
                 * @return The input gradient value id.
                 */
                ValueId upsample2d_backward(ValueId dy, Shape input_shape,
                                           int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits SPATIAL_MEAN_BACKWARD broadcasting by 1/(H*W).
                 * @param dy The [N,C] output gradient.
                 * @param input_shape The [N,H,W,C] shape averaged over.
                 * @return The [N,H,W,C] input gradient value id.
                 * @throws Error If @p dy is not rank 2 or input_shape is not rank 4.
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
                 * @brief Emits SLICE_CHANNELS extracting a contiguous channel slice.
                 * @param x The [N,H,W,C] input tensor.
                 * @param offset The starting channel index.
                 * @param count The number of channels to extract.
                 * @return The [N,H,W,count] sliced tensor id.
                 * @throws Error If range exceeds channels or @p x is not rank 4 NHWC.
                 */
                ValueId slice_channels(ValueId x, int64_t offset, int64_t count) const;

                /**
                 * @brief Emits CHANNEL_AFFINE: per-channel Y = gamma * X + beta.
                 * @param x The [N,H,W,C] input tensor.
                 * @param gamma The [C] per-channel scale parameter.
                 * @param beta The [C] per-channel shift parameter.
                 * @return The [N,H,W,C] affine result id.
                 * @throws Error If shapes or parameter dimensions mismatch.
                 */
                ValueId channel_affine(ValueId x, ValueId gamma, ValueId beta) const;

                /**
                 * @brief Sums an [N,H,W,C] tensor down to [C] per-channel sums.
                 * @param x The [N,H,W,C] input tensor.
                 * @return The [C] per-channel sum value id.
                 * @throws Error If @p x is not rank 4 or not NHWC.
                 */
                ValueId sum_channels(ValueId x) const;

                /**
                 * @brief Emits BATCHNORM2D normalising across N, H, and W dimensions.
                 * @param x The [N,H,W,C] input tensor.
                 * @param gamma The [C] scale; @p beta The [C] shift.
                 * @param epsilon Small constant added to variance; must be positive.
                 * @return The normalised value and saved batch statistics.
                 */
                BatchNormResult batchnorm2d(ValueId x, ValueId gamma, ValueId beta,
                                            double epsilon) const;

                /**
                 * @brief Emits BATCHNORM2D_BACKWARD producing dx, dgamma, and dbeta.
                 * @param dy Output gradient; @p x Forward input tensor.
                 * @param gamma The [C] scale parameter.
                 * @param saved_mean The [C] batch mean computed during forward.
                 * @param saved_inv_std The [C] inverse standard deviation.
                 * @return The input, scale, and shift gradient value ids.
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
