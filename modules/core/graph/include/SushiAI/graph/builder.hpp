/**************************************************************************/
/* builder.hpp                                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file builder.hpp
 * @brief Declares the GraphBuilder class and shape-inferring node emitters.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>

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

        /** @brief Holds the height and the width a spatial operation yields. */
        struct SpatialExtents
        {
            int64_t height = 0; /**< @brief Extent along the height axis. */
            int64_t width = 0;  /**< @brief Extent along the width axis. */
        };

        /**
         * @brief Computes the extents a window yields over a height by width input.
         * @param height Input height, padded above and below by the geometry's top pad.
         * @param width Input width, padded left and right by the geometry's left pad.
         * @return The number of window rows and of window columns, both positive, or
         *         InvalidDescription for a kernel, stride or dilation below one, a negative
         *         padding or a window the padded input does not hold.
         */
        [[nodiscard]] Result<SpatialExtents> window_extents(int64_t height, int64_t width,
                                                            const ConvGeometry& g);

        /**
         * @brief Computes the extents UPSAMPLE2D yields from a height by width input.
         * @return The scaled height and width, or InvalidDescription for a scale below one or
         *         a product beyond 64 bits.
         */
        [[nodiscard]] Result<SpatialExtents> upsampled_extents(int64_t height, int64_t width,
                                                               int64_t scale_h, int64_t scale_w);

        /**
         * @brief Tells whether LEAKY_RELU and its gradient accept @p slope.
         * @return True when @p slope lies strictly inside (0, 1), compared in double.
         */
        [[nodiscard]] bool is_legal_leaky_relu_slope(double slope) noexcept;

        /**
         * @struct PoolResult
         * @brief Holds pooled tensor output and argmax indices.
         */
        struct PoolResult
        {
            ValueID output;   /**< @brief The pooled @c [N,H_o,W_o,C] value. */
            ValueID indices;  /**< @brief The INT64 argmax, same shape. */
        };

        /**
         * @struct BatchNormResult
         * @brief Holds normalised output tensor and saved batch statistics.
         */
        struct BatchNormResult
        {
            ValueID output;         /**< @brief The normalised [N,H,W,C] value. */
            ValueID saved_mean;     /**< @brief The batch mean, [C]. */
            ValueID saved_inv_std;  /**< @brief 1/sqrt(var + eps), [C]. */
            ValueID batch_variance; /**< @brief The biased batch variance before epsilon, [C]. */
        };

        /** @brief Holds the per-channel scale and shift BATCHNORM_FOLD yields. */
        struct BatchNormFold
        {
            ValueID scale;  /**< @brief Holds gamma / sqrt(variance + epsilon), [C]. */
            ValueID shift;  /**< @brief Holds beta - mean * scale, [C]. */
        };

        /**
         * @struct BatchNormGrads
         * @brief Holds the three gradients produced by batch normalisation backward.
         */
        struct BatchNormGrads
        {
            ValueID dx;      /**< @brief The [N,H,W,C] input gradient. */
            ValueID dgamma;  /**< @brief The [C] scale gradient. */
            ValueID dbeta;   /**< @brief The [C] shift gradient. */
        };

        /** @brief Holds the normalised value and the two row statistics LAYER_NORM saves. */
        struct LayerNormResult
        {
            ValueID output;         /**< @brief The normalised [..., D] value. */
            ValueID saved_mean;     /**< @brief The mean of each row, [rows]. */
            ValueID saved_inv_std;  /**< @brief 1/sqrt(var + eps) of each row, [rows]. */
        };

        /** @brief Holds the three gradients LAYER_NORM_BACKWARD yields. */
        struct LayerNormGrads
        {
            ValueID dx;      /**< @brief The [..., D] input gradient. */
            ValueID dgamma;  /**< @brief The [D] gain gradient. */
            ValueID dbeta;   /**< @brief The [D] bias gradient. */
        };

        /** @brief Holds the masked value and the mask K_DROPOUT saves for its gradient. */
        struct DropoutResult
        {
            ValueID output;      /**< @brief The input times the mask, shaped like the input. */
            ValueID saved_mask;  /**< @brief The mask of the run: 0 or 1/(1 - rate) per element. */
        };

        /** @brief Names the attribute slot of a K_DROPOUT node that holds its rate. */
        inline constexpr std::size_t K_DROPOUT_RATE_SLOT = 0;

        /** @brief Returns the site seed a K_DROPOUT node holds in attribute slots 1 and 2. */
        [[nodiscard]] std::uint64_t dropout_seed(const Attributes& attributes) noexcept;

        /**
         * @class GraphBuilder
         * @brief Emits shape-checked IR nodes into a Graph.
         *
         * An emitter takes its operands as Traced and returns a Traced or a Result. Handed a
         * failed operand it returns that failure and emits nothing; a description it refuses
         * marks the graph through refuse() and returns InvalidDescription.
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
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueID input(Shape shape, std::string_view debug_name,
                              DType dtype = DType::Float32) const;

                /**
                 * @brief Declares a learnable parameter: a leaf that needs a gradient.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueID parameter(Shape shape, std::string_view debug_name,
                                  DType dtype = DType::Float32) const;

                /**
                 * @brief Declares a buffer leaf: persistent, no gradient, not rewritten per step.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id, with the BUFFER lifetime.
                 */
                ValueID buffer(Shape shape, std::string_view debug_name,
                               DType dtype = DType::Float32) const;

                /**
                 * @brief Emits MATMUL, C = op(A) * op(B): [m, n], or [B, m, n] for rank 3.
                 * @param a Left operand: a rank-2 matrix, or a rank-3 [B, rows, cols] batch.
                 * @param b Right operand, of the same rank and batch count as @p a.
                 * @return The product, or InvalidDescription when the ranks differ or are not 2
                 *         or 3, or the element types, inner extents or batches disagree.
                 */
                [[nodiscard]] Traced matmul(const Traced& a, const Traced& b,
                                            bool transpose_a = false,
                                            bool transpose_b = false) const;

                /**
                 * @brief Emits @c K_BIAS_ADD: Y = X + 1 * bias^T, over an [N, C] X.
                 * @param x The [N, C] matrix.
                 * @param bias The [C] row added to every row of @p x.
                 * @return The sum, or InvalidDescription when the ranks, extents or element
                 *         types do not line up.
                 */
                [[nodiscard]] Traced bias_add(const Traced& x, const Traced& bias) const;

                /**
                 * @brief Emits @c K_SUM_ROWS: the column sums of an [N, C] matrix.
                 * @return The [C] result, or InvalidDescription when @p x is not rank 2.
                 */
                [[nodiscard]] Traced sum_rows(const Traced& x) const;

                /**
                 * @brief Emits @c K_GATHER_ROWS: the rows of @p table that @p indices names.
                 * @param table A [V, D] value of a real floating type.
                 * @param indices An INT32 value of rank one or more; each element is in [0, V).
                 * @return The [..., D] result, or InvalidDescription when @p table is not rank
                 *         2 or not floating, @p indices is not INT32, or an extent does not fit.
                 */
                [[nodiscard]] Traced gather_rows(const Traced& table, const Traced& indices) const;

                /**
                 * @brief Emits @c K_SCATTER_ADD_ROWS: a zero [table_rows, D] with each row of
                 *        @p rows added to the row its index names, in index order.
                 * @param rows A [..., D] value of a real floating type.
                 * @param indices An INT32 value shaped like @p rows without its last axis.
                 * @param table_rows The row count V of the result; every index is in [0, V).
                 * @return The result, or InvalidDescription when an argument does not fit.
                 */
                [[nodiscard]] Traced scatter_add_rows(const Traced& rows, const Traced& indices,
                                                      int64_t table_rows) const;

                /**
                 * @brief Emits @c K_CAUSAL_MASK: @p scores with every entry above the diagonal of
                 *        its last two axes replaced by @p fill.
                 * @param scores A [..., T, T] value of a real floating type.
                 * @param fill The replacement, stored in attribute slot 0; must be finite.
                 * @return The result, or InvalidDescription when an argument does not fit.
                 */
                [[nodiscard]] Traced causal_mask(const Traced& scores, double fill) const;

                /** @brief Emits @c K_ADD; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced add(const Traced& a, const Traced& b) const;

                /** @brief Emits @c K_SUB; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced sub(const Traced& a, const Traced& b) const;

                /** @brief Emits @c K_MUL; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced mul(const Traced& a, const Traced& b) const;

                /** @brief Emits DIV: a / b; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced div(const Traced& a, const Traced& b) const;

                /** @brief Emits MINIMUM; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced minimum(const Traced& a, const Traced& b) const;

                /** @brief Emits MAXIMUM; InvalidDescription for unequal shapes or types. */
                [[nodiscard]] Traced maximum(const Traced& a, const Traced& b) const;

                /**
                 * @brief Emits LESS_EQUAL: 1 where a <= b and 0 elsewhere.
                 * @return The mask, in the shape, dtype and layout of @p a, or
                 *         InvalidDescription for unequal shapes or types.
                 */
                [[nodiscard]] Traced less_equal(const Traced& a, const Traced& b) const;

                /**
                 * @brief Emits SELECT: a where the condition is non-zero, b elsewhere.
                 * @param condition A mask holding 0 or 1, in the shape and dtype of @p a.
                 * @return The selection, or InvalidDescription when the three operands differ
                 *         in shape or type.
                 */
                [[nodiscard]] Traced select(const Traced& condition, const Traced& a,
                                            const Traced& b) const;

                /** @brief Emits @c K_SCALE: Y = alpha * X. */
                [[nodiscard]] Traced scale(const Traced& x, double alpha) const;

                /**
                 * @brief Emits DYNAMIC_SCALE: Y = X * S[0] with S in device memory.
                 * @param factor A 1-element value of the element type of @p x.
                 * @return The scaled value, or InvalidDescription when @p factor is not one
                 *         element of that type.
                 */
                [[nodiscard]] Traced dynamic_scale(const Traced& x, const Traced& factor,
                                                   std::string_view debug_name =
                                                       "dynamic_scale") const;

                /** @brief Emits @c K_EXP: elementwise exp(X), out of place. */
                [[nodiscard]] Traced exp(const Traced& x) const;

                /** @brief Emits ABS: elementwise |x|, out of place. */
                [[nodiscard]] Traced abs(const Traced& x) const;

                /** @brief Emits ATAN: elementwise arctangent of x, out of place. */
                [[nodiscard]] Traced atan(const Traced& x) const;

                /** @brief Emits SOFTPLUS: elementwise log(1 + exp(x)), out of place. */
                [[nodiscard]] Traced softplus(const Traced& x) const;

                /**
                 * @brief Emits ZERO: a fresh all-zero tensor with no operands.
                 * @return The new value id.
                 */
                ValueID zeros(Shape shape, DType dtype = DType::Float32) const;

                /**
                 * @brief Emits ZERO adopting the layout of @p like.
                 * @param like Value whose layout the zeros adopt.
                 * @return The new value, or the failure @p like holds.
                 */
                [[nodiscard]] Traced zeros_like(const Traced& like, Shape shape,
                                                DType dtype = DType::Float32) const;

                /**
                 * @brief Emits CONSTANT: a tensor with no operands, filled with @p value.
                 * @param value The fill, stored in attribute slot 0.
                 * @return The new value in the NHWC layout, requiring no gradient, or
                 *         InvalidDescription when @p dtype is not half, float32 or float64.
                 */
                [[nodiscard]] Traced constant(Shape shape, double value,
                                              DType dtype = DType::Float32) const;

                /**
                 * @brief Emits CONSTANT in the shape, element type and layout of @p x.
                 * @param x The value described; it is not read.
                 * @return The new value, or InvalidDescription when @p x is not real floating.
                 */
                [[nodiscard]] Traced constant_like(const Traced& x, double value) const;

                /** @brief Emits @c K_RELU. */
                [[nodiscard]] Traced relu(const Traced& x) const;

                /** @brief Emits @c K_GELU (tanh approximation). */
                [[nodiscard]] Traced gelu(const Traced& x) const;

                /** @brief Emits @c K_SIGMOID. */
                [[nodiscard]] Traced sigmoid(const Traced& x) const;

                /** @brief Emits @c K_TANH. */
                [[nodiscard]] Traced tanh(const Traced& x) const;

                /** @brief Emits SILU: elementwise x * sigmoid(x). */
                [[nodiscard]] Traced silu(const Traced& x) const;

                /**
                 * @brief Emits LEAKY_RELU with the given negative slope.
                 * @param slope The negative-side slope; must lie in (0, 1).
                 * @return The activated value, or InvalidDescription for a slope outside (0, 1).
                 */
                [[nodiscard]] Traced leaky_relu(const Traced& x, double slope) const;

                /**
                 * @brief Emits an activation gradient operation.
                 * @param op An activation backward OpID; any other is fatal.
                 * @param primal The forward input or forward output tensor.
                 * @param attributes Optional attributes such as leaky ReLU slope.
                 * @return The gradient, or InvalidDescription for an illegal slope or operands
                 *         of unequal shape or type.
                 */
                [[nodiscard]] Traced activation_backward(OpID op, const Traced& dy,
                                                         const Traced& primal,
                                                         const Attributes& attributes = {}) const;

                /**
                 * @brief Emits @c K_LOG_SOFTMAX over the rows of an [N, C] matrix.
                 * @return The [N, C] log-probabilities, or InvalidDescription when @p x is not
                 *         rank 2.
                 */
                [[nodiscard]] Traced log_softmax(const Traced& x) const;

                /**
                 * @brief Emits a softmax over the last axis: RESHAPE, LOG_SOFTMAX, EXP, RESHAPE.
                 * @return The probabilities, with the shape of @p x, or InvalidDescription when
                 *         @p x has rank below two.
                 */
                [[nodiscard]] Traced softmax(const Traced& x) const;

                /**
                 * @brief Emits @c K_LOG_SOFTMAX_BACKWARD: dy - exp(log_probs) * rowsum(dy).
                 * @param log_probs The [N, C] result of the @c K_LOG_SOFTMAX being differentiated.
                 * @return The [N, C] gradient of the logits, or InvalidDescription when the two
                 *         operands are not rank 2 of one shape and one real floating type.
                 */
                [[nodiscard]] Traced log_softmax_backward(const Traced& dy,
                                                          const Traced& log_probs) const;

                /**
                 * @brief Emits @c K_NLL_LOSS: the mean negative log-likelihood.
                 * @param log_probs An [N, C] value, normally a @c K_LOG_SOFTMAX result.
                 * @param one_hot An [N, C] one-hot target matrix.
                 * @return A 1-element loss, or InvalidDescription when the operands are not
                 *         rank 2 of one shape and one type.
                 */
                [[nodiscard]] Traced nll_loss(const Traced& log_probs,
                                              const Traced& one_hot) const;

                /**
                 * @brief Emits ACCUMULATE: acc_out = acc + delta into buffer of @p acc.
                 * @return The accumulated value, or InvalidDescription when the operands differ
                 *         in shape or type.
                 */
                [[nodiscard]] Traced accumulate(const Traced& acc, const Traced& delta) const;

                /**
                 * @brief Emits LAYOUT_CONVERT to permute between NHWC and NCHW layouts.
                 * @return The converted value, or InvalidDescription when @p x is not rank 4.
                 */
                [[nodiscard]] Traced layout_convert(const Traced& x, TensorLayout layout) const;

                /**
                 * @brief Emits CAST: a copy of @p x converted to @p dtype.
                 * @return The converted value, or InvalidDescription when either element type is
                 *         not a real float.
                 */
                [[nodiscard]] Traced cast(const Traced& x, DType dtype,
                                          std::string_view debug_name = "cast") const;

                /**
                 * @brief Emits CAST writing directly into an existing value buffer.
                 * @param destination The value to write; must have no producer yet.
                 * @return Success, or InvalidDescription when the shapes differ or an element
                 *         type is not a real float.
                 */
                [[nodiscard]] Status cast_into(const Traced& source, ValueID destination) const;

                /**
                 * @brief Declares a value with the same shape, dtype and lifetime as @p like.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueID like(ValueID like, std::string_view debug_name = {}) const;

                /**
                 * @brief Declares a value inheriting the layout of @p like.
                 * @return The new value id.
                 */
                ValueID declare_like(ValueID like, Shape shape, DType dtype,
                                     Lifetime lifetime, std::string_view debug_name) const;

                /**
                 * @brief Emits IM2COL: expands convolution windows into matrix rows.
                 * @param x The [N,H,W,C] input tensor.
                 * @return The column matrix, or InvalidDescription when @p x is not rank 4
                 *         NHWC or the window does not fit.
                 */
                [[nodiscard]] Traced im2col(const Traced& x, const ConvGeometry& g) const;

                /**
                 * @brief Emits IM2COL with explicit output extents.
                 * @param out_h Number of window rows; @p out_w Number of window columns.
                 * @return The column matrix, or InvalidDescription.
                 */
                [[nodiscard]] Traced im2col_with_extents(const Traced& x, const ConvGeometry& g,
                                                         int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits COL2IM: the adjoint operation of IM2COL.
                 * @param image_shape The [N,H,W,C] image shape to reconstruct.
                 * @return The image, or InvalidDescription when shapes, ranks or implied
                 *         extents disagree.
                 */
                [[nodiscard]] Traced col2im(const Traced& cols, Shape image_shape,
                                            const ConvGeometry& g) const;

                /**
                 * @brief Emits COL2IM with explicit window counts.
                 * @param out_h Window rows; @p out_w Window columns.
                 * @return The reconstructed image, or InvalidDescription.
                 */
                [[nodiscard]] Traced col2im_with_extents(const Traced& cols, Shape image_shape,
                                                         const ConvGeometry& g,
                                                         int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits a convolution as IM2COL and MATMUL, with symmetric padding assumed.
                 * @param weight The [C_out,kernel_h,kernel_w,C_in] filters, matching @p g and @p x.
                 * @param bias The [C_out] bias, or an absent Traced to add none.
                 * @return The [N,out_h,out_w,C_out] value, or InvalidDescription.
                 */
                [[nodiscard]] Traced conv2d(const Traced& x, const Traced& weight,
                                            const Traced& bias, const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D: each channel filtered by its own taps.
                 * @param weight The [kernel_h,kernel_w,C] taps, of the input's element type.
                 * @return The [N,out_h,out_w,C] value, or InvalidDescription when @p x, the
                 *         taps or the window disagree with @p g.
                 */
                [[nodiscard]] Traced depthwise_conv2d(const Traced& x, const Traced& weight,
                                                      const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D_BACKWARD_INPUT: the gradient of the input.
                 * @param input_shape The [N,H,W,C] shape of the forward input.
                 * @return The input gradient, or InvalidDescription when the shapes disagree.
                 */
                [[nodiscard]] Traced depthwise_conv2d_backward_input(const Traced& dy,
                                                                     const Traced& weight,
                                                                     Shape input_shape,
                                                                     const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT: the gradient of the taps.
                 * @return The tap gradient, or InvalidDescription when the shapes disagree.
                 */
                [[nodiscard]] Traced depthwise_conv2d_backward_weight(const Traced& dy,
                                                                      const Traced& x,
                                                                      const ConvGeometry& g) const;

                /**
                 * @brief Emits @c K_MAXPOOL2D, with symmetric padding assumed.
                 * @return The pooled value and its argmax indices, or InvalidDescription when
                 *         @p x is not rank 4 NHWC or the window does not fit.
                 */
                [[nodiscard]] Result<PoolResult> maxpool2d(const Traced& x,
                                                           const ConvGeometry& g) const;

                /**
                 * @brief Emits MAXPOOL2D with explicit output extents.
                 * @return The pooled value and argmax indices, or InvalidDescription.
                 */
                [[nodiscard]] Result<PoolResult> maxpool2d_with_extents(const Traced& x,
                                                                        const ConvGeometry& g,
                                                                        int64_t out_h,
                                                                        int64_t out_w) const;

                /**
                 * @brief Emits @c K_AVGPOOL2D. Divides by the full window area.
                 * @return The pooled value, or InvalidDescription when @p x is not rank 4
                 *         NHWC or the window does not fit.
                 */
                [[nodiscard]] Traced avgpool2d(const Traced& x, const ConvGeometry& g) const;

                /**
                 * @brief Emits UPSAMPLE2D using nearest-neighbour integer scaling.
                 * @return The upsampled value, or InvalidDescription when @p x is not rank 4
                 *         NHWC or a scale is below 1.
                 */
                [[nodiscard]] Traced upsample2d(const Traced& x, int64_t scale_h,
                                                int64_t scale_w) const;

                /**
                 * @brief Emits @c K_SPATIAL_MEAN: the mean of every [H,W] plane.
                 * @return The @c [N,C] result, or InvalidDescription when @p x is not rank 4
                 *         NHWC.
                 */
                [[nodiscard]] Traced spatial_mean(const Traced& x) const;

                /**
                 * @brief Emits MAXPOOL2D_BACKWARD routing gradients through argmax indices.
                 * @param input_shape The [N,H,W,C] shape of the pooled input.
                 * @return The input gradient, or InvalidDescription.
                 */
                [[nodiscard]] Traced maxpool2d_backward(const Traced& dy, const Traced& indices,
                                                        Shape input_shape,
                                                        const ConvGeometry& g) const;

                /**
                 * @brief Emits @c K_AVGPOOL2D_BACKWARD: spreads each gradient over its window.
                 * @return The @c [N,H,W,C] input gradient, or InvalidDescription.
                 */
                [[nodiscard]] Traced avgpool2d_backward(const Traced& dy, Shape input_shape,
                                                        const ConvGeometry& g) const;

                /**
                 * @brief Emits UPSAMPLE2D_BACKWARD summing replicas into source pixels.
                 * @return The input gradient, or InvalidDescription.
                 */
                [[nodiscard]] Traced upsample2d_backward(const Traced& dy, Shape input_shape,
                                                         int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits SPATIAL_MEAN_BACKWARD broadcasting by 1/(H*W).
                 * @return The [N,H,W,C] input gradient, or InvalidDescription when @p dy is not
                 *         rank 2 or @p input_shape is not rank 4.
                 */
                [[nodiscard]] Traced spatial_mean_backward(const Traced& dy,
                                                           Shape input_shape) const;

                /**
                 * @brief Emits @c K_CONCAT_CHANNELS: joins operands along the channel axis.
                 * @param parts The @c [N,H,W,C_i] operands sharing N, H, W and dtype.
                 * @return The @c [N,H,W,sum(C_i)] result, or InvalidDescription when @p parts
                 *         is empty or an operand misfits.
                 */
                [[nodiscard]] Traced concat_channels(
                    SushiRuntime::span<const ValueID> parts) const;

                /**
                 * @brief Emits SLICE_CHANNELS extracting a contiguous channel slice.
                 * @return The [N,H,W,count] slice, or InvalidDescription when the range exceeds
                 *         the channels or @p x is not rank 4 NHWC.
                 */
                [[nodiscard]] Traced slice_channels(const Traced& x, int64_t offset,
                                                    int64_t count) const;

                /**
                 * @brief Emits CHANNEL_AFFINE: per-channel Y = gamma * X + beta.
                 * @return The [N,H,W,C] result, or InvalidDescription.
                 */
                [[nodiscard]] Traced channel_affine(const Traced& x, const Traced& gamma,
                                                    const Traced& beta) const;

                /**
                 * @brief Sums an [N,H,W,C] tensor down to [C] per-channel sums.
                 * @return The [C] sums, or InvalidDescription when @p x is not rank 4 NHWC.
                 */
                [[nodiscard]] Traced sum_channels(const Traced& x) const;

                /**
                 * @brief Emits BATCHNORM2D normalising across N, H, and W dimensions.
                 * @param epsilon Small constant added to variance; must be positive.
                 * @return The normalised value, the saved statistics and the batch variance, or
                 *         InvalidDescription.
                 */
                [[nodiscard]] Result<BatchNormResult> batchnorm2d(const Traced& x,
                                                                  const Traced& gamma,
                                                                  const Traced& beta,
                                                                  double epsilon) const;

                /**
                 * @brief Emits BATCHNORM2D_BACKWARD producing dx, dgamma, and dbeta.
                 * @return The input, scale, and shift gradients, or InvalidDescription.
                 */
                [[nodiscard]] Result<BatchNormGrads> batchnorm2d_backward(
                    const Traced& dy, const Traced& x, const Traced& gamma,
                    const Traced& saved_mean, const Traced& saved_inv_std) const;

                /**
                 * @brief Emits BATCHNORM_FOLD: the scale and shift of fixed statistics.
                 * @param epsilon Constant added to the variance; must be positive.
                 * @return scale = gamma / sqrt(variance + epsilon) and shift = beta - mean * scale,
                 *         or InvalidDescription when the four operands do not share one [C]
                 *         shape, one real floating type and a layout.
                 */
                [[nodiscard]] Result<BatchNormFold> batchnorm_fold(const Traced& gamma,
                                                                   const Traced& beta,
                                                                   const Traced& mean,
                                                                   const Traced& variance,
                                                                   double epsilon) const;

                /**
                 * @brief Emits @c K_LAYER_NORM: each [D] row of @p x minus its mean, over the
                 *        root of its biased variance plus @p epsilon, times gamma, plus beta.
                 * @param x A [..., D] value of a real floating type, of rank one or more.
                 * @param epsilon Added to each row's variance; must be positive.
                 * @return The result and the saved statistics, or InvalidDescription.
                 */
                [[nodiscard]] Result<LayerNormResult> layer_norm(const Traced& x,
                                                                 const Traced& gamma,
                                                                 const Traced& beta,
                                                                 double epsilon) const;

                /**
                 * @brief Emits @c K_LAYER_NORM_BACKWARD: the gradients of x, gamma and beta.
                 * @param saved_mean The [rows] means and @p saved_inv_std the [rows] inverse
                 *        deviations the forward saved.
                 * @return The three gradients, or InvalidDescription when an operand misfits.
                 */
                [[nodiscard]] Result<LayerNormGrads> layer_norm_backward(
                    const Traced& dy, const Traced& x, const Traced& gamma,
                    const Traced& saved_mean, const Traced& saved_inv_std) const;

                /**
                 * @brief Emits @c K_DROPOUT: @p x times a mask whose elements are 0 with
                 *        probability @p rate and 1/(1 - rate) otherwise, drawn at each replay.
                 * @param rate Stored in attribute slot 0; must lie in (0, 1).
                 * @param seed The site seed, stored in slots 1 and 2; dropout_seed() reads it.
                 * @return The result and the saved mask, or InvalidDescription.
                 */
                [[nodiscard]] Result<DropoutResult> dropout(const Traced& x, double rate,
                                                            std::uint64_t seed) const;

                /**
                 * @brief Emits @c K_RESHAPE: a view of @p x with different extents.
                 * @return The reshaped value, or InvalidDescription when @p shape's element
                 *         count differs from @p x's.
                 */
                [[nodiscard]] Traced reshape(const Traced& x, Shape shape) const;

                /**
                 * @brief Emits STOP_GRADIENT: the value of @p x, with no gradient path back to it.
                 * @return A value shaped, typed and laid out like @p x that requires no gradient.
                 */
                [[nodiscard]] Traced stop_gradient(const Traced& x) const;

                /**
                 * @brief Records on the graph that a description was refused.
                 * @param failure The refusal an emitter built; see SA_REFUSE_IF.
                 * @return @p failure, for the emitter to return.
                 */
                [[nodiscard]] Status refuse(Status failure) const noexcept
                {
                    graph_->mark_refused();
                    return failure;
                }

                /**
                 * @brief Records on the graph a failure that a call made while tracing returned.
                 * @return @p outcome unchanged; a success leaves the graph as it is.
                 */
                [[nodiscard]] Status observe(Status outcome) const noexcept
                {
                    if (!outcome.is_ok())
                        graph_->mark_refused();

                    return outcome;
                }

                /**
                 * @brief Records on the graph a failure that a call made while tracing returned.
                 * @return @p outcome unchanged; a held value leaves the graph as it is.
                 */
                template<typename T>
                [[nodiscard]] Result<T> observe(Result<T> outcome) const noexcept
                {
                    if (!outcome.has_value())
                        graph_->mark_refused();

                    return outcome;
                }

            private:
                /** @brief Appends a value with no producer to the graph. */
                ValueID declare(Shape shape, DType dtype, Lifetime lifetime,
                                std::string_view debug_name = {}) const;

                /** @brief Emits a binary node over two operands of one shape and type. */
                Traced elementwise(OpID op, const Traced& a, const Traced& b,
                                   std::string_view what) const;

                /** @brief Emits a unary node shaped, typed and laid out like its operand. */
                Traced unary(OpID op, const Traced& x) const;

                /**
                 * @brief Returns InvalidDescription, without marking the graph, when @p a and
                 *        @p b differ in shape or element type.
                 */
                Status require_same(ValueID a, ValueID b, std::string_view what) const;

                /** @brief Points at the graph this builder appends to. */
                Graph* graph_;
        };

    } // namespace Graph
} // namespace SushiAI

/**
 * @brief Returns a refusal through @p builder with a formatted message when @p condition holds.
 * @param builder An object whose @c refuse(Status) records the refusal and returns the Status.
 * @param condition Expression to evaluate; the failure path runs when it is true.
 * @param ... Format string and arguments for SushiLog's format_text, evaluated only on failure.
 */
#define SA_REFUSE_IF(builder, condition, ...)                                          \
    do                                                                                 \
    {                                                                                  \
        if (condition)                                                                 \
        {                                                                              \
            return (builder).refuse(::SushiAI::Status::from_message(                   \
                ::SushiAI::StatusCode::InvalidDescription, [&]()                       \
                {                                                                      \
                    return ::SushiLog::format_text(__VA_ARGS__);                     \
                }));                                                                   \
        }                                                                              \
    } while (0)
