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
#include <SushiAI/failure/error.hpp>
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
         * @param g Needs a positive kernel, stride and dilation, a non-negative padding and a
         *        window the padded input holds; anything else throws.
         * @return The number of window rows and of window columns, both positive.
         */
        [[nodiscard]] SpatialExtents window_extents(int64_t height, int64_t width,
                                                    const ConvGeometry& g);

        /**
         * @brief Computes the extents UPSAMPLE2D yields from a height by width input.
         * @param scale_h Vertical replication factor; below one throws.
         * @param scale_w Horizontal replication factor; below one throws.
         * @return The scaled height and width; a product beyond 64 bits throws.
         */
        [[nodiscard]] SpatialExtents upsampled_extents(int64_t height, int64_t width,
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
                ValueID input(Shape shape, std::string_view debug_name,
                              DType dtype = DType::Float32) const;

                /**
                 * @brief Declares a learnable parameter: a leaf that needs a gradient.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id.
                 */
                ValueID parameter(Shape shape, std::string_view debug_name,
                                  DType dtype = DType::Float32) const;

                /**
                 * @brief Declares a buffer leaf: persistent, no gradient, not rewritten per step.
                 * @param shape The value's extents.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @param dtype The element type.
                 * @return The new value's id, with the BUFFER lifetime.
                 */
                ValueID buffer(Shape shape, std::string_view debug_name,
                               DType dtype = DType::Float32) const;

                /**
                 * @brief Emits MATMUL, C = op(A) * op(B): [m, n], or [B, m, n] for rank 3.
                 * @param a Left operand: a rank-2 matrix, or a rank-3 [B, rows, cols] batch.
                 * @param b Right operand, of the same rank and batch count as @p a.
                 * @param transpose_a Whether to transpose each matrix of @p a.
                 * @param transpose_b Whether to transpose each matrix of @p b.
                 * @throws Error If ranks differ or are not 2 or 3, or extents or batches disagree.
                 */
                ValueID matmul(ValueID a, ValueID b,
                               bool transpose_a = false, bool transpose_b = false) const;

                /**
                 * @brief Emits @c K_BIAS_ADD: Y = X + 1 * bias^T, over an [N, C] X.
                 * @param x The [N, C] matrix.
                 * @param bias The [C] row added to every row of @p x.
                 * @return The sum's id.
                 * @throws Error If the ranks or extents do not line up.
                 */
                ValueID bias_add(ValueID x, ValueID bias) const;

                /**
                 * @brief Emits @c K_SUM_ROWS: the column sums of an [N, C] matrix.
                 * @param x The [N, C] matrix.
                 * @return The [C] result's id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueID sum_rows(ValueID x) const;

                /**
                 * @brief Emits @c K_GATHER_ROWS: the rows of @p table that @p indices names.
                 * @param table A [V, D] value of a real floating type.
                 * @param indices An INT32 value of rank one or more; each element is in [0, V).
                 * @return The [..., D] result's id, or InvalidDescription when @p table is not
                 *         rank 2 or not floating, @p indices is not INT32, or a rank or an extent
                 *         does not fit.
                 */
                Traced gather_rows(ValueID table, ValueID indices) const;

                /**
                 * @brief Emits @c K_SCATTER_ADD_ROWS: a zero [table_rows, D] with each row of
                 *        @p rows added to the row its index names, in index order.
                 * @param rows A [..., D] value of a real floating type.
                 * @param indices An INT32 value shaped like @p rows without its last axis.
                 * @param table_rows The row count V of the result; every index is in [0, V).
                 * @return The result's id, or InvalidDescription when an argument does not fit.
                 */
                Traced scatter_add_rows(ValueID rows, ValueID indices, int64_t table_rows) const;

                /**
                 * @brief Emits @c K_CAUSAL_MASK: @p scores with every entry above the diagonal of
                 *        its last two axes replaced by @p fill.
                 * @param scores A [..., T, T] value of a real floating type.
                 * @param fill The replacement, stored in attribute slot 0; must be finite.
                 * @return The result's id, shaped like @p scores, or InvalidDescription when
                 *         an argument does not fit.
                 */
                Traced causal_mask(ValueID scores, double fill) const;

                /** @brief Emits @c K_ADD: elementwise A + B. @throws Error on mismatch. */
                ValueID add(ValueID a, ValueID b) const;

                /** @brief Emits @c K_SUB: elementwise A - B. @throws Error on mismatch. */
                ValueID sub(ValueID a, ValueID b) const;

                /** @brief Emits @c K_MUL: elementwise A * B. @throws Error on mismatch. */
                ValueID mul(ValueID a, ValueID b) const;

                /**
                 * @brief Emits DIV: elementwise a / b.
                 * @param a The dividend.
                 * @param b The divisor; must share the shape and dtype of @p a.
                 * @return The quotient, shaped and laid out like @p a.
                 */
                ValueID div(ValueID a, ValueID b) const;

                /**
                 * @brief Emits MINIMUM: the elementwise smaller of a and b.
                 * @param a The first operand.
                 * @param b Must share the shape and dtype of @p a.
                 * @return The minimum, shaped and laid out like @p a.
                 */
                ValueID minimum(ValueID a, ValueID b) const;

                /**
                 * @brief Emits MAXIMUM: the elementwise larger of a and b.
                 * @param a The first operand.
                 * @param b Must share the shape and dtype of @p a.
                 * @return The maximum, shaped and laid out like @p a.
                 */
                ValueID maximum(ValueID a, ValueID b) const;

                /**
                 * @brief Emits LESS_EQUAL: 1 where a <= b and 0 elsewhere.
                 * @param a The first operand.
                 * @param b Must share the shape and dtype of @p a.
                 * @return The mask, in the shape, dtype and layout of @p a.
                 */
                ValueID less_equal(ValueID a, ValueID b) const;

                /**
                 * @brief Emits SELECT: a where the condition is non-zero, b elsewhere.
                 * @param condition A mask holding 0 or 1, in the shape and dtype of @p a.
                 * @param a The branch taken where the condition is non-zero.
                 * @param b Must share the shape and dtype of @p a.
                 * @return The selection, shaped and laid out like @p a.
                 */
                ValueID select(ValueID condition, ValueID a, ValueID b) const;

                /**
                 * @brief Emits @c K_SCALE: Y = alpha * X.
                 * @param x The operand.
                 * @param alpha The host-side constant factor.
                 * @return The scaled value's id.
                 */
                ValueID scale(ValueID x, double alpha) const;

                /**
                 * @brief Emits DYNAMIC_SCALE: Y = X * S[0] with S in device memory.
                 * @param x The tensor to scale.
                 * @param factor A 1-element value holding the factor.
                 * @param debug_name Diagnostic label for the operation.
                 * @return The scaled value id.
                 * @throws Error If factor is not 1 element or dtypes mismatch.
                 */
                ValueID dynamic_scale(ValueID x, ValueID factor,
                                      std::string_view debug_name = "dynamic_scale") const;

                /** @brief Emits @c K_EXP: elementwise exp(X), out of place. */
                ValueID exp(ValueID x) const;

                /** @brief Emits ABS: elementwise |x|, out of place. */
                ValueID abs(ValueID x) const;

                /** @brief Emits ATAN: elementwise arctangent of x, out of place. */
                ValueID atan(ValueID x) const;

                /** @brief Emits SOFTPLUS: elementwise log(1 + exp(x)), out of place. */
                ValueID softplus(ValueID x) const;

                /**
                 * @brief Emits ZERO: a fresh all-zero tensor with no operands.
                 * @param shape The tensor extents.
                 * @param dtype The element data type.
                 * @return The new value id.
                 */
                ValueID zeros(Shape shape, DType dtype = DType::Float32) const;

                /**
                 * @brief Emits ZERO adopting the layout of @p like.
                 * @param like Value whose layout the zeros adopt.
                 * @param shape The tensor extents.
                 * @param dtype The element data type.
                 * @return The new value id.
                 */
                ValueID zeros_like(ValueID like, Shape shape,
                                   DType dtype = DType::Float32) const;

                /**
                 * @brief Emits CONSTANT: a tensor with no operands, filled with @p value.
                 * @param shape The tensor extents.
                 * @param value The fill, stored in attribute slot 0.
                 * @param dtype The element type; must be half, float32 or float64.
                 * @return The new value id, in the NHWC layout, requiring no gradient.
                 */
                ValueID constant(Shape shape, double value, DType dtype = DType::Float32) const;

                /**
                 * @brief Emits CONSTANT in the shape, element type and layout of @p x.
                 * @param x The value described; it is not read. Must be a real floating value.
                 * @param value The fill, stored in attribute slot 0.
                 * @return The new value id, requiring no gradient.
                 */
                ValueID constant_like(ValueID x, double value) const;

                /** @brief Emits @c K_RELU. */
                ValueID relu(ValueID x) const;

                /** @brief Emits @c K_GELU (tanh approximation). */
                ValueID gelu(ValueID x) const;

                /** @brief Emits @c K_SIGMOID. */
                ValueID sigmoid(ValueID x) const;

                /** @brief Emits @c K_TANH. */
                ValueID tanh(ValueID x) const;

                /** @brief Emits SILU: elementwise x * sigmoid(x). */
                ValueID silu(ValueID x) const;

                /**
                 * @brief Emits LEAKY_RELU with the given negative slope.
                 * @param x The input tensor.
                 * @param slope The negative-side slope; must lie in (0, 1).
                 * @return The activated value id.
                 * @throws Error If @p slope is outside (0, 1).
                 */
                ValueID leaky_relu(ValueID x, double slope) const;

                /**
                 * @brief Emits an activation gradient operation.
                 * @param op The activation backward OpID.
                 * @param dy The incoming gradient tensor.
                 * @param primal The forward input or forward output tensor.
                 * @param attributes Optional attributes such as leaky ReLU slope.
                 * @return The gradient value id.
                 */
                ValueID activation_backward(OpID op, ValueID dy, ValueID primal,
                                            const Attributes& attributes = {}) const;

                /**
                 * @brief Emits @c K_LOG_SOFTMAX over the rows of an [N, C] matrix.
                 * @param x The logits.
                 * @return The [N, C] log-probabilities' id.
                 * @throws Error If @p x is not rank 2.
                 */
                ValueID log_softmax(ValueID x) const;

                /**
                 * @brief Emits a softmax over the last axis: RESHAPE, LOG_SOFTMAX, EXP, RESHAPE.
                 * @param x The logits, of rank two or more; the last axis is normalised.
                 * @return The probabilities' id, with the shape of @p x.
                 * @throws Error If @p x has rank below two.
                 */
                ValueID softmax(ValueID x) const;

                /**
                 * @brief Emits @c K_LOG_SOFTMAX_BACKWARD: dy - exp(log_probs) * rowsum(dy).
                 * @param dy The [N, C] gradient of the log-probabilities.
                 * @param log_probs The [N, C] result of the @c K_LOG_SOFTMAX being differentiated.
                 * @return The [N, C] gradient of the logits, or InvalidDescription when the two
                 *         operands are not rank 2 of one shape and one real floating type.
                 */
                Traced log_softmax_backward(ValueID dy, ValueID log_probs) const;

                /**
                 * @brief Emits @c K_NLL_LOSS: the mean negative log-likelihood.
                 *
                 * @param log_probs An [N, C] value, normally a @c K_LOG_SOFTMAX result.
                 * @param one_hot An [N, C] one-hot target matrix.
                 * @return A 1-element loss value's id.
                 * @throws Error If the two operands disagree in shape or dtype.
                 */
                ValueID nll_loss(ValueID log_probs, ValueID one_hot) const;

                /**
                 * @brief Emits ACCUMULATE: acc_out = acc + delta into buffer of @p acc.
                 * @param acc The buffer accumulated into.
                 * @param delta The contribution added.
                 * @return The accumulated value id.
                 * @throws Error If operands disagree in shape or dtype.
                 */
                ValueID accumulate(ValueID acc, ValueID delta) const;

                /**
                 * @brief Emits LAYOUT_CONVERT to permute between NHWC and NCHW layouts.
                 * @param x The rank-4 value to convert.
                 * @param layout The target tensor layout.
                 * @return The converted value id.
                 * @throws Error If @p x is not rank 4.
                 */
                ValueID layout_convert(ValueID x, TensorLayout layout) const;

                /**
                 * @brief Emits CAST: a copy of @p x converted to @p dtype.
                 * @param x The value to convert.
                 * @param dtype The target element type.
                 * @param debug_name Diagnostic label for the operation.
                 * @return The converted value id.
                 * @throws Error If either element type is not a real float.
                 */
                ValueID cast(ValueID x, DType dtype, std::string_view debug_name = "cast") const;

                /**
                 * @brief Emits CAST writing directly into an existing value buffer.
                 * @param source The value to read.
                 * @param destination The value to write; must have no producer yet.
                 * @throws Error If shapes disagree, non-float dtype, or destination exists.
                 */
                void cast_into(ValueID source, ValueID destination) const;

                /**
                 * @brief Declares a value with the same shape, dtype and lifetime as @p like.
                 * @param like The value to copy the description from.
                 * @param debug_name A label for diagnostics; must outlive the graph.
                 * @return The new value's id.
                 */
                ValueID like(ValueID like, std::string_view debug_name = {}) const;

                /**
                 * @brief Declares a value inheriting the layout of @p like.
                 * @param like Value whose layout the result adopts.
                 * @param shape Result extents.
                 * @param dtype Result element type.
                 * @param lifetime Storage arena lifetime.
                 * @return The new value id.
                 */
                ValueID declare_like(ValueID like, Shape shape, DType dtype,
                                     Lifetime lifetime, std::string_view debug_name) const;

                /**
                 * @brief Emits IM2COL: expands convolution windows into matrix rows.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @return The column matrix value id.
                 * @throws Error If @p x is not rank 4 NHWC or window does not fit.
                 */
                ValueID im2col(ValueID x, const ConvGeometry& g) const;

                /**
                 * @brief Emits IM2COL with explicit output extents.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @param out_h Number of window rows; @p out_w Number of window columns.
                 * @return The column matrix value id.
                 */
                ValueID im2col_with_extents(ValueID x, const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits COL2IM: the adjoint operation of IM2COL.
                 * @param cols The column matrix.
                 * @param image_shape The [N,H,W,C] image shape to reconstruct.
                 * @param g The window geometry used during production.
                 * @return The image value id.
                 * @throws Error If shapes, ranks, or implied dimensions mismatch.
                 */
                ValueID col2im(ValueID cols, Shape image_shape, const ConvGeometry& g) const;

                /**
                 * @brief Emits COL2IM with explicit window counts.
                 * @param cols The column matrix.
                 * @param image_shape Reconstructed [N,H,W,C] image shape.
                 * @param g Window geometry.
                 * @param out_h Window rows; @p out_w Window columns.
                 * @return The reconstructed image value id.
                 */
                ValueID col2im_with_extents(ValueID cols, Shape image_shape,
                                            const ConvGeometry& g,
                                            int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits a convolution as IM2COL and MATMUL, with symmetric padding assumed.
                 * @param x The [N,H,W,C_in] input; must be NHWC.
                 * @param weight The [C_out,kernel_h,kernel_w,C_in] filters, matching @p g and @p x.
                 * @param bias The [C_out] bias, or NO_VALUE to add none.
                 * @param g The window geometry; a 1x1 unit-stride unpadded one emits no IM2COL.
                 * @return The [N,out_h,out_w,C_out] value id.
                 */
                ValueID conv2d(ValueID x, ValueID weight, ValueID bias,
                               const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D: each channel filtered by its own taps.
                 * @param x The [N,H,W,C] input; must be NHWC.
                 * @param weight The [kernel_h,kernel_w,C] taps, of the input's element type.
                 * @param g The window geometry; symmetric padding is assumed.
                 * @return The [N,out_h,out_w,C] value id.
                 * @throws Error If @p x, the taps or the window disagree with @p g.
                 */
                ValueID depthwise_conv2d(ValueID x, ValueID weight, const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D_BACKWARD_INPUT: the gradient of the input.
                 * @param dy The [N,out_h,out_w,C] output gradient; must be NHWC.
                 * @param weight The [kernel_h,kernel_w,C] taps the forward used.
                 * @param input_shape The [N,H,W,C] shape of the forward input.
                 * @param g The window geometry the forward used.
                 * @throws Error If the shapes disagree with one another or with @p g.
                 */
                ValueID depthwise_conv2d_backward_input(ValueID dy, ValueID weight,
                                                        Shape input_shape,
                                                        const ConvGeometry& g) const;

                /**
                 * @brief Emits K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT: the gradient of the taps.
                 * @param dy The [N,out_h,out_w,C] output gradient; must be NHWC.
                 * @param x The [N,H,W,C] forward input; must be NHWC, of dy's element type.
                 * @param g The window geometry the forward used.
                 * @throws Error If the shapes disagree with one another or with @p g.
                 */
                ValueID depthwise_conv2d_backward_weight(ValueID dy, ValueID x,
                                                         const ConvGeometry& g) const;

                /**
                 * @brief Emits @c K_MAXPOOL2D, with symmetric padding assumed.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value and its argmax indices.
                 * @throws Error If @p x is not rank 4, not NHWC, or the
                 *         window does not fit.
                 */
                PoolResult maxpool2d(ValueID x, const ConvGeometry& g) const;

                /**
                 * @brief Emits MAXPOOL2D with explicit output extents.
                 * @param x The [N,H,W,C] input tensor.
                 * @param g The window geometry.
                 * @param out_h Pooled height; @p out_w Pooled width.
                 * @return The pooled value and argmax indices.
                 */
                PoolResult maxpool2d_with_extents(ValueID x, const ConvGeometry& g,
                                                  int64_t out_h, int64_t out_w) const;

                /**
                 * @brief Emits @c K_AVGPOOL2D. Divides by the full window area.
                 * @param x The @c [N,H,W,C] input.
                 * @param g The window geometry.
                 * @return The pooled value's id.
                 * @throws Error If @p x is not rank 4, not NHWC, or the
                 *         window does not fit.
                 */
                ValueID avgpool2d(ValueID x, const ConvGeometry& g) const;

                /**
                 * @brief Emits UPSAMPLE2D using nearest-neighbour integer scaling.
                 * @param x The [N,H,W,C] input tensor.
                 * @param scale_h Vertical replication factor; must be positive.
                 * @param scale_w Horizontal replication factor; must be positive.
                 * @return The upsampled value id.
                 * @throws Error If @p x is not rank 4 NHWC or scales are below 1.
                 */
                ValueID upsample2d(ValueID x, int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits @c K_SPATIAL_MEAN: the mean of every [H,W] plane.
                 * @param x The @c [N,H,W,C] input.
                 * @return The @c [N,C] result's id.
                 * @throws Error If @p x is not rank 4 or not NHWC.
                 */
                ValueID spatial_mean(ValueID x) const;

                /**
                 * @brief Emits MAXPOOL2D_BACKWARD routing gradients through argmax indices.
                 * @param dy The [N,H_o,W_o,C] output gradient.
                 * @param indices The INT64 argmax indices tensor.
                 * @param input_shape The [N,H,W,C] shape of the pooled input.
                 * @param g The window geometry.
                 * @return The input gradient value id.
                 */
                ValueID maxpool2d_backward(ValueID dy, ValueID indices, Shape input_shape,
                                          const ConvGeometry& g) const;

                /**
                 * @brief Emits @c K_AVGPOOL2D_BACKWARD: spreads each gradient over its window.
                 * @param dy The @c [N,H_o,W_o,C] output gradient.
                 * @param input_shape The @c [N,H,W,C] shape of the pooled input.
                 * @param g The window geometry the forward used.
                 * @return The @c [N,H,W,C] input gradient's id.
                 * @throws Error If @p dy is not rank 4 or not NHWC.
                 */
                ValueID avgpool2d_backward(ValueID dy, Shape input_shape,
                                          const ConvGeometry& g) const;

                /**
                 * @brief Emits UPSAMPLE2D_BACKWARD summing replicas into source pixels.
                 * @param dy The output gradient tensor.
                 * @param input_shape The input tensor shape.
                 * @param scale_h Vertical factor; @p scale_w Horizontal factor.
                 * @return The input gradient value id.
                 */
                ValueID upsample2d_backward(ValueID dy, Shape input_shape,
                                           int64_t scale_h, int64_t scale_w) const;

                /**
                 * @brief Emits SPATIAL_MEAN_BACKWARD broadcasting by 1/(H*W).
                 * @param dy The [N,C] output gradient.
                 * @param input_shape The [N,H,W,C] shape averaged over.
                 * @return The [N,H,W,C] input gradient value id.
                 * @throws Error If @p dy is not rank 2 or input_shape is not rank 4.
                 */
                ValueID spatial_mean_backward(ValueID dy, Shape input_shape) const;

                /**
                 * @brief Emits @c K_CONCAT_CHANNELS: joins operands along the channel axis.
                 * @param parts The @c [N,H,W,C_i] operands, at least one, sharing
                 *        N, H, W and dtype.
                 * @return The @c [N,H,W,sum(C_i)] result's id.
                 * @throws Error If @p parts is empty, any operand is not rank 4
                 *         or not NHWC, or the operands disagree on N, H, W or dtype.
                 */
                ValueID concat_channels(SushiRuntime::span<const ValueID> parts) const;

                /**
                 * @brief Emits SLICE_CHANNELS extracting a contiguous channel slice.
                 * @param x The [N,H,W,C] input tensor.
                 * @param offset The starting channel index.
                 * @param count The number of channels to extract.
                 * @return The [N,H,W,count] sliced tensor id.
                 * @throws Error If range exceeds channels or @p x is not rank 4 NHWC.
                 */
                ValueID slice_channels(ValueID x, int64_t offset, int64_t count) const;

                /**
                 * @brief Emits CHANNEL_AFFINE: per-channel Y = gamma * X + beta.
                 * @param x The [N,H,W,C] input tensor.
                 * @param gamma The [C] per-channel scale parameter.
                 * @param beta The [C] per-channel shift parameter.
                 * @return The [N,H,W,C] affine result id.
                 * @throws Error If shapes or parameter dimensions mismatch.
                 */
                ValueID channel_affine(ValueID x, ValueID gamma, ValueID beta) const;

                /**
                 * @brief Sums an [N,H,W,C] tensor down to [C] per-channel sums.
                 * @param x The [N,H,W,C] input tensor.
                 * @return The [C] per-channel sum value id.
                 * @throws Error If @p x is not rank 4 or not NHWC.
                 */
                ValueID sum_channels(ValueID x) const;

                /**
                 * @brief Emits BATCHNORM2D normalising across N, H, and W dimensions.
                 * @param x The [N,H,W,C] input tensor.
                 * @param gamma The [C] scale; @p beta The [C] shift.
                 * @param epsilon Small constant added to variance; must be positive.
                 * @return The normalised value, the saved statistics and the batch variance.
                 */
                BatchNormResult batchnorm2d(ValueID x, ValueID gamma, ValueID beta,
                                            double epsilon) const;

                /**
                 * @brief Emits BATCHNORM2D_BACKWARD producing dx, dgamma, and dbeta.
                 * @param dy Output gradient; @p x Forward input tensor.
                 * @param gamma The [C] scale parameter.
                 * @param saved_mean The [C] batch mean computed during forward.
                 * @param saved_inv_std The [C] inverse standard deviation.
                 * @return The input, scale, and shift gradient value ids.
                 */
                BatchNormGrads batchnorm2d_backward(ValueID dy, ValueID x, ValueID gamma,
                                                    ValueID saved_mean,
                                                    ValueID saved_inv_std) const;

                /**
                 * @brief Emits BATCHNORM_FOLD: the scale and shift of fixed statistics.
                 * @param gamma The [C] scale; @p beta The [C] shift.
                 * @param mean The [C] mean; @p variance The [C] variance before epsilon.
                 * @param epsilon Constant added to the variance; must be positive.
                 * @return scale = gamma / sqrt(variance + epsilon) and shift = beta - mean * scale.
                 * @pre The four operands share one [C] shape, one real floating type and a layout.
                 */
                BatchNormFold batchnorm_fold(ValueID gamma, ValueID beta, ValueID mean,
                                             ValueID variance, double epsilon) const;

                /**
                 * @brief Emits @c K_LAYER_NORM: each [D] row of @p x minus its mean, over the
                 *        root of its biased variance plus @p epsilon, times gamma, plus beta.
                 * @param x A [..., D] value of a real floating type, of rank one or more.
                 * @param gamma The [D] gain; @p beta The [D] bias; both of the type of @p x.
                 * @param epsilon Added to each row's variance; must be positive.
                 * @return The result and the saved statistics, or InvalidDescription.
                 */
                Result<LayerNormResult> layer_norm(ValueID x, ValueID gamma, ValueID beta,
                                                   double epsilon) const;

                /**
                 * @brief Emits @c K_LAYER_NORM_BACKWARD: the gradients of x, gamma and beta.
                 * @param dy The output gradient, shaped and typed like @p x.
                 * @param x The forward input; @p gamma The [D] gain the forward read.
                 * @param saved_mean The [rows] means and @p saved_inv_std the [rows] inverse
                 *        deviations the forward saved.
                 * @return The three gradients, or InvalidDescription when an operand misfits.
                 */
                Result<LayerNormGrads> layer_norm_backward(ValueID dy, ValueID x, ValueID gamma,
                                                           ValueID saved_mean,
                                                           ValueID saved_inv_std) const;

                /**
                 * @brief Emits @c K_DROPOUT: @p x times a mask whose elements are 0 with
                 *        probability @p rate and 1/(1 - rate) otherwise, drawn at each replay.
                 * @param x A value of a real floating type with an element.
                 * @param rate Stored in attribute slot 0; must lie in (0, 1).
                 * @param seed The site seed, stored in slots 1 and 2; dropout_seed() reads it.
                 * @return The result and the saved mask, or InvalidDescription.
                 */
                Result<DropoutResult> dropout(ValueID x, double rate, std::uint64_t seed) const;

                /**
                 * @brief Emits @c K_RESHAPE: a view of @p x with different extents.
                 * @param x The value reshaped.
                 * @param shape The result's extents.
                 * @return The reshaped value's id.
                 * @throws Error If @p shape's element count differs from @p x's.
                 */
                ValueID reshape(ValueID x, Shape shape) const;

                /**
                 * @brief Emits STOP_GRADIENT: the value of @p x, with no gradient path back to it.
                 * @param x The value to pass through.
                 * @return A value shaped, typed and laid out like @p x that requires no gradient.
                 */
                ValueID stop_gradient(ValueID x) const;

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
                ValueID declare(Shape shape, DType dtype, Lifetime lifetime,
                                std::string_view debug_name = {}) const;

                ValueID elementwise(OpID op, ValueID a, ValueID b, std::string_view what) const;

                ValueID unary(OpID op, ValueID x) const;

                void require_same(ValueID a, ValueID b, std::string_view what) const;

                Graph* graph_;
        };

    } // namespace Graph
} // namespace SushiAI

/**
 * @brief Returns a refusal through @p builder with a formatted message when @p condition holds.
 * @param builder An object whose @c refuse(Status) records the refusal and returns the Status.
 * @param condition Expression to evaluate; the same polarity as SA_THROW_IF.
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
