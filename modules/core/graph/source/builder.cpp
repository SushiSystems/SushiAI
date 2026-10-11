/**************************************************************************/
/* builder.cpp                                                            */
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
 * @file builder.cpp
 * @brief Implements GraphBuilder construction methods for SushiAI IR nodes.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/builder.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Names the attribute slot of the high 32 bits of a dropout site seed. */
            constexpr std::size_t K_DROPOUT_SEED_HIGH_SLOT = 1;

            /** @brief Names the attribute slot of the low 32 bits of a dropout site seed. */
            constexpr std::size_t K_DROPOUT_SEED_LOW_SLOT = 2;

            /** @brief Holds the width in bits of each half of a dropout site seed. */
            constexpr int K_SEED_HALF_BITS = 32;

            /** @brief Selects the low half of a dropout site seed. */
            constexpr std::uint64_t K_SEED_HALF_MASK = 0xFFFFFFFFull;

            /** @brief Builds a rank-1 or rank-2 Shape from loose extents. */
            Shape shape_of(int64_t rows) { return Shape{rows}; }
            Shape shape_of(int64_t rows, int64_t cols) { return Shape{rows, cols}; }
            Shape shape_of(int64_t batch, int64_t rows, int64_t cols)
            {
                return Shape{batch, rows, cols};
            }

            /**
             * @brief Returns @p leading with @p extent appended as its last axis.
             * @pre @p leading has a rank below K_MAX_RANK.
             */
            Shape with_last_axis(const Shape& leading, int64_t extent)
            {
                std::array<int64_t, K_MAX_RANK> dims{};
                const std::size_t rank = static_cast<std::size_t>(leading.rank());
                for (std::size_t axis = 0; axis < rank; ++axis)
                    dims[axis] = leading[static_cast<int32_t>(axis)];

                dims[rank] = extent;
                return Shape(SushiRuntime::span<const int64_t>(dims.data(), rank + 1));
            }

            /**
             * @brief Reports whether INT32 holds every row of a table and every offset of a
             *        list of @p count indices, which runs from 0 to @p count.
             */
            bool fits_row_index(int64_t table_rows, int64_t count) noexcept
            {
                constexpr int64_t K_LARGEST_INDEX = std::numeric_limits<int32_t>::max();
                return table_rows <= K_LARGEST_INDEX && count < K_LARGEST_INDEX;
            }

            /** @brief Throws unless @p value is tagged NHWC, naming the emitter @p what. */
            void require_nhwc(const Value& value, std::string_view what)
            {
                SA_THROW_IF(value.layout != TensorLayout::NHWC,
                            "{} reads NHWC; this value is NCHW. Cross with "
                            "layout_convert first (CV-3).",
                            what);
            }

            /** @brief Throws unless both upsampling scales are at least one, naming @p what. */
            void require_upsample_scales(std::string_view what, int64_t scale_h, int64_t scale_w)
            {
                SA_THROW_IF(scale_h < 1 || scale_w < 1,
                            "{} needs scales of at least one; got {}x{}",
                            what, scale_h, scale_w);
            }
        } // namespace

        SpatialExtents window_extents(int64_t height, int64_t width, const ConvGeometry& g)
        {
            SA_THROW_IF(g.kernel_h <= 0 || g.kernel_w <= 0,
                        "A window needs a positive kernel; got kernel {}x{}",
                        g.kernel_h, g.kernel_w);
            SA_THROW_IF(g.stride_h <= 0 || g.stride_w <= 0,
                        "A window needs a positive stride; got stride {},{}",
                        g.stride_h, g.stride_w);
            SA_THROW_IF(g.dilation_h <= 0 || g.dilation_w <= 0,
                        "A window needs a positive dilation; got dilation {},{}",
                        g.dilation_h, g.dilation_w);
            SA_THROW_IF(g.pad_top < 0 || g.pad_left < 0,
                        "A window needs a non-negative padding; got padding {},{}",
                        g.pad_top, g.pad_left);

            const SpatialExtents extents{
                output_extent(height, g.pad_top, g.pad_top, g.dilation_h, g.kernel_h, g.stride_h),
                output_extent(width, g.pad_left, g.pad_left, g.dilation_w, g.kernel_w, g.stride_w)};

            SA_THROW_IF(extents.height <= 0 || extents.width <= 0,
                        "A window needs to fit its padded input; got a {}x{} window dilated by "
                        "{}x{} over a {}x{} input padded by {},{}",
                        g.kernel_h, g.kernel_w, g.dilation_h, g.dilation_w, height, width,
                        g.pad_top, g.pad_left);

            return extents;
        }

        SpatialExtents upsampled_extents(int64_t height, int64_t width,
                                         int64_t scale_h, int64_t scale_w)
        {
            require_upsample_scales("upsample2d", scale_h, scale_w);

            constexpr int64_t K_LARGEST_EXTENT = std::numeric_limits<int64_t>::max();
            SA_THROW_IF(height > K_LARGEST_EXTENT / scale_h || width > K_LARGEST_EXTENT / scale_w,
                        "upsample2d needs scaled extents that fit in 64 bits; got a {}x{} input "
                        "scaled by {}x{}",
                        height, width, scale_h, scale_w);

            return SpatialExtents{height * scale_h, width * scale_w};
        }

        bool is_legal_leaky_relu_slope(double slope) noexcept
        {
            return slope > 0.0 && slope < 1.0;
        }

        ValueID GraphBuilder::declare(Shape shape, DType dtype, Lifetime lifetime,
                                      std::string_view debug_name) const
        {
            return graph_->add_value(shape, dtype, lifetime, debug_name);
        }

        ValueID GraphBuilder::declare_like(ValueID like, Shape shape, DType dtype,
                                           Lifetime lifetime, std::string_view debug_name) const
        {
            const ValueID out = declare(shape, dtype, lifetime, debug_name);
            graph_->value(out).layout = graph_->value(like).layout;
            return out;
        }

        void GraphBuilder::require_same(ValueID a, ValueID b, std::string_view what) const
        {
            const Value& va = graph_->value(a);
            const Value& vb = graph_->value(b);

            SA_THROW_IF(va.shape != vb.shape,
                        "{} needs equally shaped operands; got {} and {}",
                        what, va.shape.to_string(), vb.shape.to_string());
            SA_THROW_IF(va.dtype != vb.dtype,
                        "{} needs operands of one element type; got {} and {}",
                        what, dtype_name(va.dtype), dtype_name(vb.dtype));
        }

        ValueID GraphBuilder::input(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueID id = declare(shape, dtype, Lifetime::Input, debug_name);
            graph_->add_node(Ops::K_LEAF.id, {}, {id});
            return id;
        }

        ValueID GraphBuilder::parameter(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueID id = declare(shape, dtype, Lifetime::Parameter, debug_name);
            graph_->value(id).requires_grad = true;
            graph_->add_node(Ops::K_LEAF.id, {}, {id});
            return id;
        }

        ValueID GraphBuilder::buffer(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueID id = declare(shape, dtype, Lifetime::Buffer, debug_name);
            graph_->add_node(Ops::K_LEAF.id, {}, {id});
            return id;
        }

        ValueID GraphBuilder::like(ValueID source, std::string_view debug_name) const
        {
            const Value& v = graph_->value(source);
            return declare_like(source, v.shape, v.dtype, v.lifetime, debug_name);
        }

        ValueID GraphBuilder::matmul(ValueID a, ValueID b, bool transpose_a, bool transpose_b) const
        {
            const Value& va = graph_->value(a);
            const Value& vb = graph_->value(b);
            const int32_t rank = va.shape.rank();

            SA_THROW_IF((rank != 2 && rank != 3) || vb.shape.rank() != rank,
                        "MATMUL needs two rank-2 operands or two rank-3 batches; got {} and {}",
                        va.shape.to_string(), vb.shape.to_string());
            SA_THROW_IF(va.dtype != vb.dtype,
                        "MATMUL needs operands of one element type; got {} and {}",
                        dtype_name(va.dtype), dtype_name(vb.dtype));

            // A rank-3 operand is [B, rows, cols]; the matrix axes start at `first`.
            const int32_t first = rank - 2;
            const int64_t m = transpose_a ? va.shape[first + 1] : va.shape[first];
            const int64_t ka = transpose_a ? va.shape[first] : va.shape[first + 1];
            const int64_t kb = transpose_b ? vb.shape[first + 1] : vb.shape[first];
            const int64_t n = transpose_b ? vb.shape[first] : vb.shape[first + 1];

            SA_THROW_IF(ka != kb,
                        "MATMUL contracts {} against {}; the inner extents must agree "
                        "(operands {} and {}, transposed {}/{})",
                        ka, kb, va.shape.to_string(), vb.shape.to_string(),
                        transpose_a, transpose_b);
            SA_THROW_IF(rank == 3 && va.shape[0] != vb.shape[0],
                        "MATMUL batches must agree; got {} and {}",
                        va.shape.to_string(), vb.shape.to_string());

            // The shape is read before declare_like, which may grow the table va and vb refer to.
            const Shape product = rank == 3 ? shape_of(va.shape[0], m, n) : shape_of(m, n);

            // Propagate layout from operand to trigger build-time checks across contractions.
            const ValueID out = declare_like(a, product, va.dtype, Lifetime::Activation, "matmul");

            Attributes attributes{};
            attributes.values[0] = transpose_a ? 1.0 : 0.0;
            attributes.values[1] = transpose_b ? 1.0 : 0.0;

            graph_->add_node(Ops::K_MATMUL.id, {a, b}, {out}, attributes);
            return out;
        }

        ValueID GraphBuilder::bias_add(ValueID x, ValueID bias) const
        {
            const Value& vx = graph_->value(x);
            const Value& vb = graph_->value(bias);

            SA_THROW_IF(vx.shape.rank() != 2,
                        "BIAS_ADD needs a rank-2 left operand; got {}", vx.shape.to_string());
            SA_THROW_IF(vb.shape.rank() != 1,
                        "BIAS_ADD needs a rank-1 bias; got {}", vb.shape.to_string());
            SA_THROW_IF(vb.shape[0] != vx.shape[1],
                        "BIAS_ADD bias has {} entries but {} has {} columns",
                        vb.shape[0], vx.shape.to_string(), vx.shape[1]);
            SA_THROW_IF(vx.dtype != vb.dtype,
                        "BIAS_ADD needs operands of one element type; got {} and {}",
                        dtype_name(vx.dtype), dtype_name(vb.dtype));

            const ValueID out = declare_like(x, vx.shape, vx.dtype, Lifetime::Activation,
                                             "bias_add");
            graph_->add_node(Ops::K_BIAS_ADD.id, {x, bias}, {out});
            return out;
        }

        ValueID GraphBuilder::sum_rows(ValueID x) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", vx.shape.to_string());

            const ValueID out = declare_like(x, shape_of(vx.shape[1]), vx.dtype,
                                             Lifetime::Activation, "sum_rows");
            graph_->add_node(Ops::K_SUM_ROWS.id, {x}, {out});
            return out;
        }

        Traced GraphBuilder::gather_rows(ValueID table, ValueID indices) const
        {
            // Both descriptions are copied; declaring the result grows the value table.
            const Shape table_shape = graph_->value(table).shape;
            const DType table_dtype = graph_->value(table).dtype;
            const Shape index_shape = graph_->value(indices).shape;
            const DType index_dtype = graph_->value(indices).dtype;

            SA_REFUSE_IF(*this, table_shape.rank() != 2,
                         "gather_rows needs a rank-2 [V, D] table; got {}",
                         table_shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(table_dtype),
                         "gather_rows needs a half, float32 or float64 table; got {}",
                         dtype_name(table_dtype));
            SA_REFUSE_IF(*this, index_dtype != DType::Int32,
                         "gather_rows needs an INT32 index; got {}", dtype_name(index_dtype));
            SA_REFUSE_IF(*this, index_shape.rank() < 1 || index_shape.rank() >= K_MAX_RANK,
                         "gather_rows needs an index of rank 1 to {}, so the result fits rank "
                         "{}; got {}",
                         K_MAX_RANK - 1, K_MAX_RANK, index_shape.to_string());
            SA_REFUSE_IF(*this, !fits_row_index(table_shape[0], index_shape.num_elements()),
                         "gather_rows addresses rows in INT32; a table of {} rows read by {} "
                         "indices does not fit",
                         table_shape[0], index_shape.num_elements());

            const ValueID out = declare_like(table, with_last_axis(index_shape, table_shape[1]),
                                             table_dtype, Lifetime::Activation, "gather_rows");
            graph_->add_node(Ops::K_GATHER_ROWS.id, {table, indices}, {out});
            return out;
        }

        Traced GraphBuilder::scatter_add_rows(ValueID rows, ValueID indices,
                                              int64_t table_rows) const
        {
            // Both descriptions are copied; declaring the result grows the value table.
            const Shape row_shape = graph_->value(rows).shape;
            const DType row_dtype = graph_->value(rows).dtype;
            const Shape index_shape = graph_->value(indices).shape;
            const DType index_dtype = graph_->value(indices).dtype;

            SA_REFUSE_IF(*this, row_shape.rank() < 2,
                         "scatter_add_rows needs [..., D] rows of rank 2 or more; got {}",
                         row_shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(row_dtype),
                         "scatter_add_rows needs half, float32 or float64 rows; got {}",
                         dtype_name(row_dtype));
            SA_REFUSE_IF(*this, index_dtype != DType::Int32,
                         "scatter_add_rows needs an INT32 index; got {}",
                         dtype_name(index_dtype));

            const int64_t width = row_shape[row_shape.rank() - 1];
            SA_REFUSE_IF(*this, index_shape.rank() != row_shape.rank() - 1
                                    || with_last_axis(index_shape, width) != row_shape,
                         "scatter_add_rows needs one index per row; rows of {} take an index "
                         "shaped like them without the last axis, and got {}",
                         row_shape.to_string(), index_shape.to_string());
            SA_REFUSE_IF(*this, table_rows < 1
                                    || !fits_row_index(table_rows, index_shape.num_elements()),
                         "scatter_add_rows addresses rows in INT32; a table of {} rows written "
                         "by {} indices does not fit",
                         table_rows, index_shape.num_elements());

            const ValueID out = declare_like(rows, shape_of(table_rows, width), row_dtype,
                                             Lifetime::Activation, "scatter_add_rows");
            graph_->add_node(Ops::K_SCATTER_ADD_ROWS.id, {rows, indices}, {out});
            return out;
        }

        Traced GraphBuilder::causal_mask(ValueID scores, double fill) const
        {
            // The description is copied; declaring the result grows the value table.
            const Shape shape = graph_->value(scores).shape;
            const DType dtype = graph_->value(scores).dtype;
            const int32_t rank = shape.rank();

            SA_REFUSE_IF(*this, rank < 2 || shape[rank - 1] != shape[rank - 2],
                         "causal_mask needs [..., T, T] scores whose last two extents agree; "
                         "got {}",
                         shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(dtype),
                         "causal_mask needs half, float32 or float64 scores; got {}",
                         dtype_name(dtype));
            SA_REFUSE_IF(*this, !std::isfinite(fill),
                         "causal_mask needs a finite fill; got {}", fill);

            const ValueID out = declare_like(scores, shape, dtype, Lifetime::Activation,
                                             "causal_mask");

            Attributes attributes{};
            attributes.values[0] = fill;
            graph_->add_node(Ops::K_CAUSAL_MASK.id, {scores}, {out}, attributes);
            return out;
        }

        ValueID GraphBuilder::elementwise(OpID op, ValueID a, ValueID b,
                                          std::string_view what) const
        {
            require_same(a, b, what);

            const Value& va = graph_->value(a);
            const ValueID out = declare_like(a, va.shape, va.dtype, Lifetime::Activation,
                                             "elementwise");
            graph_->add_node(op, {a, b}, {out});
            return out;
        }

        ValueID GraphBuilder::add(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_ADD.id, a, b, "ADD");
        }

        ValueID GraphBuilder::sub(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_SUB.id, a, b, "SUB");
        }

        ValueID GraphBuilder::mul(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_MUL.id, a, b, "MUL");
        }

        ValueID GraphBuilder::div(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_DIV.id, a, b, "DIV");
        }

        ValueID GraphBuilder::minimum(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_MINIMUM.id, a, b, "MINIMUM");
        }

        ValueID GraphBuilder::maximum(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_MAXIMUM.id, a, b, "MAXIMUM");
        }

        ValueID GraphBuilder::less_equal(ValueID a, ValueID b) const
        {
            return elementwise(Ops::K_LESS_EQUAL.id, a, b, "LESS_EQUAL");
        }

        ValueID GraphBuilder::select(ValueID condition, ValueID a, ValueID b) const
        {
            require_same(a, b, "SELECT");
            require_same(condition, a, "SELECT");

            const Value& va = graph_->value(a);
            const ValueID out = declare_like(a, va.shape, va.dtype, Lifetime::Activation,
                                             "select");
            graph_->add_node(Ops::K_SELECT.id, {condition, a, b}, {out});
            return out;
        }

        ValueID GraphBuilder::scale(ValueID x, double alpha) const
        {
            const Value& vx = graph_->value(x);
            const ValueID out = declare_like(x, vx.shape, vx.dtype, Lifetime::Activation, "scale");

            Attributes attributes{};
            attributes.values[0] = alpha;

            graph_->add_node(Ops::K_SCALE.id, {x}, {out}, attributes);
            return out;
        }

        ValueID GraphBuilder::dynamic_scale(ValueID x, ValueID factor,
                                            std::string_view debug_name) const
        {
            const Value& vx = graph_->value(x);
            const Value& vf = graph_->value(factor);

            SA_THROW_IF(vf.shape.num_elements() != 1,
                        "DYNAMIC_SCALE's factor must be a single element; got {}",
                        vf.shape.to_string());
            SA_THROW_IF(vf.dtype != vx.dtype,
                        "DYNAMIC_SCALE multiplies a {} tensor by a {} factor; the two must "
                        "agree, because the kernel multiplies in one element type and a "
                        "widening operand would change the arithmetic without saying so",
                        dtype_name(vx.dtype), dtype_name(vf.dtype));

            const ValueID out = declare_like(x, vx.shape, vx.dtype, Lifetime::Activation,
                                             debug_name);

            graph_->add_node(Ops::K_DYNAMIC_SCALE.id, {x, factor}, {out});
            return out;
        }

        ValueID GraphBuilder::unary(OpID op, ValueID x) const
        {
            const Value& vx = graph_->value(x);
            const ValueID out = declare_like(x, vx.shape, vx.dtype, Lifetime::Activation,
                                             Ops::name_of(op));
            graph_->add_node(op, {x}, {out});
            return out;
        }

        ValueID GraphBuilder::exp(ValueID x) const { return unary(Ops::K_EXP.id, x); }
        ValueID GraphBuilder::relu(ValueID x) const { return unary(Ops::K_RELU.id, x); }
        ValueID GraphBuilder::gelu(ValueID x) const { return unary(Ops::K_GELU.id, x); }
        ValueID GraphBuilder::sigmoid(ValueID x) const { return unary(Ops::K_SIGMOID.id, x); }
        ValueID GraphBuilder::tanh(ValueID x) const { return unary(Ops::K_TANH.id, x); }

        ValueID GraphBuilder::abs(ValueID x) const
        {
            return unary(Ops::K_ABS.id, x);
        }

        ValueID GraphBuilder::atan(ValueID x) const
        {
            return unary(Ops::K_ATAN.id, x);
        }

        ValueID GraphBuilder::softplus(ValueID x) const
        {
            return unary(Ops::K_SOFTPLUS.id, x);
        }

        ValueID GraphBuilder::silu(ValueID x) const
        {
            return unary(Ops::K_SILU.id, x);
        }

        ValueID GraphBuilder::leaky_relu(ValueID x, double slope) const
        {
            SA_THROW_IF(!is_legal_leaky_relu_slope(slope),
                        "leaky_relu's slope must be in (0, 1); got {}. At 0 it is a ReLU "
                        "and at 1 it is the identity, and both would make the "
                        "output-reading gradient (ops.hpp's ACTIVATIONS) unsound.",
                        slope);

            const Value& in = graph_->value(x);
            const ValueID out = declare_like(x, in.shape, in.dtype, Lifetime::Activation,
                                             "leaky_relu");

            Attributes a{};
            a.values[0] = slope;
            graph_->add_node(Ops::K_LEAKY_RELU.id, {x}, {out}, a);
            return out;
        }

        ValueID GraphBuilder::zeros(Shape shape, DType dtype) const
        {
            // Zero emitter has no inputs to inherit layout from; consumers use zeros_like.
            const ValueID out = declare(shape, dtype, Lifetime::Activation, "zeros");
            graph_->add_node(Ops::K_ZERO.id, {}, {out});
            return out;
        }

        ValueID GraphBuilder::zeros_like(ValueID like, Shape shape, DType dtype) const
        {
            const ValueID out = zeros(shape, dtype);
            graph_->value(out).layout = graph_->value(like).layout;
            return out;
        }

        ValueID GraphBuilder::constant(Shape shape, double value, DType dtype) const
        {
            SA_THROW_IF(!is_real_float(dtype),
                        "CONSTANT fills a real floating tensor (half, float32, float64); got {}",
                        dtype_name(dtype));

            const ValueID out = declare(shape, dtype, Lifetime::Activation, "constant");

            Attributes attributes{};
            attributes.values[0] = value;

            graph_->add_node(Ops::K_CONSTANT.id, {}, {out}, attributes);
            return out;
        }

        ValueID GraphBuilder::constant_like(ValueID x, double value) const
        {
            // The description is held by value; constant() grows the table a reference points into.
            const Value described = graph_->value(x);

            const ValueID out = constant(described.shape, value, described.dtype);
            graph_->value(out).layout = described.layout;
            return out;
        }

        ValueID GraphBuilder::activation_backward(OpID op, ValueID dy, ValueID primal,
                                                  const Attributes& attributes) const
        {
            SA_THROW_IF(!Ops::is_activation_backward(op),
                        "{} is not an activation gradient", Ops::name_of(op));

            // Verify slope explicitly to prevent silent emission of standard ReLU gradients.
            if (op == Ops::K_LEAKY_RELU_BACKWARD.id)
            {
                const double slope = attributes.get(0);
                SA_THROW_IF(!is_legal_leaky_relu_slope(slope),
                            "LEAKY_RELU_BACKWARD's slope (attribute slot 0) must be in (0, 1); "
                            "got {}. An absent or out-of-range slope would silently emit a "
                            "plain-ReLU gradient instead of the leaky one that was asked for.",
                            slope);
            }

            require_same(dy, primal, Ops::name_of(op));

            // Gradients inherit primal layout to ensure layout uniformity across backward nodes.
            const Value& v = graph_->value(dy);
            const ValueID out = declare_like(dy, v.shape, v.dtype, Lifetime::Activation,
                                             Ops::name_of(op));
            graph_->add_node(op, {dy, primal}, {out}, attributes);
            return out;
        }

        ValueID GraphBuilder::log_softmax(ValueID x) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 2,
                        "LOG_SOFTMAX needs a rank-2 operand; got {}", vx.shape.to_string());

            const ValueID out = declare_like(x, vx.shape, vx.dtype, Lifetime::Activation,
                                             "log_softmax");
            graph_->add_node(Ops::K_LOG_SOFTMAX.id, {x}, {out});
            return out;
        }

        ValueID GraphBuilder::softmax(ValueID x) const
        {
            // Shapes are copied out: a declaration below grows the table a reference points into.
            const Shape in = graph_->value(x).shape;
            SA_THROW_IF(in.rank() < 2, "softmax needs a rank-2 or higher operand; got {}",
                        in.to_string());

            // Every axis but the last is a row; LOG_SOFTMAX normalises each row.
            int64_t rows = 1;
            for (int32_t axis = 0; axis + 1 < in.rank(); ++axis)
                rows *= in[axis];

            const ValueID flat = reshape(x, shape_of(rows, in[in.rank() - 1]));
            const ValueID probs = exp(log_softmax(flat));
            return reshape(probs, in);
        }

        Traced GraphBuilder::log_softmax_backward(ValueID dy, ValueID log_probs) const
        {
            // The description is copied; declaring the result grows the value table.
            const Shape shape = graph_->value(log_probs).shape;
            const DType dtype = graph_->value(log_probs).dtype;
            const Shape dy_shape = graph_->value(dy).shape;
            const DType dy_dtype = graph_->value(dy).dtype;

            SA_REFUSE_IF(*this, shape.rank() != 2,
                         "log_softmax_backward needs rank-2 log-probabilities; got {}",
                         shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(dtype),
                         "log_softmax_backward needs half, float32 or float64 log-probabilities; "
                         "got {}",
                         dtype_name(dtype));
            SA_REFUSE_IF(*this, dy_shape != shape || dy_dtype != dtype,
                         "log_softmax_backward over log-probabilities of {} of {} needs a "
                         "gradient like them; got {} of {}",
                         shape.to_string(), dtype_name(dtype), dy_shape.to_string(),
                         dtype_name(dy_dtype));

            const ValueID out = declare_like(dy, shape, dtype, Lifetime::Activation,
                                             "log_softmax_backward");
            graph_->add_node(Ops::K_LOG_SOFTMAX_BACKWARD.id, {dy, log_probs}, {out});
            return out;
        }

        ValueID GraphBuilder::nll_loss(ValueID log_probs, ValueID one_hot) const
        {
            require_same(log_probs, one_hot, "NLL_LOSS");

            const Value& v = graph_->value(log_probs);
            SA_THROW_IF(v.shape.rank() != 2,
                        "NLL_LOSS needs rank-2 operands; got {}", v.shape.to_string());

            const ValueID out = declare_like(log_probs, shape_of(1), v.dtype, Lifetime::Activation,
                                             "loss");
            graph_->add_node(Ops::K_NLL_LOSS.id, {log_probs, one_hot}, {out});
            return out;
        }

        ValueID GraphBuilder::cast(ValueID x, DType dtype, std::string_view debug_name) const
        {
            const Value& vx = graph_->value(x);

            SA_THROW_IF(!is_real_float(vx.dtype) || !is_real_float(dtype),
                        "CAST converts between the real floating types only (half, float32, "
                        "float64) and was asked for {} -> {}. A complex operand has no single "
                        "real value to narrow to, and a float-to-integer conversion is a "
                        "rounding policy this library is not entitled to choose for a caller.",
                        dtype_name(vx.dtype), dtype_name(dtype));

            const ValueID out = declare_like(x, vx.shape, dtype, Lifetime::Activation, debug_name);
            graph_->add_node(Ops::K_CAST.id, {x}, {out});
            return out;
        }

        void GraphBuilder::cast_into(ValueID source, ValueID destination) const
        {
            const Value& vs = graph_->value(source);
            const Value& vd = graph_->value(destination);

            SA_THROW_IF(vs.shape != vd.shape,
                        "CAST copies element for element, so its operands must have one shape; "
                        "got {} and {}", vs.shape.to_string(), vd.shape.to_string());
            SA_THROW_IF(!is_real_float(vs.dtype) || !is_real_float(vd.dtype),
                        "CAST converts between the real floating types only (half, float32, "
                        "float64) and was asked for {} -> {}",
                        dtype_name(vs.dtype), dtype_name(vd.dtype));

            graph_->add_node(Ops::K_CAST.id, {source}, {destination});
        }

        ValueID GraphBuilder::accumulate(ValueID acc, ValueID delta) const
        {
            require_same(acc, delta, "ACCUMULATE");

            const Value& v = graph_->value(acc);
            const ValueID out = declare_like(acc, v.shape, v.dtype, v.lifetime, "accumulated_grad");
            graph_->add_node(Ops::K_ACCUMULATE.id, {acc, delta}, {out});
            return out;
        }

        ValueID GraphBuilder::layout_convert(ValueID x, TensorLayout layout) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 4,
                        "LAYOUT_CONVERT needs a rank-4 operand; got {}", vx.shape.to_string());

            // Permute dimensions based on source operand layout and requested target layout.
            const bool to_nchw = vx.layout == TensorLayout::NHWC && layout == TensorLayout::NCHW;
            const bool to_nhwc = vx.layout == TensorLayout::NCHW && layout == TensorLayout::NHWC;
            const Shape converted = to_nchw
                ? Shape{vx.shape[0], vx.shape[3], vx.shape[1], vx.shape[2]}
                : to_nhwc
                    ? Shape{vx.shape[0], vx.shape[2], vx.shape[3], vx.shape[1]}
                    : vx.shape;

            const ValueID out = declare(converted, vx.dtype, Lifetime::Activation,
                                        "layout_convert");
            graph_->value(out).layout = layout;
            graph_->add_node(Ops::K_LAYOUT_CONVERT.id, {x}, {out});
            return out;
        }

        namespace
        {
            /** @brief Writes a ConvGeometry into CV-5's eight attribute slots. */
            Attributes geometry_attributes(const ConvGeometry& g) noexcept
            {
                Attributes a{};
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

            /**
             * @brief Writes convolution geometry and output extents into attribute slots.
             *
             * Preserves explicit window grid dimensions for lowering passes.
             */
            Attributes windowed_attributes(const ConvGeometry& g,
                                           int64_t out_h, int64_t out_w) noexcept
            {
                Attributes a = geometry_attributes(g);
                a.values[8] = static_cast<double>(out_h);
                a.values[9] = static_cast<double>(out_w);
                return a;
            }
        } // namespace

        ValueID GraphBuilder::im2col(ValueID x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());
            require_nhwc(in, "im2col");

            // Default im2col assumes symmetric padding; asymmetric cases use im2col_with_extents.
            const SpatialExtents out = window_extents(in.shape[1], in.shape[2], g);

            return im2col_with_extents(x, g, out.height, out.width);
        }

        ValueID GraphBuilder::im2col_with_extents(ValueID x, const ConvGeometry& g,
                                                  int64_t out_h, int64_t out_w) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());
            require_nhwc(in, "im2col");
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "im2col was given non-positive output extents {}x{}", out_h, out_w);

            const int64_t rows = in.shape[0] * out_h * out_w;
            const int64_t cols = g.kernel_h * g.kernel_w * in.shape[3];

            const ValueID out = declare_like(x, shape_of(rows, cols), in.dtype,
                                             Lifetime::Activation, "cols");
            graph_->add_node(Ops::K_IM2COL.id, {x}, {out}, windowed_attributes(g, out_h, out_w));
            return out;
        }

        ValueID GraphBuilder::col2im(ValueID cols, Shape image_shape,
                                     const ConvGeometry& g) const
        {
            SA_THROW_IF(image_shape.rank() != 4,
                        "col2im returns a rank-4 [N,H,W,C] image; got rank {}",
                        image_shape.rank());

            // Default col2im assumes symmetric padding; backward passes use col2im_with_extents.
            const SpatialExtents out = window_extents(image_shape[1], image_shape[2], g);

            return col2im_with_extents(cols, image_shape, g, out.height, out.width);
        }

        ValueID GraphBuilder::col2im_with_extents(ValueID cols, Shape image_shape,
                                                   const ConvGeometry& g,
                                                   int64_t out_h, int64_t out_w) const
        {
            const Value& c = graph_->value(cols);
            SA_THROW_IF(image_shape.rank() != 4,
                        "col2im returns a rank-4 [N,H,W,C] image; got rank {}",
                        image_shape.rank());
            SA_THROW_IF(c.shape.rank() != 2,
                        "col2im reads a rank-2 column matrix; got rank {}", c.shape.rank());
            SA_THROW_IF(c.layout != TensorLayout::NHWC,
                        "col2im returns NHWC; this column matrix was produced from an "
                        "NCHW value. Cross with layout_convert first (CV-3).");

            const int64_t rows = image_shape[0] * out_h * out_w;
            const int64_t width = g.kernel_h * g.kernel_w * image_shape[3];

            SA_THROW_IF(c.shape[0] != rows || c.shape[1] != width,
                        "col2im was given a [{}, {}] column matrix, but that geometry "
                        "over a [{}, {}, {}, {}] image implies [{}, {}].",
                        c.shape[0], c.shape[1], image_shape[0], image_shape[1],
                        image_shape[2], image_shape[3], rows, width);

            const ValueID out = declare_like(cols, image_shape, c.dtype,
                                             Lifetime::Activation, "image");
            graph_->add_node(Ops::K_COL2IM.id, {cols}, {out}, windowed_attributes(g, out_h, out_w));
            return out;
        }

        namespace
        {
            /** @brief Tells whether @p g is an unpadded 1x1 window of unit stride and dilation. */
            bool is_pointwise(const ConvGeometry& g) noexcept
            {
                return g.kernel_h == 1 && g.kernel_w == 1 && g.stride_h == 1 && g.stride_w == 1
                    && g.dilation_h == 1 && g.dilation_w == 1 && g.pad_top == 0 && g.pad_left == 0;
            }
        } // namespace

        ValueID GraphBuilder::conv2d(ValueID x, ValueID weight, ValueID bias,
                                     const ConvGeometry& g) const
        {
            // Shapes are held by value; each emitter below grows the table a reference points into.
            const Shape image = graph_->value(x).shape;
            const Shape filter = graph_->value(weight).shape;

            SA_THROW_IF(image.rank() != 4,
                        "conv2d needs a rank-4 [N,H,W,C_in] input; got {}", image.to_string());
            require_nhwc(graph_->value(x), "conv2d");
            SA_THROW_IF(filter.rank() != 4,
                        "conv2d needs a rank-4 [C_out,kernel_h,kernel_w,C_in] weight; got {}",
                        filter.to_string());
            SA_THROW_IF(filter[1] != g.kernel_h || filter[2] != g.kernel_w || filter[3] != image[3],
                        "conv2d with a {}x{} window over a {} input needs a [C_out, {}, {}, {}] "
                        "weight; got {}",
                        g.kernel_h, g.kernel_w, image.to_string(), g.kernel_h, g.kernel_w,
                        image[3], filter.to_string());

            if (bias != K_NO_VALUE)
            {
                const Shape& offsets = graph_->value(bias).shape;
                SA_THROW_IF(offsets.rank() != 1 || offsets[0] != filter[0],
                            "conv2d with {} filters needs a [{}] bias; got {}",
                            filter[0], filter[0], offsets.to_string());
            }

            const SpatialExtents out = window_extents(image[1], image[2], g);

            // Row-major [C_out,R,S,C_in] flattens to (r * S + s) * C_in + c, IM2COL's column order.
            const int64_t taps = g.kernel_h * g.kernel_w * image[3];
            const ValueID filters = reshape(weight, Shape({filter[0], taps}));

            const ValueID cols = is_pointwise(g)
                ? reshape(x, Shape({image[0] * image[1] * image[2], image[3]}))
                : im2col_with_extents(x, g, out.height, out.width);

            const ValueID product = matmul(cols, filters, false, true);
            const ValueID rows = bias == K_NO_VALUE ? product : bias_add(product, bias);

            return reshape(rows, Shape({image[0], out.height, out.width, filter[0]}));
        }

        namespace
        {
            /** @brief Throws unless @p taps is the [kernel_h, kernel_w, C] weight of @p g. */
            void require_depthwise_taps(const Shape& taps, int64_t channels, const ConvGeometry& g,
                                        std::string_view what)
            {
                SA_THROW_IF(taps.rank() != 3,
                            "{} needs a rank-3 [kernel_h, kernel_w, C] weight; got {}",
                            what, taps.to_string());
                SA_THROW_IF(taps[0] != g.kernel_h || taps[1] != g.kernel_w || taps[2] != channels,
                            "{} with a {}x{} window over {} channels needs a [{}, {}, {}] weight; "
                            "got {}",
                            what, g.kernel_h, g.kernel_w, channels, g.kernel_h, g.kernel_w,
                            channels, taps.to_string());
            }

            /** @brief Throws unless @p a and @p b hold one element type, naming @p what. */
            void require_one_dtype(DType a, DType b, std::string_view what)
            {
                SA_THROW_IF(a != b, "{} needs operands of one element type; got {} and {}",
                            what, dtype_name(a), dtype_name(b));
            }

            /**
             * @brief Throws unless @p grid is the [N,out_h,out_w,C] window grid over @p image.
             */
            void require_window_grid(const Shape& grid, const Shape& image,
                                     const SpatialExtents& out, std::string_view what)
            {
                SA_THROW_IF(grid.rank() != 4,
                            "{} needs a rank-4 [N,H,W,C] output gradient; got {}",
                            what, grid.to_string());
                SA_THROW_IF(grid[0] != image[0] || grid[1] != out.height || grid[2] != out.width
                                || grid[3] != image[3],
                            "{} was given an output gradient of {}, but a window over {} yields "
                            "[{}, {}, {}, {}]",
                            what, grid.to_string(), image.to_string(), image[0], out.height,
                            out.width, image[3]);
            }
        } // namespace

        ValueID GraphBuilder::depthwise_conv2d(ValueID x, ValueID weight,
                                               const ConvGeometry& g) const
        {
            // Shapes are copied out: a declaration below grows the table a reference points into.
            const Shape image = graph_->value(x).shape;
            const DType dtype = graph_->value(x).dtype;

            SA_THROW_IF(image.rank() != 4,
                        "depthwise_conv2d needs a rank-4 [N,H,W,C] input; got {}",
                        image.to_string());
            require_nhwc(graph_->value(x), "depthwise_conv2d");
            require_depthwise_taps(graph_->value(weight).shape, image[3], g, "depthwise_conv2d");
            require_one_dtype(graph_->value(weight).dtype, dtype, "depthwise_conv2d");

            const SpatialExtents out = window_extents(image[1], image[2], g);

            const ValueID result = declare_like(
                x, Shape({image[0], out.height, out.width, image[3]}), dtype,
                Lifetime::Activation, "depthwise");
            graph_->add_node(Ops::K_DEPTHWISE_CONV2D.id, {x, weight}, {result},
                             geometry_attributes(g));
            return result;
        }

        ValueID GraphBuilder::depthwise_conv2d_backward_input(ValueID dy, ValueID weight,
                                                              Shape input_shape,
                                                              const ConvGeometry& g) const
        {
            const Shape grad = graph_->value(dy).shape;
            const DType dtype = graph_->value(dy).dtype;

            SA_THROW_IF(input_shape.rank() != 4,
                        "depthwise_conv2d_backward_input returns a rank-4 [N,H,W,C] input "
                        "gradient; got {}", input_shape.to_string());
            require_nhwc(graph_->value(dy), "depthwise_conv2d_backward_input");
            require_depthwise_taps(graph_->value(weight).shape, input_shape[3], g,
                                   "depthwise_conv2d_backward_input");
            require_one_dtype(graph_->value(weight).dtype, dtype,
                              "depthwise_conv2d_backward_input");

            const SpatialExtents out = window_extents(input_shape[1], input_shape[2], g);
            require_window_grid(grad, input_shape, out, "depthwise_conv2d_backward_input");

            const ValueID result = declare_like(dy, input_shape, dtype, Lifetime::Activation, "dx");
            graph_->add_node(Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT.id, {dy, weight}, {result},
                             geometry_attributes(g));
            return result;
        }

        ValueID GraphBuilder::depthwise_conv2d_backward_weight(ValueID dy, ValueID x,
                                                               const ConvGeometry& g) const
        {
            const Shape image = graph_->value(x).shape;
            const Shape grad = graph_->value(dy).shape;
            const DType dtype = graph_->value(dy).dtype;

            SA_THROW_IF(image.rank() != 4,
                        "depthwise_conv2d_backward_weight needs a rank-4 [N,H,W,C] input; got {}",
                        image.to_string());
            require_nhwc(graph_->value(x), "depthwise_conv2d_backward_weight");
            require_nhwc(graph_->value(dy), "depthwise_conv2d_backward_weight");
            require_one_dtype(graph_->value(x).dtype, dtype, "depthwise_conv2d_backward_weight");

            const SpatialExtents out = window_extents(image[1], image[2], g);
            require_window_grid(grad, image, out, "depthwise_conv2d_backward_weight");

            const ValueID result = declare_like(dy, Shape({g.kernel_h, g.kernel_w, image[3]}),
                                                dtype, Lifetime::Activation, "dweight");
            graph_->add_node(Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT.id, {dy, x}, {result},
                             geometry_attributes(g));
            return result;
        }

        PoolResult GraphBuilder::maxpool2d(ValueID x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "maxpool2d");

            // Default maxpool2d assumes symmetric padding; asymmetric cases use with_extents.
            const SpatialExtents out = window_extents(in.shape[1], in.shape[2], g);

            return maxpool2d_with_extents(x, g, out.height, out.width);
        }

        PoolResult GraphBuilder::maxpool2d_with_extents(ValueID x, const ConvGeometry& g,
                                                        int64_t out_h, int64_t out_w) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "maxpool2d");
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "maxpool2d was given non-positive output extents {}x{}",
                        out_h, out_w);

            const Shape pooled({in.shape[0], out_h, out_w, in.shape[3]});
            const ValueID out = declare_like(x, pooled, in.dtype, Lifetime::Activation, "pooled");
            const ValueID idx = declare_like(x, pooled, DType::Int64, Lifetime::Activation,
                                             "argmax");

            graph_->add_node(Ops::K_MAXPOOL2D.id, {x}, {out, idx}, geometry_attributes(g));
            return PoolResult{out, idx};
        }

        ValueID GraphBuilder::avgpool2d(ValueID x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "avgpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "avgpool2d");

            const SpatialExtents pooled = window_extents(in.shape[1], in.shape[2], g);

            const ValueID out = declare_like(
                x, Shape({in.shape[0], pooled.height, pooled.width, in.shape[3]}),
                in.dtype, Lifetime::Activation, "pooled");
            graph_->add_node(Ops::K_AVGPOOL2D.id, {x}, {out}, geometry_attributes(g));
            return out;
        }

        ValueID GraphBuilder::upsample2d(ValueID x, int64_t scale_h, int64_t scale_w) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "upsample2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "upsample2d");

            const SpatialExtents scaled = upsampled_extents(in.shape[1], in.shape[2],
                                                            scale_h, scale_w);

            const ValueID out = declare_like(
                x, Shape({in.shape[0], scaled.height, scaled.width, in.shape[3]}),
                in.dtype, Lifetime::Activation, "upsampled");

            Attributes a{};
            a.values[0] = static_cast<double>(scale_h);
            a.values[1] = static_cast<double>(scale_w);
            graph_->add_node(Ops::K_UPSAMPLE2D.id, {x}, {out}, a);
            return out;
        }

        ValueID GraphBuilder::spatial_mean(ValueID x) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "spatial_mean needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "spatial_mean reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            const ValueID out = declare_like(x, Shape({in.shape[0], in.shape[3]}), in.dtype,
                                             Lifetime::Activation, "spatial_mean");
            graph_->add_node(Ops::K_SPATIAL_MEAN.id, {x}, {out});
            return out;
        }

        ValueID GraphBuilder::maxpool2d_backward(ValueID dy, ValueID indices, Shape input_shape,
                                                 const ConvGeometry& g) const
        {
            const Value& gy = graph_->value(dy);
            SA_THROW_IF(gy.shape.rank() != 4,
                        "maxpool2d_backward needs a rank-4 [N,H,W,C] output gradient; got "
                        "rank {}", gy.shape.rank());
            SA_THROW_IF(gy.layout != TensorLayout::NHWC,
                        "maxpool2d_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(input_shape.rank() != 4,
                        "maxpool2d_backward returns a rank-4 [N,H,W,C] input gradient; got "
                        "rank {}", input_shape.rank());

            const Value& ix = graph_->value(indices);
            SA_THROW_IF(ix.dtype != DType::Int64,
                        "maxpool2d_backward's indices must be INT64; got {}",
                        dtype_name(ix.dtype));
            SA_THROW_IF(ix.shape != gy.shape,
                        "maxpool2d_backward's indices must share its output gradient's shape");

            const ValueID out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::Activation, "dx");
            graph_->add_node(Ops::K_MAXPOOL2D_BACKWARD.id, {dy, indices}, {out},
                             geometry_attributes(g));
            return out;
        }

        ValueID GraphBuilder::avgpool2d_backward(ValueID dy, Shape input_shape,
                                                 const ConvGeometry& g) const
        {
            const Value& gy = graph_->value(dy);
            SA_THROW_IF(gy.shape.rank() != 4,
                        "avgpool2d_backward needs a rank-4 [N,H,W,C] output gradient; got "
                        "rank {}", gy.shape.rank());
            SA_THROW_IF(gy.layout != TensorLayout::NHWC,
                        "avgpool2d_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(input_shape.rank() != 4,
                        "avgpool2d_backward returns a rank-4 [N,H,W,C] input gradient; got "
                        "rank {}", input_shape.rank());

            const ValueID out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::Activation, "dx");
            graph_->add_node(Ops::K_AVGPOOL2D_BACKWARD.id, {dy}, {out}, geometry_attributes(g));
            return out;
        }

        ValueID GraphBuilder::upsample2d_backward(ValueID dy, Shape input_shape,
                                                  int64_t scale_h, int64_t scale_w) const
        {
            const Value& gy = graph_->value(dy);
            SA_THROW_IF(gy.shape.rank() != 4,
                        "upsample2d_backward needs a rank-4 [N,H,W,C] output gradient; got "
                        "rank {}", gy.shape.rank());
            SA_THROW_IF(gy.layout != TensorLayout::NHWC,
                        "upsample2d_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(input_shape.rank() != 4,
                        "upsample2d_backward returns a rank-4 [N,H,W,C] input gradient; got "
                        "rank {}", input_shape.rank());
            require_upsample_scales("upsample2d_backward", scale_h, scale_w);

            const ValueID out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::Activation, "dx");

            Attributes a{};
            a.values[0] = static_cast<double>(scale_h);
            a.values[1] = static_cast<double>(scale_w);
            graph_->add_node(Ops::K_UPSAMPLE2D_BACKWARD.id, {dy}, {out}, a);
            return out;
        }

        ValueID GraphBuilder::spatial_mean_backward(ValueID dy, Shape input_shape) const
        {
            const Value& gy = graph_->value(dy);
            SA_THROW_IF(gy.shape.rank() != 2,
                        "spatial_mean_backward needs a rank-2 [N,C] output gradient; got "
                        "rank {}", gy.shape.rank());
            SA_THROW_IF(gy.layout != TensorLayout::NHWC,
                        "spatial_mean_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(input_shape.rank() != 4,
                        "spatial_mean_backward returns a rank-4 [N,H,W,C] input gradient; got "
                        "rank {}", input_shape.rank());

            const ValueID out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::Activation, "dx");
            graph_->add_node(Ops::K_SPATIAL_MEAN_BACKWARD.id, {dy}, {out});
            return out;
        }

        ValueID GraphBuilder::concat_channels(SushiRuntime::span<const ValueID> parts) const
        {
            SA_THROW_IF(parts.empty(), "concat_channels needs at least one operand");

            const Value& first = graph_->value(parts[0]);
            SA_THROW_IF(first.shape.rank() != 4,
                        "concat_channels needs rank-4 operands; got rank {}",
                        first.shape.rank());
            SA_THROW_IF(first.layout != TensorLayout::NHWC,
                        "concat_channels reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            int64_t channels = 0;
            for (std::size_t i = 0; i < parts.size(); ++i)
            {
                const Value& p = graph_->value(parts[i]);
                SA_THROW_IF(p.shape.rank() != 4,
                            "concat_channels needs rank-4 operands; got rank {}",
                            p.shape.rank());
                SA_THROW_IF(p.layout != TensorLayout::NHWC,
                            "concat_channels reads NHWC; this value is NCHW. Cross with "
                            "layout_convert first (CV-3).");
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

            const ValueID out = declare_like(
                parts[0], Shape({first.shape[0], first.shape[1], first.shape[2], channels}),
                first.dtype, Lifetime::Activation, "concat");

            graph_->add_node(Ops::K_CONCAT_CHANNELS.id,
                             std::vector<ValueID>(parts.begin(), parts.end()), {out});
            return out;
        }

        ValueID GraphBuilder::slice_channels(ValueID x, int64_t offset, int64_t count) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "slice_channels needs a rank-4 input; got rank {}", in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "slice_channels reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(offset < 0 || count <= 0 || offset + count > in.shape[3],
                        "slice_channels [{}, {}) does not fit {} channels",
                        offset, offset + count, in.shape[3]);

            const ValueID out = declare_like(
                x, Shape({in.shape[0], in.shape[1], in.shape[2], count}),
                in.dtype, Lifetime::Activation, "slice");

            Attributes a{};
            a.values[0] = static_cast<double>(offset);
            graph_->add_node(Ops::K_SLICE_CHANNELS.id, {x}, {out}, a);
            return out;
        }

        ValueID GraphBuilder::channel_affine(ValueID x, ValueID gamma, ValueID beta) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "channel_affine needs a rank-4 input; got rank {}", in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "channel_affine reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            for (ValueID p : {gamma, beta})
            {
                const Value& v = graph_->value(p);
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "channel_affine's per-channel parameters must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
            }

            const ValueID out = declare_like(x, in.shape, in.dtype, Lifetime::Activation,
                                             "affine");
            graph_->add_node(Ops::K_CHANNEL_AFFINE.id, {x, gamma, beta}, {out});
            return out;
        }

        ValueID GraphBuilder::sum_channels(ValueID x) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "sum_channels needs a rank-4 input; got rank {}", in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "sum_channels reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            // In NHWC layout, channel sums reduce across rows of a reshaped matrix.
            const int64_t rows = in.shape[0] * in.shape[1] * in.shape[2];
            return sum_rows(reshape(x, Shape({rows, in.shape[3]})));
        }

        BatchNormResult GraphBuilder::batchnorm2d(ValueID x, ValueID gamma, ValueID beta,
                                                  double epsilon) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "batchnorm2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "batchnorm2d reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(epsilon <= 0.0,
                        "batchnorm2d's epsilon must be positive; got {}", epsilon);

            for (ValueID p : {gamma, beta})
            {
                const Value& v = graph_->value(p);
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "batchnorm2d's per-channel parameters must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
                SA_THROW_IF(v.dtype != in.dtype,
                            "batchnorm2d's per-channel parameters must share the input's "
                            "element type; got {} against {}",
                            dtype_name(v.dtype), dtype_name(in.dtype));
            }

            // A declaration grows the value table, so the shape and the type are held by value.
            const Shape image = in.shape;
            const DType dtype = in.dtype;
            const Shape per_channel({image[3]});

            const ValueID y = declare_like(x, image, dtype, Lifetime::Activation, "bn");

            // Normalization statistics are preserved directly to avoid numerical drift.
            const ValueID mean = declare_like(x, per_channel, dtype, Lifetime::Activation,
                                              "bn_saved_mean");
            const ValueID inv_std = declare_like(x, per_channel, dtype, Lifetime::Activation,
                                                 "bn_saved_inv_std");
            const ValueID variance = declare_like(x, per_channel, dtype, Lifetime::Activation,
                                                  "bn_batch_variance");

            Attributes a{};
            a.values[0] = epsilon;
            graph_->add_node(Ops::K_BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std, variance},
                             a);
            return BatchNormResult{y, mean, inv_std, variance};
        }

        BatchNormGrads GraphBuilder::batchnorm2d_backward(ValueID dy, ValueID x, ValueID gamma,
                                                          ValueID saved_mean,
                                                          ValueID saved_inv_std) const
        {
            const Value& gy = graph_->value(dy);
            SA_THROW_IF(gy.shape.rank() != 4,
                        "batchnorm2d_backward needs a rank-4 [N,H,W,C] output gradient; got "
                        "rank {}", gy.shape.rank());
            SA_THROW_IF(gy.layout != TensorLayout::NHWC,
                        "batchnorm2d_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "batchnorm2d_backward needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "batchnorm2d_backward reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            require_same(dy, x, "batchnorm2d_backward");

            for (ValueID p : {gamma, saved_mean, saved_inv_std})
            {
                const Value& v = graph_->value(p);
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "batchnorm2d_backward's per-channel operands must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
            }

            const Shape per_channel({in.shape[3]});
            const ValueID dx = declare_like(x, in.shape, in.dtype, Lifetime::Activation, "bn_dx");
            const ValueID dgamma = declare_like(x, per_channel, in.dtype, Lifetime::Activation,
                                                "bn_dgamma");
            const ValueID dbeta = declare_like(x, per_channel, in.dtype, Lifetime::Activation,
                                               "bn_dbeta");

            graph_->add_node(Ops::K_BATCHNORM2D_BACKWARD.id,
                             {dy, x, gamma, saved_mean, saved_inv_std},
                             {dx, dgamma, dbeta});
            return BatchNormGrads{dx, dgamma, dbeta};
        }

        BatchNormFold GraphBuilder::batchnorm_fold(ValueID gamma, ValueID beta, ValueID mean,
                                                   ValueID variance, double epsilon) const
        {
            // A declaration grows the value table, so gamma's description is held by value.
            const Shape per_channel = graph_->value(gamma).shape;
            const DType dtype = graph_->value(gamma).dtype;
            const TensorLayout layout = graph_->value(gamma).layout;

            SA_THROW_IF(per_channel.rank() != 1,
                        "batchnorm_fold needs a rank-1 [C] gamma; got {}",
                        per_channel.to_string());
            SA_THROW_IF(!is_real_float(dtype),
                        "batchnorm_fold needs a half, float32 or float64 gamma; got {}",
                        dtype_name(dtype));
            SA_THROW_IF(epsilon <= 0.0,
                        "batchnorm_fold's epsilon must be positive; got {}", epsilon);

            const std::pair<ValueID, std::string_view> others[] = {
                {beta, "beta"}, {mean, "mean"}, {variance, "variance"}};

            for (const std::pair<ValueID, std::string_view>& other : others)
            {
                const Value& v = graph_->value(other.first);
                SA_THROW_IF(v.shape != per_channel,
                            "batchnorm_fold needs its {} shaped like gamma, {}; got {}",
                            other.second, per_channel.to_string(), v.shape.to_string());
                SA_THROW_IF(v.dtype != dtype,
                            "batchnorm_fold needs its {} in gamma's element type, {}; got {}",
                            other.second, dtype_name(dtype), dtype_name(v.dtype));
                SA_THROW_IF(v.layout != layout,
                            "batchnorm_fold needs its {} in gamma's layout; cross with "
                            "layout_convert first (CV-3).",
                            other.second);
            }

            const ValueID scale = declare_like(gamma, per_channel, dtype, Lifetime::Activation,
                                               "bn_fold_scale");
            const ValueID shift = declare_like(gamma, per_channel, dtype, Lifetime::Activation,
                                               "bn_fold_shift");

            Attributes a{};
            a.values[0] = epsilon;
            graph_->add_node(Ops::K_BATCHNORM_FOLD.id, {gamma, beta, mean, variance},
                             {scale, shift}, a);
            return BatchNormFold{scale, shift};
        }

        Result<LayerNormResult> GraphBuilder::layer_norm(ValueID x, ValueID gamma, ValueID beta,
                                                         double epsilon) const
        {
            // The description is copied; declaring a result grows the value table.
            const Shape shape = graph_->value(x).shape;
            const DType dtype = graph_->value(x).dtype;

            SA_REFUSE_IF(*this, shape.rank() < 1 || shape.num_elements() < 1,
                         "layer_norm needs a [..., D] input with an element; got {}",
                         shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(dtype),
                         "layer_norm needs a half, float32 or float64 input; got {}",
                         dtype_name(dtype));
            SA_REFUSE_IF(*this, !(epsilon > 0.0),
                         "layer_norm's epsilon must be positive; got {}", epsilon);

            const int64_t width = shape[shape.rank() - 1];
            const std::pair<ValueID, std::string_view> parameters[] = {{gamma, "gamma"},
                                                                       {beta, "beta"}};
            for (const std::pair<ValueID, std::string_view>& parameter : parameters)
            {
                const Value& v = graph_->value(parameter.first);
                SA_REFUSE_IF(*this, v.shape != shape_of(width) || v.dtype != dtype,
                             "layer_norm needs its {} as [{}] of {}; got {} of {}",
                             parameter.second, width, dtype_name(dtype), v.shape.to_string(),
                             dtype_name(v.dtype));
            }

            const Shape per_row = shape_of(shape.num_elements() / width);
            const ValueID y = declare_like(x, shape, dtype, Lifetime::Activation, "ln");
            const ValueID mean = declare_like(x, per_row, dtype, Lifetime::Activation,
                                              "ln_saved_mean");
            const ValueID inv_std = declare_like(x, per_row, dtype, Lifetime::Activation,
                                                 "ln_saved_inv_std");

            Attributes attributes{};
            attributes.values[0] = epsilon;
            graph_->add_node(Ops::K_LAYER_NORM.id, {x, gamma, beta}, {y, mean, inv_std},
                             attributes);
            return LayerNormResult{y, mean, inv_std};
        }

        Result<LayerNormGrads> GraphBuilder::layer_norm_backward(ValueID dy, ValueID x,
                                                                 ValueID gamma,
                                                                 ValueID saved_mean,
                                                                 ValueID saved_inv_std) const
        {
            // The description is copied; declaring a result grows the value table.
            const Shape shape = graph_->value(x).shape;
            const DType dtype = graph_->value(x).dtype;

            SA_REFUSE_IF(*this, shape.rank() < 1 || shape.num_elements() < 1,
                         "layer_norm_backward needs a [..., D] input with an element; got {}",
                         shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(dtype),
                         "layer_norm_backward needs a half, float32 or float64 input; got {}",
                         dtype_name(dtype));

            const int64_t width = shape[shape.rank() - 1];
            const Shape per_row = shape_of(shape.num_elements() / width);
            const std::pair<ValueID, Shape> operands[] = {{dy, shape},
                                                          {gamma, shape_of(width)},
                                                          {saved_mean, per_row},
                                                          {saved_inv_std, per_row}};
            for (const std::pair<ValueID, Shape>& operand : operands)
            {
                const Value& v = graph_->value(operand.first);
                SA_REFUSE_IF(*this, v.shape != operand.second || v.dtype != dtype,
                             "layer_norm_backward over an input of {} needs an operand of {} "
                             "in {}; got {} in {}",
                             shape.to_string(), operand.second.to_string(), dtype_name(dtype),
                             v.shape.to_string(), dtype_name(v.dtype));
            }

            const ValueID dx = declare_like(x, shape, dtype, Lifetime::Activation, "ln_dx");
            const ValueID dgamma = declare_like(x, shape_of(width), dtype, Lifetime::Activation,
                                                "ln_dgamma");
            const ValueID dbeta = declare_like(x, shape_of(width), dtype, Lifetime::Activation,
                                               "ln_dbeta");

            graph_->add_node(Ops::K_LAYER_NORM_BACKWARD.id,
                             {dy, x, gamma, saved_mean, saved_inv_std}, {dx, dgamma, dbeta});
            return LayerNormGrads{dx, dgamma, dbeta};
        }

        std::uint64_t dropout_seed(const Attributes& attributes) noexcept
        {
            const std::uint64_t high =
                static_cast<std::uint64_t>(attributes.get(K_DROPOUT_SEED_HIGH_SLOT));
            const std::uint64_t low =
                static_cast<std::uint64_t>(attributes.get(K_DROPOUT_SEED_LOW_SLOT));
            return (high << K_SEED_HALF_BITS) | low;
        }

        Result<DropoutResult> GraphBuilder::dropout(ValueID x, double rate,
                                                    std::uint64_t seed) const
        {
            // The description is copied; declaring a result grows the value table.
            const Shape shape = graph_->value(x).shape;
            const DType dtype = graph_->value(x).dtype;

            SA_REFUSE_IF(*this, shape.num_elements() < 1,
                         "dropout needs an input with an element; got {}", shape.to_string());
            SA_REFUSE_IF(*this, !is_real_float(dtype),
                         "dropout needs a half, float32 or float64 input; got {}",
                         dtype_name(dtype));
            SA_REFUSE_IF(*this, !(rate > 0.0 && rate < 1.0),
                         "dropout needs a rate in (0, 1); got {}", rate);

            const ValueID y = declare_like(x, shape, dtype, Lifetime::Activation, "dropout");
            const ValueID mask = declare_like(x, shape, dtype, Lifetime::Activation,
                                              "dropout_saved_mask");

            // A double holds 32 bits exactly, so the seed travels as its two halves.
            Attributes attributes{};
            attributes.values[K_DROPOUT_RATE_SLOT] = rate;
            attributes.values[K_DROPOUT_SEED_HIGH_SLOT] =
                static_cast<double>(seed >> K_SEED_HALF_BITS);
            attributes.values[K_DROPOUT_SEED_LOW_SLOT] =
                static_cast<double>(seed & K_SEED_HALF_MASK);
            graph_->add_node(Ops::K_DROPOUT.id, {x}, {y, mask}, attributes);
            return DropoutResult{y, mask};
        }

        ValueID GraphBuilder::reshape(ValueID x, Shape shape) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.num_elements() != shape.num_elements(),
                        "reshape moves no data, so it cannot change the element count: "
                        "{} elements cannot be seen as {}.",
                        in.shape.num_elements(), shape.num_elements());

            const ValueID out = declare_like(x, shape, in.dtype, Lifetime::Activation, "reshape");
            graph_->add_node(Ops::K_RESHAPE.id, {x}, {out});
            return out;
        }

        ValueID GraphBuilder::stop_gradient(ValueID x) const
        {
            return unary(Ops::K_STOP_GRADIENT.id, x);
        }

    } // namespace Graph
} // namespace SushiAI
