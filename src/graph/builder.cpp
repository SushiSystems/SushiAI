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

#include <limits>
#include <utility>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Builds a rank-1 or rank-2 Shape from loose extents. */
            Shape shape_of(int64_t rows) { return Shape{rows}; }
            Shape shape_of(int64_t rows, int64_t cols) { return Shape{rows, cols}; }

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

        ValueId GraphBuilder::declare(Shape shape, DType dtype, Lifetime lifetime,
                                      std::string_view debug_name) const
        {
            return graph_->add_value(shape, dtype, lifetime, debug_name);
        }

        ValueId GraphBuilder::declare_like(ValueId like, Shape shape, DType dtype,
                                           Lifetime lifetime, std::string_view debug_name) const
        {
            const ValueId out = declare(shape, dtype, lifetime, debug_name);
            graph_->value(out).layout = graph_->value(like).layout;
            return out;
        }

        void GraphBuilder::require_same(ValueId a, ValueId b, std::string_view what) const
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

        ValueId GraphBuilder::input(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueId id = declare(shape, dtype, Lifetime::INPUT, debug_name);
            graph_->add_node(Ops::LEAF.id, {}, {id});
            return id;
        }

        ValueId GraphBuilder::parameter(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueId id = declare(shape, dtype, Lifetime::PARAMETER, debug_name);
            graph_->value(id).requires_grad = true;
            graph_->add_node(Ops::LEAF.id, {}, {id});
            return id;
        }

        ValueId GraphBuilder::buffer(Shape shape, std::string_view debug_name, DType dtype) const
        {
            const ValueId id = declare(shape, dtype, Lifetime::BUFFER, debug_name);
            graph_->add_node(Ops::LEAF.id, {}, {id});
            return id;
        }

        ValueId GraphBuilder::like(ValueId source, std::string_view debug_name) const
        {
            const Value& v = graph_->value(source);
            return declare_like(source, v.shape, v.dtype, v.lifetime, debug_name);
        }

        ValueId GraphBuilder::matmul(ValueId a, ValueId b, bool transpose_a, bool transpose_b) const
        {
            const Value& va = graph_->value(a);
            const Value& vb = graph_->value(b);

            SA_THROW_IF(va.shape.rank() != 2 || vb.shape.rank() != 2,
                        "MATMUL needs two rank-2 operands; got {} and {}",
                        va.shape.to_string(), vb.shape.to_string());
            SA_THROW_IF(va.dtype != vb.dtype,
                        "MATMUL needs operands of one element type; got {} and {}",
                        dtype_name(va.dtype), dtype_name(vb.dtype));

            const int64_t m = transpose_a ? va.shape[1] : va.shape[0];
            const int64_t ka = transpose_a ? va.shape[0] : va.shape[1];
            const int64_t kb = transpose_b ? vb.shape[1] : vb.shape[0];
            const int64_t n = transpose_b ? vb.shape[0] : vb.shape[1];

            SA_THROW_IF(ka != kb,
                        "MATMUL contracts {} against {}; the inner extents must agree "
                        "(operands {} and {}, transposed {}/{})",
                        ka, kb, va.shape.to_string(), vb.shape.to_string(),
                        transpose_a, transpose_b);

            // Propagate layout from operand to trigger build-time checks across contractions.
            const ValueId out = declare_like(a, shape_of(m, n), va.dtype, Lifetime::ACTIVATION,
                                             "matmul");

            Attributes attributes{};
            attributes.values[0] = transpose_a ? 1.0 : 0.0;
            attributes.values[1] = transpose_b ? 1.0 : 0.0;

            graph_->add_node(Ops::MATMUL.id, {a, b}, {out}, attributes);
            return out;
        }

        ValueId GraphBuilder::bias_add(ValueId x, ValueId bias) const
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

            const ValueId out = declare_like(x, vx.shape, vx.dtype, Lifetime::ACTIVATION,
                                             "bias_add");
            graph_->add_node(Ops::BIAS_ADD.id, {x, bias}, {out});
            return out;
        }

        ValueId GraphBuilder::sum_rows(ValueId x) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", vx.shape.to_string());

            const ValueId out = declare_like(x, shape_of(vx.shape[1]), vx.dtype,
                                             Lifetime::ACTIVATION, "sum_rows");
            graph_->add_node(Ops::SUM_ROWS.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::elementwise(OpID op, ValueId a, ValueId b,
                                          std::string_view what) const
        {
            require_same(a, b, what);

            const Value& va = graph_->value(a);
            const ValueId out = declare_like(a, va.shape, va.dtype, Lifetime::ACTIVATION,
                                             "elementwise");
            graph_->add_node(op, {a, b}, {out});
            return out;
        }

        ValueId GraphBuilder::add(ValueId a, ValueId b) const
        {
            return elementwise(Ops::ADD.id, a, b, "ADD");
        }

        ValueId GraphBuilder::sub(ValueId a, ValueId b) const
        {
            return elementwise(Ops::SUB.id, a, b, "SUB");
        }

        ValueId GraphBuilder::mul(ValueId a, ValueId b) const
        {
            return elementwise(Ops::MUL.id, a, b, "MUL");
        }

        ValueId GraphBuilder::div(ValueId a, ValueId b) const
        {
            return elementwise(Ops::DIV.id, a, b, "DIV");
        }

        ValueId GraphBuilder::minimum(ValueId a, ValueId b) const
        {
            return elementwise(Ops::MINIMUM.id, a, b, "MINIMUM");
        }

        ValueId GraphBuilder::maximum(ValueId a, ValueId b) const
        {
            return elementwise(Ops::MAXIMUM.id, a, b, "MAXIMUM");
        }

        ValueId GraphBuilder::less_equal(ValueId a, ValueId b) const
        {
            return elementwise(Ops::LESS_EQUAL.id, a, b, "LESS_EQUAL");
        }

        ValueId GraphBuilder::select(ValueId condition, ValueId a, ValueId b) const
        {
            require_same(a, b, "SELECT");
            require_same(condition, a, "SELECT");

            const Value& va = graph_->value(a);
            const ValueId out = declare_like(a, va.shape, va.dtype, Lifetime::ACTIVATION,
                                             "select");
            graph_->add_node(Ops::SELECT.id, {condition, a, b}, {out});
            return out;
        }

        ValueId GraphBuilder::scale(ValueId x, double alpha) const
        {
            const Value& vx = graph_->value(x);
            const ValueId out = declare_like(x, vx.shape, vx.dtype, Lifetime::ACTIVATION, "scale");

            Attributes attributes{};
            attributes.values[0] = alpha;

            graph_->add_node(Ops::SCALE.id, {x}, {out}, attributes);
            return out;
        }

        ValueId GraphBuilder::dynamic_scale(ValueId x, ValueId factor,
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

            const ValueId out = declare_like(x, vx.shape, vx.dtype, Lifetime::ACTIVATION,
                                             debug_name);

            graph_->add_node(Ops::DYNAMIC_SCALE.id, {x, factor}, {out});
            return out;
        }

        ValueId GraphBuilder::unary(OpID op, ValueId x) const
        {
            const Value& vx = graph_->value(x);
            const ValueId out = declare_like(x, vx.shape, vx.dtype, Lifetime::ACTIVATION,
                                             Ops::name_of(op));
            graph_->add_node(op, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::exp(ValueId x) const { return unary(Ops::EXP.id, x); }
        ValueId GraphBuilder::relu(ValueId x) const { return unary(Ops::RELU.id, x); }
        ValueId GraphBuilder::gelu(ValueId x) const { return unary(Ops::GELU.id, x); }
        ValueId GraphBuilder::sigmoid(ValueId x) const { return unary(Ops::SIGMOID.id, x); }
        ValueId GraphBuilder::tanh(ValueId x) const { return unary(Ops::TANH.id, x); }

        ValueId GraphBuilder::abs(ValueId x) const
        {
            return unary(Ops::ABS.id, x);
        }

        ValueId GraphBuilder::atan(ValueId x) const
        {
            return unary(Ops::ATAN.id, x);
        }

        ValueId GraphBuilder::softplus(ValueId x) const
        {
            return unary(Ops::SOFTPLUS.id, x);
        }

        ValueId GraphBuilder::silu(ValueId x) const
        {
            return unary(Ops::SILU.id, x);
        }

        ValueId GraphBuilder::leaky_relu(ValueId x, double slope) const
        {
            SA_THROW_IF(!is_legal_leaky_relu_slope(slope),
                        "leaky_relu's slope must be in (0, 1); got {}. At 0 it is a ReLU "
                        "and at 1 it is the identity, and both would make the "
                        "output-reading gradient (ops.hpp's ACTIVATIONS) unsound.",
                        slope);

            const Value& in = graph_->value(x);
            const ValueId out = declare_like(x, in.shape, in.dtype, Lifetime::ACTIVATION,
                                             "leaky_relu");

            Attributes a{};
            a.values[0] = slope;
            graph_->add_node(Ops::LEAKY_RELU.id, {x}, {out}, a);
            return out;
        }

        ValueId GraphBuilder::zeros(Shape shape, DType dtype) const
        {
            // Zero emitter has no inputs to inherit layout from; consumers use zeros_like.
            const ValueId out = declare(shape, dtype, Lifetime::ACTIVATION, "zeros");
            graph_->add_node(Ops::ZERO.id, {}, {out});
            return out;
        }

        ValueId GraphBuilder::zeros_like(ValueId like, Shape shape, DType dtype) const
        {
            const ValueId out = zeros(shape, dtype);
            graph_->value(out).layout = graph_->value(like).layout;
            return out;
        }

        ValueId GraphBuilder::constant(Shape shape, double value, DType dtype) const
        {
            SA_THROW_IF(!is_real_float(dtype),
                        "CONSTANT fills a real floating tensor (half, float32, float64); got {}",
                        dtype_name(dtype));

            const ValueId out = declare(shape, dtype, Lifetime::ACTIVATION, "constant");

            Attributes attributes{};
            attributes.values[0] = value;

            graph_->add_node(Ops::CONSTANT.id, {}, {out}, attributes);
            return out;
        }

        ValueId GraphBuilder::constant_like(ValueId x, double value) const
        {
            // The description is held by value; constant() grows the table a reference points into.
            const Value described = graph_->value(x);

            const ValueId out = constant(described.shape, value, described.dtype);
            graph_->value(out).layout = described.layout;
            return out;
        }

        ValueId GraphBuilder::activation_backward(OpID op, ValueId dy, ValueId primal,
                                                  const Attributes& attributes) const
        {
            SA_THROW_IF(!Ops::is_activation_backward(op),
                        "{} is not an activation gradient", Ops::name_of(op));

            // Verify slope explicitly to prevent silent emission of standard ReLU gradients.
            if (op == Ops::LEAKY_RELU_BACKWARD.id)
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
            const ValueId out = declare_like(dy, v.shape, v.dtype, Lifetime::ACTIVATION,
                                             Ops::name_of(op));
            graph_->add_node(op, {dy, primal}, {out}, attributes);
            return out;
        }

        ValueId GraphBuilder::log_softmax(ValueId x) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 2,
                        "LOG_SOFTMAX needs a rank-2 operand; got {}", vx.shape.to_string());

            const ValueId out = declare_like(x, vx.shape, vx.dtype, Lifetime::ACTIVATION,
                                             "log_softmax");
            graph_->add_node(Ops::LOG_SOFTMAX.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::nll_loss(ValueId log_probs, ValueId one_hot) const
        {
            require_same(log_probs, one_hot, "NLL_LOSS");

            const Value& v = graph_->value(log_probs);
            SA_THROW_IF(v.shape.rank() != 2,
                        "NLL_LOSS needs rank-2 operands; got {}", v.shape.to_string());

            const ValueId out = declare_like(log_probs, shape_of(1), v.dtype, Lifetime::ACTIVATION,
                                             "loss");
            graph_->add_node(Ops::NLL_LOSS.id, {log_probs, one_hot}, {out});
            return out;
        }

        ValueId GraphBuilder::cast(ValueId x, DType dtype, std::string_view debug_name) const
        {
            const Value& vx = graph_->value(x);

            SA_THROW_IF(!is_real_float(vx.dtype) || !is_real_float(dtype),
                        "CAST converts between the real floating types only (half, float32, "
                        "float64) and was asked for {} -> {}. A complex operand has no single "
                        "real value to narrow to, and a float-to-integer conversion is a "
                        "rounding policy this library is not entitled to choose for a caller.",
                        dtype_name(vx.dtype), dtype_name(dtype));

            const ValueId out = declare_like(x, vx.shape, dtype, Lifetime::ACTIVATION, debug_name);
            graph_->add_node(Ops::CAST.id, {x}, {out});
            return out;
        }

        void GraphBuilder::cast_into(ValueId source, ValueId destination) const
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

            graph_->add_node(Ops::CAST.id, {source}, {destination});
        }

        ValueId GraphBuilder::accumulate(ValueId acc, ValueId delta) const
        {
            require_same(acc, delta, "ACCUMULATE");

            const Value& v = graph_->value(acc);
            const ValueId out = declare_like(acc, v.shape, v.dtype, v.lifetime, "accumulated_grad");
            graph_->add_node(Ops::ACCUMULATE.id, {acc, delta}, {out});
            return out;
        }

        ValueId GraphBuilder::layout_convert(ValueId x, TensorLayout layout) const
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

            const ValueId out = declare(converted, vx.dtype, Lifetime::ACTIVATION,
                                        "layout_convert");
            graph_->value(out).layout = layout;
            graph_->add_node(Ops::LAYOUT_CONVERT.id, {x}, {out});
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

        ValueId GraphBuilder::im2col(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());
            require_nhwc(in, "im2col");

            // Default im2col assumes symmetric padding; asymmetric cases use im2col_with_extents.
            const SpatialExtents out = window_extents(in.shape[1], in.shape[2], g);

            return im2col_with_extents(x, g, out.height, out.width);
        }

        ValueId GraphBuilder::im2col_with_extents(ValueId x, const ConvGeometry& g,
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

            const ValueId out = declare_like(x, shape_of(rows, cols), in.dtype,
                                             Lifetime::ACTIVATION, "cols");
            graph_->add_node(Ops::IM2COL.id, {x}, {out}, windowed_attributes(g, out_h, out_w));
            return out;
        }

        ValueId GraphBuilder::col2im(ValueId cols, Shape image_shape,
                                     const ConvGeometry& g) const
        {
            SA_THROW_IF(image_shape.rank() != 4,
                        "col2im returns a rank-4 [N,H,W,C] image; got rank {}",
                        image_shape.rank());

            // Default col2im assumes symmetric padding; backward passes use col2im_with_extents.
            const SpatialExtents out = window_extents(image_shape[1], image_shape[2], g);

            return col2im_with_extents(cols, image_shape, g, out.height, out.width);
        }

        ValueId GraphBuilder::col2im_with_extents(ValueId cols, Shape image_shape,
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

            const ValueId out = declare_like(cols, image_shape, c.dtype,
                                             Lifetime::ACTIVATION, "image");
            graph_->add_node(Ops::COL2IM.id, {cols}, {out}, windowed_attributes(g, out_h, out_w));
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

        ValueId GraphBuilder::conv2d(ValueId x, ValueId weight, ValueId bias,
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

            if (bias != NO_VALUE)
            {
                const Shape& offsets = graph_->value(bias).shape;
                SA_THROW_IF(offsets.rank() != 1 || offsets[0] != filter[0],
                            "conv2d with {} filters needs a [{}] bias; got {}",
                            filter[0], filter[0], offsets.to_string());
            }

            const SpatialExtents out = window_extents(image[1], image[2], g);

            // Row-major [C_out,R,S,C_in] flattens to (r * S + s) * C_in + c, IM2COL's column order.
            const int64_t taps = g.kernel_h * g.kernel_w * image[3];
            const ValueId filters = reshape(weight, Shape({filter[0], taps}));

            const ValueId cols = is_pointwise(g)
                ? reshape(x, Shape({image[0] * image[1] * image[2], image[3]}))
                : im2col_with_extents(x, g, out.height, out.width);

            const ValueId product = matmul(cols, filters, false, true);
            const ValueId rows = bias == NO_VALUE ? product : bias_add(product, bias);

            return reshape(rows, Shape({image[0], out.height, out.width, filter[0]}));
        }

        PoolResult GraphBuilder::maxpool2d(ValueId x, const ConvGeometry& g) const
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

        PoolResult GraphBuilder::maxpool2d_with_extents(ValueId x, const ConvGeometry& g,
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
            const ValueId out = declare_like(x, pooled, in.dtype, Lifetime::ACTIVATION, "pooled");
            const ValueId idx = declare_like(x, pooled, DType::INT64, Lifetime::ACTIVATION,
                                             "argmax");

            graph_->add_node(Ops::MAXPOOL2D.id, {x}, {out, idx}, geometry_attributes(g));
            return PoolResult{out, idx};
        }

        ValueId GraphBuilder::avgpool2d(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "avgpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "avgpool2d");

            const SpatialExtents pooled = window_extents(in.shape[1], in.shape[2], g);

            const ValueId out = declare_like(
                x, Shape({in.shape[0], pooled.height, pooled.width, in.shape[3]}),
                in.dtype, Lifetime::ACTIVATION, "pooled");
            graph_->add_node(Ops::AVGPOOL2D.id, {x}, {out}, geometry_attributes(g));
            return out;
        }

        ValueId GraphBuilder::upsample2d(ValueId x, int64_t scale_h, int64_t scale_w) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "upsample2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            require_nhwc(in, "upsample2d");

            const SpatialExtents scaled = upsampled_extents(in.shape[1], in.shape[2],
                                                            scale_h, scale_w);

            const ValueId out = declare_like(
                x, Shape({in.shape[0], scaled.height, scaled.width, in.shape[3]}),
                in.dtype, Lifetime::ACTIVATION, "upsampled");

            Attributes a{};
            a.values[0] = static_cast<double>(scale_h);
            a.values[1] = static_cast<double>(scale_w);
            graph_->add_node(Ops::UPSAMPLE2D.id, {x}, {out}, a);
            return out;
        }

        ValueId GraphBuilder::spatial_mean(ValueId x) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "spatial_mean needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "spatial_mean reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            const ValueId out = declare_like(x, Shape({in.shape[0], in.shape[3]}), in.dtype,
                                             Lifetime::ACTIVATION, "spatial_mean");
            graph_->add_node(Ops::SPATIAL_MEAN.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::maxpool2d_backward(ValueId dy, ValueId indices, Shape input_shape,
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
            SA_THROW_IF(ix.dtype != DType::INT64,
                        "maxpool2d_backward's indices must be INT64; got {}",
                        dtype_name(ix.dtype));
            SA_THROW_IF(ix.shape != gy.shape,
                        "maxpool2d_backward's indices must share its output gradient's shape");

            const ValueId out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::ACTIVATION, "dx");
            graph_->add_node(Ops::MAXPOOL2D_BACKWARD.id, {dy, indices}, {out},
                             geometry_attributes(g));
            return out;
        }

        ValueId GraphBuilder::avgpool2d_backward(ValueId dy, Shape input_shape,
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

            const ValueId out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::ACTIVATION, "dx");
            graph_->add_node(Ops::AVGPOOL2D_BACKWARD.id, {dy}, {out}, geometry_attributes(g));
            return out;
        }

        ValueId GraphBuilder::upsample2d_backward(ValueId dy, Shape input_shape,
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

            const ValueId out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::ACTIVATION, "dx");

            Attributes a{};
            a.values[0] = static_cast<double>(scale_h);
            a.values[1] = static_cast<double>(scale_w);
            graph_->add_node(Ops::UPSAMPLE2D_BACKWARD.id, {dy}, {out}, a);
            return out;
        }

        ValueId GraphBuilder::spatial_mean_backward(ValueId dy, Shape input_shape) const
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

            const ValueId out = declare_like(dy, input_shape, gy.dtype,
                                             Lifetime::ACTIVATION, "dx");
            graph_->add_node(Ops::SPATIAL_MEAN_BACKWARD.id, {dy}, {out});
            return out;
        }

        ValueId GraphBuilder::concat_channels(SushiRuntime::span<const ValueId> parts) const
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

            const ValueId out = declare_like(
                parts[0], Shape({first.shape[0], first.shape[1], first.shape[2], channels}),
                first.dtype, Lifetime::ACTIVATION, "concat");

            graph_->add_node(Ops::CONCAT_CHANNELS.id,
                             std::vector<ValueId>(parts.begin(), parts.end()), {out});
            return out;
        }

        ValueId GraphBuilder::slice_channels(ValueId x, int64_t offset, int64_t count) const
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

            const ValueId out = declare_like(
                x, Shape({in.shape[0], in.shape[1], in.shape[2], count}),
                in.dtype, Lifetime::ACTIVATION, "slice");

            Attributes a{};
            a.values[0] = static_cast<double>(offset);
            graph_->add_node(Ops::SLICE_CHANNELS.id, {x}, {out}, a);
            return out;
        }

        ValueId GraphBuilder::channel_affine(ValueId x, ValueId gamma, ValueId beta) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "channel_affine needs a rank-4 input; got rank {}", in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "channel_affine reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            for (ValueId p : {gamma, beta})
            {
                const Value& v = graph_->value(p);
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "channel_affine's per-channel parameters must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
            }

            const ValueId out = declare_like(x, in.shape, in.dtype, Lifetime::ACTIVATION,
                                             "affine");
            graph_->add_node(Ops::CHANNEL_AFFINE.id, {x, gamma, beta}, {out});
            return out;
        }

        ValueId GraphBuilder::sum_channels(ValueId x) const
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

        BatchNormResult GraphBuilder::batchnorm2d(ValueId x, ValueId gamma, ValueId beta,
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

            for (ValueId p : {gamma, beta})
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

            const ValueId y = declare_like(x, image, dtype, Lifetime::ACTIVATION, "bn");

            // Normalization statistics are preserved directly to avoid numerical drift.
            const ValueId mean = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                              "bn_saved_mean");
            const ValueId inv_std = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                                 "bn_saved_inv_std");
            const ValueId variance = declare_like(x, per_channel, dtype, Lifetime::ACTIVATION,
                                                  "bn_batch_variance");

            Attributes a{};
            a.values[0] = epsilon;
            graph_->add_node(Ops::BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std, variance},
                             a);
            return BatchNormResult{y, mean, inv_std, variance};
        }

        BatchNormGrads GraphBuilder::batchnorm2d_backward(ValueId dy, ValueId x, ValueId gamma,
                                                          ValueId saved_mean,
                                                          ValueId saved_inv_std) const
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

            for (ValueId p : {gamma, saved_mean, saved_inv_std})
            {
                const Value& v = graph_->value(p);
                SA_THROW_IF(v.shape.rank() != 1 || v.shape[0] != in.shape[3],
                            "batchnorm2d_backward's per-channel operands must be [{}]; got a "
                            "rank-{} value", in.shape[3], v.shape.rank());
            }

            const Shape per_channel({in.shape[3]});
            const ValueId dx = declare_like(x, in.shape, in.dtype, Lifetime::ACTIVATION, "bn_dx");
            const ValueId dgamma = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                                "bn_dgamma");
            const ValueId dbeta = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                               "bn_dbeta");

            graph_->add_node(Ops::BATCHNORM2D_BACKWARD.id,
                             {dy, x, gamma, saved_mean, saved_inv_std},
                             {dx, dgamma, dbeta});
            return BatchNormGrads{dx, dgamma, dbeta};
        }

        BatchNormFold GraphBuilder::batchnorm_fold(ValueId gamma, ValueId beta, ValueId mean,
                                                   ValueId variance, double epsilon) const
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

            const std::pair<ValueId, std::string_view> others[] = {
                {beta, "beta"}, {mean, "mean"}, {variance, "variance"}};

            for (const std::pair<ValueId, std::string_view>& other : others)
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

            const ValueId scale = declare_like(gamma, per_channel, dtype, Lifetime::ACTIVATION,
                                               "bn_fold_scale");
            const ValueId shift = declare_like(gamma, per_channel, dtype, Lifetime::ACTIVATION,
                                               "bn_fold_shift");

            Attributes a{};
            a.values[0] = epsilon;
            graph_->add_node(Ops::BATCHNORM_FOLD.id, {gamma, beta, mean, variance},
                             {scale, shift}, a);
            return BatchNormFold{scale, shift};
        }

        ValueId GraphBuilder::reshape(ValueId x, Shape shape) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.num_elements() != shape.num_elements(),
                        "reshape moves no data, so it cannot change the element count: "
                        "{} elements cannot be seen as {}.",
                        in.shape.num_elements(), shape.num_elements());

            const ValueId out = declare_like(x, shape, in.dtype, Lifetime::ACTIVATION, "reshape");
            graph_->add_node(Ops::RESHAPE.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::stop_gradient(ValueId x) const
        {
            return unary(Ops::STOP_GRADIENT.id, x);
        }

    } // namespace Graph
} // namespace SushiAI
