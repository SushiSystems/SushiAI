/**************************************************************************/
/* builder.cpp                                                            */
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

#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Builds a rank-1 or rank-2 Shape from loose extents. */
            Shape shape_of(int64_t rows) { return Shape{rows}; }
            Shape shape_of(int64_t rows, int64_t cols) { return Shape{rows, cols}; }

            /**
             * @brief The one definition of a legal LeakyReLU slope.
             *
             * Asked for by both leaky_relu() and activation_backward(), which
             * must agree: a slope the forward emitter accepts and the backward
             * one rejects is a graph that builds and cannot be differentiated,
             * and the reverse is a plain-ReLU gradient emitted silently. Two
             * copies of the predicate is how that disagreement arrives, so
             * there is one. The messages stay separate - they explain
             * different mistakes - but the rule does not.
             *
             * At 0 it is a ReLU and at 1 the identity, and both make the
             * output-reading gradient (ops.hpp's ACTIVATIONS) unsound.
             */
            bool is_legal_leaky_relu_slope(double slope)
            {
                return slope > 0.0 && slope < 1.0;
            }
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

        ValueId GraphBuilder::like(ValueId source, std::string_view debug_name) const
        {
            const Value& v = graph_->value(source);
            return declare(v.shape, v.dtype, v.lifetime, debug_name);
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

            const ValueId out = declare(shape_of(m, n), va.dtype, Lifetime::ACTIVATION, "matmul");

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

            const ValueId out = declare(vx.shape, vx.dtype, Lifetime::ACTIVATION, "bias_add");
            graph_->add_node(Ops::BIAS_ADD.id, {x, bias}, {out});
            return out;
        }

        ValueId GraphBuilder::sum_rows(ValueId x) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", vx.shape.to_string());

            const ValueId out = declare(shape_of(vx.shape[1]), vx.dtype,
                                        Lifetime::ACTIVATION, "sum_rows");
            graph_->add_node(Ops::SUM_ROWS.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::elementwise(OpID op, ValueId a, ValueId b, std::string_view what) const
        {
            require_same(a, b, what);

            const Value& va = graph_->value(a);
            const ValueId out = declare(va.shape, va.dtype, Lifetime::ACTIVATION, "elementwise");
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

        ValueId GraphBuilder::scale(ValueId x, double alpha) const
        {
            const Value& vx = graph_->value(x);
            const ValueId out = declare(vx.shape, vx.dtype, Lifetime::ACTIVATION, "scale");

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

            const ValueId out = declare(vx.shape, vx.dtype, Lifetime::ACTIVATION, debug_name);

            graph_->add_node(Ops::DYNAMIC_SCALE.id, {x, factor}, {out});
            return out;
        }

        ValueId GraphBuilder::unary(OpID op, ValueId x) const
        {
            const Value& vx = graph_->value(x);
            const ValueId out = declare(vx.shape, vx.dtype, Lifetime::ACTIVATION,
                                        Ops::name_of(op));
            graph_->add_node(op, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::exp(ValueId x) const { return unary(Ops::EXP.id, x); }
        ValueId GraphBuilder::relu(ValueId x) const { return unary(Ops::RELU.id, x); }
        ValueId GraphBuilder::gelu(ValueId x) const { return unary(Ops::GELU.id, x); }
        ValueId GraphBuilder::sigmoid(ValueId x) const { return unary(Ops::SIGMOID.id, x); }
        ValueId GraphBuilder::tanh(ValueId x) const { return unary(Ops::TANH.id, x); }

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
            const ValueId out = declare(shape, dtype, Lifetime::ACTIVATION, "zeros");
            graph_->add_node(Ops::ZERO.id, {}, {out});
            return out;
        }

        ValueId GraphBuilder::activation_backward(OpID op, ValueId dy, ValueId primal,
                                                  const Attributes& attributes) const
        {
            SA_THROW_IF(op != Ops::RELU_BACKWARD.id && op != Ops::GELU_BACKWARD.id
                        && op != Ops::SIGMOID_BACKWARD.id && op != Ops::TANH_BACKWARD.id
                        && op != Ops::LEAKY_RELU_BACKWARD.id,
                        "{} is not an activation gradient", Ops::name_of(op));

            // A default-constructed Attributes reads as slope 0 in slot 0,
            // which is a legal-looking double but an illegal slope: silently
            // running LEAKY_RELU_BACKWARD at slope 0 is a plain-ReLU
            // gradient, not the one the caller asked for. The predicate is
            // shared with leaky_relu()'s own guard, so the two cannot
            // disagree about what a legal slope is.
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

            // declare_like, not declare: a gradient has its primal's layout.
            // Rank-2 activations make the two indistinguishable today, but a
            // spatial dy carries NHWC and a default-layout dx would make
            // finalise()'s uniformity proof fail - or worse, pass while the
            // consumer reads the wrong axis order (A-2).
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

            const ValueId out = declare(vx.shape, vx.dtype, Lifetime::ACTIVATION, "log_softmax");
            graph_->add_node(Ops::LOG_SOFTMAX.id, {x}, {out});
            return out;
        }

        ValueId GraphBuilder::nll_loss(ValueId log_probs, ValueId one_hot) const
        {
            require_same(log_probs, one_hot, "NLL_LOSS");

            const Value& v = graph_->value(log_probs);
            SA_THROW_IF(v.shape.rank() != 2,
                        "NLL_LOSS needs rank-2 operands; got {}", v.shape.to_string());

            const ValueId out = declare(shape_of(1), v.dtype, Lifetime::ACTIVATION, "loss");
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

            const ValueId out = declare(vx.shape, dtype, Lifetime::ACTIVATION, debug_name);
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
            const ValueId out = declare(v.shape, v.dtype, v.lifetime, "accumulated_grad");
            graph_->add_node(Ops::ACCUMULATE.id, {acc, delta}, {out});
            return out;
        }

        ValueId GraphBuilder::layout_convert(ValueId x, TensorLayout layout) const
        {
            const Value& vx = graph_->value(x);
            SA_THROW_IF(vx.shape.rank() != 4,
                        "LAYOUT_CONVERT needs a rank-4 operand; got {}", vx.shape.to_string());

            // NHWC's [N,H,W,C] is NCHW's [N,C,H,W]; permute whichever direction
            // this conversion actually crosses, based on the operand's own
            // layout rather than the target alone.
            const Shape converted = (vx.layout == TensorLayout::NHWC && layout == TensorLayout::NCHW)
                ? Shape{vx.shape[0], vx.shape[3], vx.shape[1], vx.shape[2]}
                : (vx.layout == TensorLayout::NCHW && layout == TensorLayout::NHWC)
                    ? Shape{vx.shape[0], vx.shape[2], vx.shape[3], vx.shape[1]}
                    : vx.shape;

            const ValueId out = declare(converted, vx.dtype, Lifetime::ACTIVATION, "layout_convert");
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
             * @brief CV-5's eight slots plus the output extents in slots 8-9.
             *
             * IM2COL and COL2IM carry their window grid rather than letting
             * the lowering re-derive it, because a re-derivation has to assume
             * symmetric padding and an asymmetric grid can derive to its own
             * transpose with the same window count - which every shape check
             * accepts.
             */
            Attributes windowed_attributes(const ConvGeometry& g,
                                           int64_t out_h, int64_t out_w) noexcept
            {
                Attributes a = geometry_attributes(g);
                a.values[8] = static_cast<double>(out_h);
                a.values[9] = static_cast<double>(out_w);
                return a;
            }
        }

        ValueId GraphBuilder::im2col(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
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
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "im2col needs a rank-4 [N,H,W,C] input; got rank {}", in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "im2col reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
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
            // Symmetric padding is assumed here and stated in the header. A
            // backward rule that knows the forward's actual window counts -
            // which it always does, because the forward wrote them into the
            // graph - calls col2im_with_extents instead.
            const int64_t out_h = output_extent(image_shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(image_shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);

            return col2im_with_extents(cols, image_shape, g, out_h, out_w);
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

        PoolResult GraphBuilder::maxpool2d(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());

            // Symmetric padding is assumed here and stated in the header. A
            // layer that pads asymmetrically calls maxpool2d_with_extents.
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
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "maxpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "maxpool2d reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "maxpool2d was given non-positive output extents {}x{}",
                        out_h, out_w);

            const Shape pooled({in.shape[0], out_h, out_w, in.shape[3]});
            const ValueId out = declare_like(x, pooled, in.dtype, Lifetime::ACTIVATION, "pooled");
            const ValueId idx = declare_like(x, pooled, DType::INT64, Lifetime::ACTIVATION, "argmax");

            graph_->add_node(Ops::MAXPOOL2D.id, {x}, {out, idx}, geometry_attributes(g));
            return PoolResult{out, idx};
        }

        ValueId GraphBuilder::avgpool2d(ValueId x, const ConvGeometry& g) const
        {
            const Value& in = graph_->value(x);
            SA_THROW_IF(in.shape.rank() != 4,
                        "avgpool2d needs a rank-4 [N,H,W,C] input; got rank {}",
                        in.shape.rank());
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "avgpool2d reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");

            const int64_t out_h = output_extent(in.shape[1], g.pad_top, g.pad_top,
                                                g.dilation_h, g.kernel_h, g.stride_h);
            const int64_t out_w = output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                g.dilation_w, g.kernel_w, g.stride_w);
            SA_THROW_IF(out_h <= 0 || out_w <= 0,
                        "A {}x{} pool window does not fit a {}x{} input padded by {},{}.",
                        g.kernel_h, g.kernel_w, in.shape[1], in.shape[2],
                        g.pad_top, g.pad_left);

            const ValueId out = declare_like(x, Shape({in.shape[0], out_h, out_w, in.shape[3]}),
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
            SA_THROW_IF(in.layout != TensorLayout::NHWC,
                        "upsample2d reads NHWC; this value is NCHW. Cross with "
                        "layout_convert first (CV-3).");
            SA_THROW_IF(scale_h < 1 || scale_w < 1,
                        "upsample2d needs scales of at least one; got {}x{}",
                        scale_h, scale_w);

            const ValueId out = declare_like(x, Shape({in.shape[0], in.shape[1] * scale_h,
                                                        in.shape[2] * scale_w, in.shape[3]}),
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
            SA_THROW_IF(scale_h < 1 || scale_w < 1,
                        "upsample2d_backward needs scales of at least one; got {}x{}",
                        scale_h, scale_w);

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

            // Channels innermost is what makes this two existing nodes rather
            // than a new one: [N,H,W,C] and [N*H*W, C] are the same bytes in
            // the same order, so the per-channel sum is a column sum of that
            // view and SUM_ROWS already is one.
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

            const Shape per_channel({in.shape[3]});
            const ValueId y = declare_like(x, in.shape, in.dtype, Lifetime::ACTIVATION, "bn");

            // Saved rather than recomputed: the backward needs exactly the
            // statistics the forward used, and recomputing them from a value
            // some later node may have overwritten is how a normalisation
            // silently drifts.
            const ValueId mean = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                              "bn_saved_mean");
            const ValueId inv_std = declare_like(x, per_channel, in.dtype, Lifetime::ACTIVATION,
                                                 "bn_saved_inv_std");

            Attributes a{};
            a.values[0] = epsilon;
            graph_->add_node(Ops::BATCHNORM2D.id, {x, gamma, beta}, {y, mean, inv_std}, a);
            return BatchNormResult{y, mean, inv_std};
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

    } // namespace Graph
} // namespace SushiAI
