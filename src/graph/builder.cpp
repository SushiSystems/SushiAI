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
        }

        ValueId GraphBuilder::declare(Shape shape, DType dtype, Lifetime lifetime,
                                      std::string_view debug_name) const
        {
            return graph_->add_value(shape, dtype, lifetime, debug_name);
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

        ValueId GraphBuilder::zeros(Shape shape, DType dtype) const
        {
            const ValueId out = declare(shape, dtype, Lifetime::ACTIVATION, "zeros");
            graph_->add_node(Ops::ZERO.id, {}, {out});
            return out;
        }

        ValueId GraphBuilder::activation_backward(OpID op, ValueId dy, ValueId primal) const
        {
            SA_THROW_IF(op != Ops::RELU_BACKWARD.id && op != Ops::GELU_BACKWARD.id
                        && op != Ops::SIGMOID_BACKWARD.id && op != Ops::TANH_BACKWARD.id,
                        "{} is not an activation gradient", Ops::name_of(op));

            require_same(dy, primal, Ops::name_of(op));

            const Value& v = graph_->value(dy);
            const ValueId out = declare(v.shape, v.dtype, Lifetime::ACTIVATION,
                                        Ops::name_of(op));
            graph_->add_node(op, {dy, primal}, {out});
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

    } // namespace Graph
} // namespace SushiAI
