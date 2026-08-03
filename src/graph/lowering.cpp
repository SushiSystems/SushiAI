/**************************************************************************/
/* lowering.cpp                                                           */
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

#include <span>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/graph/lowering.hpp>

namespace SushiAI
{
    namespace Graph
    {
        // =====================================================================
        // TensorPool
        // =====================================================================

        SushiBLAS::Tensor TensorPool::create(const Shape& shape, DType dtype) const
        {
            return allocate_tensor(*engine_, shape, dtype);
        }

        void TensorPool::bind(ValueId id, SushiBLAS::Tensor tensor)
        {
            SA_THROW_IF(is_bound(id),
                        "Value {} is already bound to a tensor; a value has one buffer", id);
            tensors_.emplace(id, std::move(tensor));
        }

        SushiBLAS::Tensor& TensorPool::tensor(ValueId id)
        {
            const auto it = tensors_.find(id);
            SA_THROW_IF(it == tensors_.end(),
                        "Value {} has no tensor; bind or materialise it before lowering", id);
            return it->second;
        }

        const SushiBLAS::Tensor& TensorPool::tensor(ValueId id) const
        {
            const auto it = tensors_.find(id);
            SA_THROW_IF(it == tensors_.end(),
                        "Value {} has no tensor; bind or materialise it before lowering", id);
            return it->second;
        }

        SushiBLAS::Tensor& TensorPool::materialise(ValueId id, const Value& value)
        {
            const auto it = tensors_.find(id);
            if (it != tensors_.end()) return it->second;

            allocated_bytes_ += static_cast<std::size_t>(value.shape.num_elements())
                              * dtype_size(value.dtype);

            return tensors_.emplace(id, create(value.shape, value.dtype)).first->second;
        }

        void TensorPool::materialise_leaves(const Graph& graph)
        {
            for (const Node& node : graph.nodes())
            {
                if (node.op != Ops::LEAF.id) continue;
                for (ValueId out : node.outputs)
                    materialise(out, graph.value(out));
            }
        }

        void TensorPool::alias(ValueId target, ValueId source)
        {
            const auto it = tensors_.find(source);
            SA_THROW_IF(it == tensors_.end(),
                        "Cannot alias value {} onto unbound value {}", target, source);
            SA_THROW_IF(is_bound(target),
                        "Value {} is already bound; it cannot also alias value {}", target, source);
            tensors_.emplace(target, it->second);
        }

        SushiBLAS::Tensor& TensorPool::ones(const Shape& shape, DType dtype)
        {
            const ShapeKey key{shape.hash(), static_cast<std::int32_t>(dtype)};

            const auto it = ones_.find(key);
            if (it != ones_.end()) return it->second;

            allocated_bytes_ += static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype);

            SushiBLAS::Tensor& slot = ones_.emplace(key, create(shape, dtype)).first->second;
            request_constant(slot, 1.0);
            return slot;
        }

        SushiBLAS::Tensor& TensorPool::scratch(NodeId node, int slot,
                                               const Shape& shape, DType dtype)
        {
            const ScratchKey key{node, slot};

            const auto it = scratch_.find(key);
            if (it != scratch_.end()) return it->second;

            allocated_bytes_ += static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype);

            return scratch_.emplace(key, create(shape, dtype)).first->second;
        }

        void TensorPool::request_constant(SushiBLAS::Tensor& tensor, double value)
        {
            pending_constants_.emplace_back(&tensor, value);
        }

        void TensorPool::flush_constants()
        {
            if (pending_constants_.empty()) return;

            for (const auto& [tensor, value] : pending_constants_)
                engine_->random().constant(*tensor, value);

            pending_constants_.clear();

            engine_->execute();
            engine_->get_context().wait_all();
        }

        std::size_t TensorPool::tensor_count() const noexcept
        {
            return tensors_.size() + ones_.size() + scratch_.size();
        }

        // =====================================================================
        // Lowering
        // =====================================================================

        namespace
        {
            /** @brief Reads a node's operand shape straight off the graph. */
            const Shape& shape_of(const Graph& graph, ValueId id)
            {
                return graph.value(id).shape;
            }

            /** @brief Reads a node's operand element type straight off the graph. */
            DType dtype_of(const Graph& graph, ValueId id)
            {
                return graph.value(id).dtype;
            }
        }

        const SushiBLAS::Tensor& Lowering::input(const Node& node, std::size_t index) const
        {
            SA_THROW_IF(index >= node.inputs.size(),
                        "{} expects at least {} operands but was given {}",
                        Ops::name_of(node.op), index + 1, node.inputs.size());
            return pool_->tensor(node.inputs[index]);
        }

        SushiBLAS::Tensor& Lowering::output(const Graph& graph, const Node& node, std::size_t index)
        {
            SA_THROW_IF(index >= node.outputs.size(),
                        "{} expects at least {} results but declares {}",
                        Ops::name_of(node.op), index + 1, node.outputs.size());

            const ValueId id = node.outputs[index];
            return pool_->materialise(id, graph.value(id));
        }

        void Lowering::record(const Graph& graph)
        {
            emitted_ = 0;
            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
                record_node(graph, id);
        }

        void Lowering::record_node(const Graph& graph, NodeId id)
        {
            const Node& node = graph.node(id);

            if (node.op == Ops::LEAF.id)
            {
                // A leaf is written from outside the step. Refusing an unbound
                // one is the difference between a clear failure here and a run
                // that trains on whatever the allocator happened to return.
                for (ValueId out : node.outputs)
                    SA_THROW_IF(!pool_->is_bound(out),
                                "Leaf value {} ('{}') has no tensor bound. Bind it, or call "
                                "TensorPool::materialise_leaves() before lowering.",
                                out, graph.value(out).debug_name);
                return;
            }

            if (node.op == Ops::MATMUL.id)                 { record_matmul(graph, id, node); return; }
            if (node.op == Ops::BIAS_ADD.id)               { record_bias_add(graph, id, node); return; }
            if (node.op == Ops::ADD.id
                || node.op == Ops::SUB.id
                || node.op == Ops::MUL.id)                 { record_binary(graph, node); return; }
            if (node.op == Ops::SCALE.id)                  { record_scale(graph, node); return; }
            if (node.op == Ops::EXP.id
                || node.op == Ops::RELU.id
                || node.op == Ops::GELU.id
                || node.op == Ops::SIGMOID.id
                || node.op == Ops::TANH.id)                { record_unary_inplace(graph, node); return; }
            if (node.op == Ops::RELU_BACKWARD.id
                || node.op == Ops::GELU_BACKWARD.id
                || node.op == Ops::SIGMOID_BACKWARD.id
                || node.op == Ops::TANH_BACKWARD.id)       { record_activation_backward(graph, node); return; }
            if (node.op == Ops::LOG_SOFTMAX.id)            { record_log_softmax(graph, id, node); return; }
            if (node.op == Ops::NLL_LOSS.id)               { record_nll_loss(graph, id, node); return; }
            if (node.op == Ops::SUM_ROWS.id)               { record_sum_rows(graph, node); return; }
            if (node.op == Ops::ACCUMULATE.id)             { record_accumulate(node); return; }
            if (node.op == Ops::ZERO.id)                   { record_zero(graph, node); return; }

            SA_THROW_IF(true,
                        "No lowering for operation {} (node {}). Add it to Lowering::record_node "
                        "or remove it from the graph.",
                        Ops::name_of(node.op), id);
        }

        void Lowering::record_matmul(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "MATMUL (node {}) takes two operands and one result", id);

            const SushiBLAS::Tensor& a = input(node, 0);
            const SushiBLAS::Tensor& b = input(node, 1);
            SushiBLAS::Tensor& c = output(graph, node, 0);

            engine_->blas().gemm(a, b, c,
                                 node.attributes.flag(0), node.attributes.flag(1), 1.0, 0.0);
            ++emitted_;
        }

        void Lowering::record_bias_add(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "BIAS_ADD (node {}) takes two operands and one result", id);

            const Shape& x_shape = shape_of(graph, node.inputs[0]);
            SA_THROW_IF(x_shape.rank() != 2,
                        "BIAS_ADD (node {}) needs a rank-2 left operand; got {}",
                        id, x_shape.to_string());

            const SushiBLAS::Tensor& x = input(node, 0);
            const SushiBLAS::Tensor& bias = input(node, 1);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            SushiBLAS::Tensor& rows = pool_->ones(Shape{x_shape[0]}, dtype_of(graph, node.inputs[0]));

            // Y = X, then a rank-1 update adds the bias row to every row. The
            // copy and the update are both writes to Y, so the runtime's
            // write-after-write edge orders them without an explicit barrier.
            engine_->blas().copy(x, y);
            engine_->blas().ger(rows, bias, y, 1.0f);
            emitted_ += 2;
        }

        void Lowering::record_binary(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& a = input(node, 0);
            const SushiBLAS::Tensor& b = input(node, 1);
            SushiBLAS::Tensor& c = output(graph, node, 0);

            if (node.op == Ops::ADD.id)      engine_->elementwise().add(a, b, c);
            else if (node.op == Ops::SUB.id) engine_->elementwise().sub(a, b, c);
            else                             engine_->elementwise().mul(a, b, c);

            ++emitted_;
        }

        void Lowering::record_scale(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            engine_->blas().copy(x, y);
            engine_->blas().scal(static_cast<float>(node.attributes.get(0)), y);
            emitted_ += 2;
        }

        void Lowering::record_unary_inplace(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            // Copy first so the operation stays out of place at the IR level:
            // the forward input survives for whichever backward rule needs it,
            // which is exactly what makes activations differentiable without a
            // separate save-for-backward mechanism.
            engine_->blas().copy(x, y);

            if (node.op == Ops::EXP.id)          engine_->elementwise().exp(y);
            else if (node.op == Ops::RELU.id)    engine_->nonlinear().relu(y);
            else if (node.op == Ops::GELU.id)    engine_->nonlinear().gelu(y);
            else if (node.op == Ops::SIGMOID.id) engine_->nonlinear().sigmoid(y);
            else                                 engine_->nonlinear().tanh(y);

            emitted_ += 2;
        }

        void Lowering::record_activation_backward(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& primal = input(node, 1);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            if (node.op == Ops::RELU_BACKWARD.id)         engine_->nonlinear().relu_backward(dy, primal, dx);
            else if (node.op == Ops::GELU_BACKWARD.id)    engine_->nonlinear().gelu_backward(dy, primal, dx);
            else if (node.op == Ops::SIGMOID_BACKWARD.id) engine_->nonlinear().sigmoid_backward(dy, primal, dx);
            else                                          engine_->nonlinear().tanh_backward(dy, primal, dx);

            ++emitted_;
        }

        void Lowering::record_log_softmax(const Graph& graph, NodeId id, const Node& node)
        {
            const ValueId in = node.inputs.at(0);
            const Shape& x_shape = shape_of(graph, in);
            SA_THROW_IF(x_shape.rank() != 2,
                        "LOG_SOFTMAX (node {}) needs a rank-2 operand; got {}",
                        id, x_shape.to_string());

            const int64_t rows = x_shape[0];
            const int64_t cols = x_shape[1];
            const DType dtype = dtype_of(graph, in);

            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            SushiBLAS::Tensor& peak      = pool_->scratch(id, 0, Shape{1}, dtype);
            SushiBLAS::Tensor& peak_row  = pool_->scratch(id, 1, Shape{cols}, dtype);
            SushiBLAS::Tensor& shifted   = pool_->scratch(id, 2, x_shape, dtype);
            SushiBLAS::Tensor& exponents = pool_->scratch(id, 3, x_shape, dtype);
            SushiBLAS::Tensor& row_sums  = pool_->scratch(id, 4, Shape{rows}, dtype);

            SushiBLAS::Tensor& ones_rows   = pool_->ones(Shape{rows}, dtype);
            SushiBLAS::Tensor& ones_cols   = pool_->ones(Shape{cols}, dtype);
            SushiBLAS::Tensor& ones_column = pool_->ones(Shape{cols, 1}, dtype);

            // 1. The global maximum. Log-softmax is invariant to subtracting any
            //    single scalar, so a whole-tensor reduction is as correct as a
            //    row-wise one and SushiBLAS has no row-wise one to offer.
            engine_->reductions().max(x, peak);

            // 2. Broadcast it to a row without a host round trip, so nothing
            //    per-step is baked into the recorded call (see the file comment).
            engine_->blas().gemv(ones_column, peak, peak_row, false, 1.0f, 0.0f);

            // 3. shifted = X - 1 (max)^T; every exponent is now at most zero.
            engine_->blas().copy(x, shifted);
            engine_->blas().ger(ones_rows, peak_row, shifted, -1.0f);

            // 4. row_sums = rowwise sum of exp(shifted).
            engine_->blas().copy(shifted, exponents);
            engine_->elementwise().exp(exponents);
            engine_->blas().gemv(exponents, ones_cols, row_sums, false, 1.0f, 0.0f);

            // 5. out = shifted - log(rowsum) broadcast back over the columns.
            engine_->elementwise().log(row_sums);
            engine_->blas().copy(shifted, out);
            engine_->blas().ger(row_sums, ones_cols, out, -1.0f);

            emitted_ += 9;
        }

        void Lowering::record_nll_loss(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "NLL_LOSS (node {}) takes two operands and one result", id);

            const ValueId log_probs = node.inputs[0];
            const Shape& shape = shape_of(graph, log_probs);
            SA_THROW_IF(shape.rank() != 2,
                        "NLL_LOSS (node {}) needs rank-2 operands; got {}", id, shape.to_string());

            const SushiBLAS::Tensor& l = input(node, 0);
            const SushiBLAS::Tensor& targets = input(node, 1);
            SushiBLAS::Tensor& loss = output(graph, node, 0);

            SushiBLAS::Tensor& picked = pool_->scratch(id, 0, shape, dtype_of(graph, log_probs));

            engine_->elementwise().mul(l, targets, picked);
            engine_->reductions().sum(picked, loss);
            engine_->blas().scal(-1.0f / static_cast<float>(shape[0]), loss);
            emitted_ += 3;
        }

        void Lowering::record_sum_rows(const Graph& graph, const Node& node)
        {
            const Shape& shape = shape_of(graph, node.inputs.at(0));
            SA_THROW_IF(shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", shape.to_string());

            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& sums = output(graph, node, 0);
            SushiBLAS::Tensor& ones_rows = pool_->ones(Shape{shape[0]}, dtype_of(graph, node.inputs[0]));

            // Column sums are X^T * 1, expressed with the transpose FLAG rather
            // than a transposed view - a view would report the wrong leading
            // dimension and read the matrix in the wrong order.
            engine_->blas().gemv(x, ones_rows, sums, true, 1.0f, 0.0f);
            ++emitted_;
        }

        void Lowering::record_accumulate(const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "ACCUMULATE takes two operands and one result");

            // The result IS the accumulator's buffer. Aliasing before the axpy
            // keeps the pool from ever allocating a third buffer for a sum that
            // has a perfectly good one already.
            pool_->alias(node.outputs[0], node.inputs[0]);

            SushiBLAS::Tensor& accumulator = pool_->tensor(node.inputs[0]);
            const SushiBLAS::Tensor& delta = pool_->tensor(node.inputs[1]);

            engine_->blas().axpy(1.0f, delta, accumulator);
            ++emitted_;
        }

        void Lowering::record_zero(const Graph& graph, const Node& node)
        {
            // Nothing writes a ZERO buffer after the first fill, so it is a
            // one-time constant rather than a per-replay kernel.
            SushiBLAS::Tensor& zeros = output(graph, node, 0);
            pool_->request_constant(zeros, 0.0);
        }

    } // namespace Graph
} // namespace SushiAI
