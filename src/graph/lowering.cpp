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

#include <array>
#include <span>
#include <type_traits>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
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

        void TensorPool::adopt(MemoryPlan plan)
        {
            SA_THROW_IF(arenas_.has_value(),
                        "This pool has already adopted a memory plan; its arenas are allocated "
                        "once and a second plan would leave the first one's views dangling");

            arenas_.emplace(*engine_, std::move(plan));
            arena_bytes_ = arenas_->bytes();
        }

        const ArenaSet& TensorPool::arenas() const
        {
            SA_THROW_IF(!arenas_.has_value(),
                        "This pool has no memory plan; call adopt() before asking for its arenas");
            return *arenas_;
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

        void TensorPool::note_standalone(const Shape& shape, DType dtype) noexcept
        {
            const std::size_t bytes = static_cast<std::size_t>(shape.num_elements())
                                    * dtype_size(dtype);
            unpacked_bytes_ += bytes;
            standalone_bytes_ += bytes;
        }

        SushiBLAS::Tensor& TensorPool::materialise(ValueId id, const Value& value)
        {
            const auto it = tensors_.find(id);
            if (it != tensors_.end()) return it->second;

            if (arenas_.has_value())
            {
                // An alias chain is one buffer under several names, and the plan
                // says which value owns it. Materialising through the owner is
                // what keeps the planner's liveness reasoning and the lowering's
                // aliasing from drifting apart - the second name costs nothing,
                // packed or not, so it is not counted either way.
                const ValueId owner = arenas_->plan().buffer_of(id);
                if (owner != id)
                {
                    SushiBLAS::Tensor& shared = materialise(owner, value);
                    return tensors_.emplace(id, shared).first->second;
                }

                if (arenas_->holds(id))
                {
                    // Counted against the counterfactual only: the bytes are
                    // already inside an arena that adopt() paid for.
                    unpacked_bytes_ += static_cast<std::size_t>(value.shape.num_elements())
                                     * dtype_size(value.dtype);
                    return tensors_.emplace(id, arenas_->view(id, value.shape, value.dtype))
                                   .first->second;
                }
            }

            note_standalone(value.shape, value.dtype);
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

            const auto existing = tensors_.find(target);
            if (existing != tensors_.end())
            {
                // Already aliased - by an adopted memory plan, which resolves
                // the same chain to the same buffer, or by a second lowering of
                // the same graph. Only a *different* buffer is a mistake.
                SA_THROW_IF(existing->second.storage != it->second.storage
                            || existing->second.storage_offset != it->second.storage_offset,
                            "Value {} is already bound to a different buffer; it cannot also "
                            "alias value {}", target, source);
                return;
            }

            tensors_.emplace(target, it->second);
        }

        SushiBLAS::Tensor& TensorPool::ones(const Shape& shape, DType dtype)
        {
            const ShapeKey key{shape.hash(), static_cast<std::int32_t>(dtype)};

            const auto it = ones_.find(key);
            if (it != ones_.end()) return it->second;

            note_standalone(shape, dtype);

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

            note_standalone(shape, dtype);

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

            // =============================================================
            // The GEMM epilogues
            // =============================================================
            //
            // One functor per catalog entry, because that is what a catalog
            // entry *is* on this stack: `sycl_ext_codeplay_kernel_fusion` is
            // gone from intel-llvm, so a composition exists only if some
            // template was instantiated with it, and these four instantiations
            // are the set. See graph/fusion.hpp.
            //
            // WHY THE BIAS IS A `const void*`
            //
            // `Dispatch::execute_gemm` instantiates the epilogue for every real
            // element type in its Domain - half, float and double - and picks
            // the arm matching the tensor's runtime dtype. A functor holding a
            // `const float*` therefore compiles into the double arm too and
            // would read the bias at the wrong stride if a double graph ever
            // reached it. Carrying the untyped pointer and casting inside the
            // call operator makes the arm that runs the one that is right, and
            // costs nothing: the cast disappears at compile time.
            //
            // WHY THESE ARE BIT-IDENTICAL TO THE OPERATIONS THEY REPLACE
            //
            // `BIAS_ADD` lowers to `copy(x, y)` then `ger(ones, bias, y, 1.0f)`,
            // whose kernel is `y += alpha * ones[row] * bias[col]`. With alpha
            // and ones[row] both exactly 1, that is `y + bias[col]` with no
            // rounding of its own, which is what `operator()` below computes -
            // and `-ffp-contract=off` means neither spelling can contract the
            // multiply-add. The activations are SushiBLAS's own expressions,
            // copied rather than paraphrased, for the same reason.

            /** @brief Adds a per-column bias to the GEMM's result. */
            struct BiasEpilogue
            {
                const void* bias;

                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    return value + static_cast<const T*>(bias)[col];
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's ReLU. */
            struct BiasReluEpilogue
            {
                const void* bias;

                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];
                    return SushiBLAS::Dispatch::greater_than_zero(y) ? y : T(0);
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's logistic sigmoid. */
            struct BiasSigmoidEpilogue
            {
                const void* bias;

                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];
                    return T(1) / (T(1) + SushiBLAS::Dispatch::safe_exp(-y));
                }
            };

            /** @brief Adds a per-column bias, then SushiBLAS's tanh. */
            struct BiasTanhEpilogue
            {
                const void* bias;

                template<typename T>
                T operator()(T value, std::int64_t, std::int64_t col) const
                {
                    return SushiBLAS::Dispatch::safe_tanh(
                        value + static_cast<const T*>(bias)[col]);
                }
            };
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

        std::vector<ValueId> Lowering::values_needing_own_allocation(const Graph& graph)
        {
            std::vector<ValueId> pinned;

            for (const Node& node : graph.nodes())
            {
                // Both entries below are operands of SushiBLAS's whole-tensor
                // reductions - see the header for why an arena view breaks
                // them. Delete this whole function the day
                // Dispatch::execute_reduction builds its ResourceRegions from
                // Tensor::region() like every other op family already does;
                // nothing else here depends on it.
                if (node.op == Ops::LOG_SOFTMAX.id && !node.inputs.empty())
                    pinned.push_back(node.inputs[0]);   // reductions().max reads it

                if (node.op == Ops::NLL_LOSS.id && !node.outputs.empty())
                    pinned.push_back(node.outputs[0]);  // reductions().sum writes it
            }

            return pinned;
        }

        void Lowering::record(const Graph& graph)
        {
            emitted_ = 0;
            ordering_nodes_ = 0;
            selections_.clear();

            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
                record_node(graph, id);
        }

        FusionChoice Lowering::select(const FusionQuery& query)
        {
            // Built here rather than held as a member so a Lowering costs
            // nothing on a graph with no fused nodes, and so the profile is
            // read from the engine this record() is actually recording on.
            const FusionChoice choice =
                selector_ != nullptr
                    ? selector_->choose(query)
                    : DeviceFusionSelector::for_engine(*engine_).choose(query);

            SA_THROW_IF(choice.form != FusedForm::NONE && choice.form != query.matched
                        && !(is_gemm_epilogue(choice.form) && is_gemm_epilogue(query.matched)),
                        "A fusion selector narrowed {} to {}, which is not a spelling of the "
                        "same node. A selector may take the whole match, a shorter entry of the "
                        "same family, or nothing at all.",
                        fused_form_name(query.matched), fused_form_name(choice.form));

            selections_.push_back(FusionSelection{query.matched, choice.form, choice.reason, 0});
            return choice;
        }

        void Lowering::declare_epilogue_operand(const SushiBLAS::Tensor& operand,
                                                const SushiBLAS::Tensor& result)
        {
            // A host node that runs nothing, whose entire purpose is its two
            // access declarations. See lowering.hpp's warning for why the fused
            // GEMM cannot make them itself; in short, SushiBLAS declares A, B
            // and C and explicitly hands the epilogue's own reads back to the
            // caller, and the bias here is a parameter the optimizer rewrites
            // inside this very plan.
            //
            // Recorded immediately before the GEMM, so the write-after-write on
            // `result` pulls the GEMM in behind it and the read of `operand`
            // pushes the optimizer's update out in front. Neither edge is
            // implied by anything else - the transitive path through the
            // gradient happens to exist for a plain feed-forward model today,
            // and relying on it would be exactly the kind of accident that
            // survives every test and breaks the first time somebody adds a
            // second writer.
            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = Ops::EPILOGUE_OPERANDS.name.data();
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = Ops::EPILOGUE_OPERANDS.id;

            engine_->get_graph().add_native({operand.region()}, {result.region()},
                                            []{}, {}, meta);
            ++ordering_nodes_;
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

            const FusedForm form = fused_form_of(node.op);
            if (is_gemm_epilogue(form))       { record_fused_gemm(graph, id, node, form); return; }
            if (form == FusedForm::EXP_SUB_SCALE)
            {
                record_fused_exp_sub_scale(graph, id, node, form);
                return;
            }

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

        void Lowering::record_fused_gemm(const Graph& graph, NodeId id, const Node& node,
                                         FusedForm matched)
        {
            SA_THROW_IF(node.inputs.size() != 3 || node.outputs.size() != 1,
                        "{} (node {}) takes three operands - A, B and the bias - and one result",
                        Ops::name_of(node.op), id);

            const Shape& out_shape = shape_of(graph, node.outputs[0]);
            SA_THROW_IF(out_shape.rank() != 2,
                        "{} (node {}) needs a rank-2 result; got {}",
                        Ops::name_of(node.op), id, out_shape.to_string());

            const SushiBLAS::Tensor& a = input(node, 0);
            const SushiBLAS::Tensor& b = input(node, 1);
            const SushiBLAS::Tensor& bias = input(node, 2);
            SushiBLAS::Tensor& c = output(graph, node, 0);

            const bool transpose_a = node.attributes.flag(0);
            const bool transpose_b = node.attributes.flag(1);
            const DType dtype = dtype_of(graph, node.outputs[0]);

            FusionQuery query;
            query.matched = matched;
            query.dtype = dtype;
            query.rows = out_shape[0];
            query.columns = out_shape[1];
            query.operand_is_contiguous = bias.is_contiguous();

            const FusionChoice choice = select(query);
            const std::size_t before = emitted_;

            if (choice.form == FusedForm::NONE)
            {
                // The unfused spelling, and still cheaper than the IR it came
                // from: the fused node names A, B and the bias directly, so the
                // rank-1 update and the activation both run in place on C and
                // neither needs the copy the out-of-place IR made them pay.
                // Same operations in the same order, so the same bytes.
                engine_->blas().gemm(a, b, c, transpose_a, transpose_b, 1.0, 0.0);

                SushiBLAS::Tensor& rows = pool_->ones(Shape{out_shape[0]}, dtype);
                engine_->blas().ger(rows, bias, c, 1.0f);
                emitted_ += 2;
            }
            else
            {
                declare_epilogue_operand(bias, c);

                const void* const values = bias.data();

                // One arm per catalog entry. This switch is the point at which
                // "which compositions exist" stops being a runtime question:
                // each arm instantiates a different template, and a composition
                // with no arm here does not exist in the binary at all.
                switch (choice.form)
                {
                    case FusedForm::GEMM_BIAS_RELU:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_RELU.name.data(), Ops::GEMM_BIAS_RELU.id,
                            BiasReluEpilogue{values});
                        break;

                    case FusedForm::GEMM_BIAS_SIGMOID:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_SIGMOID.name.data(), Ops::GEMM_BIAS_SIGMOID.id,
                            BiasSigmoidEpilogue{values});
                        break;

                    case FusedForm::GEMM_BIAS_TANH:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_TANH.name.data(), Ops::GEMM_BIAS_TANH.id,
                            BiasTanhEpilogue{values});
                        break;

                    default:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS.name.data(), Ops::GEMM_BIAS.id,
                            BiasEpilogue{values});
                        break;
                }
                ++emitted_;
            }

            // Whatever activation the chosen spelling did not fold in still has
            // to run, in place on the result. Correct for every activation in
            // Ops::ACTIVATIONS, fused or not, because the fused node's output
            // *is* the activation's output value - the pass only ever absorbs
            // an activation whose gradient reads that same tensor.
            const OpID activation = fused_form_activation(matched);
            if (activation != Ops::LEAF.id && fused_form_activation(choice.form) != activation)
            {
                if (activation == Ops::RELU.id)         engine_->nonlinear().relu(c);
                else if (activation == Ops::SIGMOID.id) engine_->nonlinear().sigmoid(c);
                else                                    engine_->nonlinear().tanh(c);
                ++emitted_;
            }

            selections_.back().tasks = emitted_ - before;
        }

        void Lowering::record_fused_exp_sub_scale(const Graph& graph, NodeId id, const Node& node,
                                                  FusedForm matched)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "{} (node {}) takes two operands and one result",
                        Ops::name_of(node.op), id);

            const Shape& out_shape = shape_of(graph, node.outputs[0]);

            // Non-const, and from the pool rather than through input(), because
            // the zip seam takes every operand as a writable slot; see below.
            SushiBLAS::Tensor& a = pool_->tensor(node.inputs[0]);
            SushiBLAS::Tensor& b = pool_->tensor(node.inputs[1]);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            // Through float on both paths: the unfused spelling ends in
            // blas().scal, whose alpha is a float, so a double graph would
            // otherwise pick up a differently rounded constant on the two
            // paths and break the byte-for-byte comparison for a reason that
            // has nothing to do with fusion.
            const float alpha = static_cast<float>(node.attributes.get(0));

            FusionQuery query;
            query.matched = matched;
            query.dtype = dtype_of(graph, node.outputs[0]);
            query.rows = out_shape.rank() > 0 ? out_shape[0] : 0;
            query.columns = out_shape.rank() > 1 ? out_shape[1] : 1;

            const FusionChoice choice = select(query);
            const std::size_t before = emitted_;

            if (choice.form == FusedForm::NONE)
            {
                // Out is its own scratch: it has the operands' shape, so the
                // whole recipe runs through it and no temporary is allocated.
                engine_->blas().copy(a, out);
                engine_->elementwise().exp(out);
                engine_->elementwise().sub(out, b, out);
                engine_->blas().scal(alpha, out);
                emitted_ += 4;
            }
            else
            {
                // Every slot of a zip is declared to the tracker as both read
                // and written, and every slot is staged and written back - so
                // `a` and `b` are rewritten with the bytes they already held.
                // That is a real write and it is why they are named here rather
                // than passed as const: it is what makes the ordering against
                // whoever else reads them explicit instead of accidental.
                std::array<SushiBLAS::Tensor*, 3> operands{&out, &a, &b};

                SushiBLAS::Dispatch::execute_zip_inplace(*engine_, operands,
                    Ops::EXP_SUB_SCALE.name.data(), Ops::EXP_SUB_SCALE.id,
                    [alpha](auto& result, auto& logit, auto& target)
                    {
                        using T = std::decay_t<decltype(result)>;
                        result = (SushiBLAS::Dispatch::safe_exp(logit) - target)
                               * static_cast<T>(alpha);
                    });
                ++emitted_;
            }

            selections_.back().tasks = emitted_ - before;
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
            // Recorded into the step, not queued as a one-time constant, even
            // though nothing else ever writes this buffer. The memory planner
            // hands a dead value's bytes to a later one, so a buffer filled
            // once outside the step would hold zeros for exactly one replay and
            // then whatever overwrote it - and nothing downstream could tell,
            // because a gradient that is subtly wrong still trains. One tiny
            // fill per step is the price of the buffer being packable at all.
            SushiBLAS::Tensor& zeros = output(graph, node, 0);
            engine_->random().constant(zeros, 0.0);
            ++emitted_;
        }

    } // namespace Graph
} // namespace SushiAI
