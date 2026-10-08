/**************************************************************************/
/* lowering.cpp                                                           */
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
 * @file lowering.cpp
 * @brief Implements graph execution lowering and kernel dispatch recording.
 * @author Mustafa Garip
 */

#include <array>
#include <type_traits>
#include <SushiBLAS/engine/blas/dispatch.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiBLAS/engine/math/elementwise/dispatch.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/lowering.hpp>

namespace SushiAI
{
    namespace Graph
    {
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
                // Materialises alias chains through buffer owner to align with planner liveness.
                const ValueId owner = arenas_->plan().buffer_of(id);
                if (owner != id)
                {
                    SushiBLAS::Tensor& shared = materialise(owner, value);
                    return tensors_.emplace(id, shared).first->second;
                }

                if (arenas_->holds(id))
                {
                    // Tracks counterfactual unpacked byte totals without reallocating arena storage.
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
                if (node.op != Ops::K_LEAF.id) continue;
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
                // Validates that existing alias points to the same storage and offset.
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

            /**
             * @brief GEMM epilogue functors for fused catalog kernels.
             *
             * Evaluates bias addition and activation in-place during GEMM dispatch;
             * carries untyped pointers to allow runtime dtype specialization.
             */

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

            /**
             * @brief Adds a per-column bias, spills intermediate pre-activation, and applies GELU.
             *
             * Preserves pre-activation values in auxiliary buffer for backward differentiation.
             */
            struct BiasGeluSpillEpilogue
            {
                const void* bias;
                void* aux;
                std::int64_t aux_row_stride;
                std::int64_t aux_col_stride;

                template<typename T>
                T operator()(T value, std::int64_t row, std::int64_t col) const
                {
                    const T y = value + static_cast<const T*>(bias)[col];

                    static_cast<T*>(aux)[row * aux_row_stride + col * aux_col_stride] = y;

                    // Applies exact GELU polynomial approximation matching SushiBLAS kernel.
                    const T y3 = y * y * y;
                    const T inner = T(0.7978845608028654) * (y + T(0.044715) * y3);
                    return T(0.5) * y * (T(1) + SushiBLAS::Dispatch::safe_tanh(inner));
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
                // Pins reduction operands requiring contiguous standalone allocation.
                if (node.op == Ops::K_LOG_SOFTMAX.id && !node.inputs.empty())
                    pinned.push_back(node.inputs[0]);   // reductions().max reads it

                if (node.op == Ops::K_NLL_LOSS.id && !node.outputs.empty())
                    pinned.push_back(node.outputs[0]);  // reductions().sum writes it
            }

            return pinned;
        }

        void Lowering::record(const Graph& graph)
        {
            emitted_ = 0;
            selections_.clear();

            // Verifies layout uniformity across all graph values prior to lowering.
            graph.prove_layout_uniformity();

            for (NodeId id = 0; id < static_cast<NodeId>(graph.node_count()); ++id)
                record_node(graph, id);
        }

        FusionChoice Lowering::select(const FusionQuery& query)
        {
            // Queries engine-specific fusion selector to evaluate kernel viability.
            const FusionChoice choice =
                selector_ != nullptr
                    ? selector_->choose(query)
                    : DeviceFusionSelector::for_engine(*engine_).choose(query);

            // Allows narrowing from extended epilogues to GEMM_BIAS within the same family.
            const bool narrowed_within_family =
                is_gemm_epilogue(query.matched) && choice.form == FusedForm::GemmBias;

            SA_THROW_IF(choice.form != FusedForm::None && choice.form != query.matched
                        && !narrowed_within_family,
                        "A fusion selector answered {} to a {} node, which is not a spelling of "
                        "it. A selector may take the whole match, narrow a GEMM epilogue to "
                        "gemm+bias, or take nothing at all.",
                        fused_form_name(choice.form), fused_form_name(query.matched));

            selections_.push_back(FusionSelection{query.matched, choice.form, choice.reason, 0});
            return choice;
        }

        void Lowering::record_node(const Graph& graph, NodeId id)
        {
            const Node& node = graph.node(id);

            if (node.op == Ops::K_LEAF.id)
            {
                // Validates that leaf output values are explicitly bound before execution.
                for (ValueId out : node.outputs)
                    SA_THROW_IF(!pool_->is_bound(out),
                                "Leaf value {} ('{}') has no tensor bound. Bind it, or call "
                                "TensorPool::materialise_leaves() before lowering.",
                                out, graph.value(out).debug_name);
                return;
            }

            if (node.op == Ops::K_MATMUL.id)                 { record_matmul(graph, id, node); return; }
            if (node.op == Ops::K_BIAS_ADD.id)               { record_bias_add(graph, id, node); return; }
            if (node.op == Ops::K_ADD.id
                || node.op == Ops::K_SUB.id
                || node.op == Ops::K_MUL.id)                 { record_binary(graph, node); return; }
            if (node.op == Ops::K_SCALE.id)                  { record_scale(graph, node); return; }
            if (node.op == Ops::K_DYNAMIC_SCALE.id)          { record_dynamic_scale(graph, id, node); return; }
            if (node.op == Ops::K_EXP.id
                || node.op == Ops::K_RELU.id
                || node.op == Ops::K_GELU.id
                || node.op == Ops::K_SIGMOID.id
                || node.op == Ops::K_TANH.id)                { record_unary_inplace(graph, node); return; }
            if (node.op == Ops::K_LEAKY_RELU.id)             { record_leaky_relu(graph, node); return; }
            if (node.op == Ops::K_RELU_BACKWARD.id
                || node.op == Ops::K_GELU_BACKWARD.id
                || node.op == Ops::K_SIGMOID_BACKWARD.id
                || node.op == Ops::K_TANH_BACKWARD.id)       { record_activation_backward(graph, node); return; }
            if (node.op == Ops::K_LEAKY_RELU_BACKWARD.id)    { record_leaky_relu_backward(graph, node); return; }
            if (node.op == Ops::K_LOG_SOFTMAX.id)            { record_log_softmax(graph, id, node); return; }
            if (node.op == Ops::K_NLL_LOSS.id)               { record_nll_loss(graph, id, node); return; }
            if (node.op == Ops::K_SUM_ROWS.id)               { record_sum_rows(graph, node); return; }
            if (node.op == Ops::K_ACCUMULATE.id)             { record_accumulate(node); return; }
            if (node.op == Ops::K_ZERO.id)                   { record_zero(graph, node); return; }
            if (node.op == Ops::K_CAST.id)                   { record_cast(graph, id, node); return; }
            if (node.op == Ops::K_RESHAPE.id)                { record_reshape(graph, id, node); return; }
            if (node.op == Ops::K_LAYOUT_CONVERT.id)         { record_layout_convert(graph, id, node); return; }
            if (node.op == Ops::K_IM2COL.id)                 { record_im2col(graph, id, node); return; }
            if (node.op == Ops::K_COL2IM.id)                 { record_col2im(graph, id, node); return; }
            if (node.op == Ops::K_MAXPOOL2D.id)              { record_maxpool2d(graph, id, node); return; }
            if (node.op == Ops::K_MAXPOOL2D_BACKWARD.id)     { record_maxpool2d_backward(graph, id, node); return; }
            if (node.op == Ops::K_AVGPOOL2D.id)              { record_avgpool2d(graph, id, node); return; }
            if (node.op == Ops::K_AVGPOOL2D_BACKWARD.id)     { record_avgpool2d_backward(graph, id, node); return; }
            if (node.op == Ops::K_UPSAMPLE2D.id)             { record_upsample2d(graph, id, node); return; }
            if (node.op == Ops::K_UPSAMPLE2D_BACKWARD.id)    { record_upsample2d_backward(graph, id, node); return; }
            if (node.op == Ops::K_SPATIAL_MEAN.id)           { record_spatial_mean(graph, id, node); return; }
            if (node.op == Ops::K_SPATIAL_MEAN_BACKWARD.id)  { record_spatial_mean_backward(graph, id, node); return; }
            if (node.op == Ops::K_CONCAT_CHANNELS.id)        { record_concat_channels(graph, id, node); return; }
            if (node.op == Ops::K_SLICE_CHANNELS.id)         { record_slice_channels(graph, id, node); return; }
            if (node.op == Ops::K_CHANNEL_AFFINE.id)         { record_channel_affine(graph, id, node); return; }
            if (node.op == Ops::K_BATCHNORM2D.id)            { record_batchnorm2d(graph, id, node); return; }
            if (node.op == Ops::K_BATCHNORM2D_BACKWARD.id)   { record_batchnorm2d_backward(graph, id, node); return; }

            const FusedForm form = fused_form_of(node.op);
            if (is_gemm_epilogue(form))       { record_fused_gemm(graph, id, node, form); return; }
            if (form == FusedForm::ExpSubScale)
            {
                record_fused_exp_sub_scale(graph, id, node, form);
                return;
            }

            // Rejects undeclared ops or operations managed outside IR lowering.
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

            // Copies input to destination before accumulating outer product bias in-place.
            engine_->blas().copy(x, y);
            engine_->blas().ger(rows, bias, y, 1.0f);
            emitted_ += 2;
        }

        void Lowering::record_fused_gemm(const Graph& graph, NodeId id, const Node& node,
                                         FusedForm matched)
        {
            // Validates operand and result counts against catalog spill specification.
            const bool spills = fused_form_spills(matched);
            const std::size_t results = spills ? 2u : 1u;

            SA_THROW_IF(node.inputs.size() != 3 || node.outputs.size() != results,
                        "{} (node {}) takes three operands - A, B and the bias - and {} result{}",
                        Ops::name_of(node.op), id, results, results == 1 ? "" : "s");

            const Shape& out_shape = shape_of(graph, node.outputs[0]);

            // Enforces rank-2 result to ensure unbatched epilogue element indexing.
            SA_THROW_IF(out_shape.rank() != 2,
                        "{} (node {}) needs a rank-2 result; got {}",
                        Ops::name_of(node.op), id, out_shape.to_string());

            const SushiBLAS::Tensor& a = input(node, 0);
            const SushiBLAS::Tensor& b = input(node, 1);
            const SushiBLAS::Tensor& bias = input(node, 2);
            SushiBLAS::Tensor& c = output(graph, node, 0);

            // Materialises auxiliary output buffer required by fused or fallback arms.
            SushiBLAS::Tensor* const aux = spills ? &output(graph, node, 1) : nullptr;

            if (spills)
            {
                SA_THROW_IF(shape_of(graph, node.outputs[1]) != out_shape,
                            "{} (node {}) spills its pre-activation into a {} buffer, but its "
                            "result is {}; the two are the same tensor written twice",
                            Ops::name_of(node.op), id,
                            shape_of(graph, node.outputs[1]).to_string(), out_shape.to_string());
                SA_THROW_IF(dtype_of(graph, node.outputs[1]) != dtype_of(graph, node.outputs[0]),
                            "{} (node {}) spills its pre-activation as {} but computes in {}; "
                            "the epilogue stores the value it is holding, in the element type "
                            "the GEMM was instantiated for",
                            Ops::name_of(node.op), id,
                            dtype_name(dtype_of(graph, node.outputs[1])),
                            dtype_name(dtype_of(graph, node.outputs[0])));
            }

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

            if (choice.form == FusedForm::None)
            {
                // Executes unfused GEMM and bias rank-1 update in-place on result buffer.
                engine_->blas().gemm(a, b, c, transpose_a, transpose_b, 1.0, 0.0);

                SushiBLAS::Tensor& rows = pool_->ones(Shape{out_shape[0]}, dtype);
                engine_->blas().ger(rows, bias, c, 1.0f);
                emitted_ += 2;
            }
            else
            {
                const void* const values = bias.data();

                // Dispatches fused epilogue with declared operand dependencies per catalog arm.
                switch (choice.form)
                {
                    case FusedForm::GemmBiasRelu:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::K_GEMM_BIAS_RELU.name.data(), Ops::K_GEMM_BIAS_RELU.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasReluEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GemmBiasSigmoid:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::K_GEMM_BIAS_SIGMOID.name.data(), Ops::K_GEMM_BIAS_SIGMOID.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasSigmoidEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GemmBiasTanh:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::K_GEMM_BIAS_TANH.name.data(), Ops::K_GEMM_BIAS_TANH.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasTanhEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GemmBiasGelu:
                        // Declares auxiliary buffer write dependency for backward graph ordering.
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::K_GEMM_BIAS_GELU.name.data(), Ops::K_GEMM_BIAS_GELU.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasGeluSpillEpilogue{values, aux->data(),
                                                      aux->strides[0], aux->strides[1]})
                                .reads(bias)
                                .writes(*aux));
                        break;

                    case FusedForm::GemmBias:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::K_GEMM_BIAS.name.data(), Ops::K_GEMM_BIAS.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasEpilogue{values}).reads(bias));
                        break;

                    default:
                        SA_THROW_IF(true,
                                    "No GEMM epilogue for fused form {} ({}, node {}). Add it to "
                                    "Lowering::record_fused_gemm or stop the selector choosing it.",
                                    fused_form_name(choice.form), Ops::name_of(node.op), id);
                        break;
                }
                ++emitted_;
            }

            // Copies pre-activation result to auxiliary buffer prior to in-place activation.
            if (spills && !fused_form_spills(choice.form))
            {
                engine_->blas().copy(c, *aux);
                ++emitted_;
            }

            // Executes trailing activation in-place when excluded from fused epilogue.
            const OpID activation = fused_form_activation(matched);
            if (activation != Ops::K_LEAF.id && fused_form_activation(choice.form) != activation)
            {
                if (activation == Ops::K_RELU.id)         engine_->nonlinear().relu(c);
                else if (activation == Ops::K_SIGMOID.id) engine_->nonlinear().sigmoid(c);
                else if (activation == Ops::K_GELU.id)    engine_->nonlinear().gelu(c);
                else if (activation == Ops::K_TANH.id)    engine_->nonlinear().tanh(c);
                else
                    SA_THROW_IF(true,
                                "No activation kernel for operation {}, the tail of {} (node {}). "
                                "Add it to Lowering::record_fused_gemm or remove it from the "
                                "fusion catalog.",
                                Ops::name_of(activation), Ops::name_of(node.op), id);
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

            // Acquires writable tensor references from pool for zip in-place binding.
            SushiBLAS::Tensor& a = pool_->tensor(node.inputs[0]);
            SushiBLAS::Tensor& b = pool_->tensor(node.inputs[1]);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            // Casts scaling factor to float to maintain numerical symmetry with blas::scal.
            const float alpha = static_cast<float>(node.attributes.get(0));

            FusionQuery query;
            query.matched = matched;
            query.dtype = dtype_of(graph, node.outputs[0]);
            query.rows = out_shape.rank() > 0 ? out_shape[0] : 0;
            query.columns = out_shape.rank() > 1 ? out_shape[1] : 1;

            const FusionChoice choice = select(query);
            const std::size_t before = emitted_;

            if (choice.form == FusedForm::None)
            {
                // Reuses output tensor as scratch across sequential elementwise stages.
                engine_->blas().copy(a, out);
                engine_->elementwise().exp(out);
                engine_->elementwise().sub(out, b, out);
                engine_->blas().scal(alpha, out);
                emitted_ += 4;
            }
            else
            {
                // Dispatches fused elementwise zip with explicit read-write slot declarations.
                std::array<SushiBLAS::Tensor*, 3> operands{&out, &a, &b};

                SushiBLAS::Dispatch::execute_zip_inplace(*engine_, operands,
                    Ops::K_EXP_SUB_SCALE.name.data(), Ops::K_EXP_SUB_SCALE.id,
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

            if (node.op == Ops::K_ADD.id)      engine_->elementwise().add(a, b, c);
            else if (node.op == Ops::K_SUB.id) engine_->elementwise().sub(a, b, c);
            else if (node.op == Ops::K_MUL.id) engine_->elementwise().mul(a, b, c);
            else
                SA_THROW_IF(true,
                            "No elementwise kernel for operation {}. Add it to "
                            "Lowering::record_binary or stop routing it there.",
                            Ops::name_of(node.op));

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

        void Lowering::record_dynamic_scale(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "DYNAMIC_SCALE (node {}) takes two operands and one result", id);

            const SushiBLAS::Tensor& x = input(node, 0);
            const SushiBLAS::Tensor& factor = input(node, 1);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            SA_THROW_IF(factor.num_elements != 1,
                        "DYNAMIC_SCALE (node {}) was given a {}-element factor; it must be one",
                        id, factor.num_elements);

            // Copies operand to destination before scaling in-place.
            engine_->blas().copy(x, y);

            // Captures factor buffer pointer to allow host updates without recompilation.
            constexpr const char* K_TASK_NAME = "sushiai.dynamic_scale";
            static_assert(Ops::K_DYNAMIC_SCALE.name == std::string_view(K_TASK_NAME),
                          "the task name must be the string its OpID hashes");

            SushiBLAS::Dispatch::execute_unary_inplace(*engine_, y,
                K_TASK_NAME, Ops::K_DYNAMIC_SCALE.id,
                [f = static_cast<const void*>(factor.data())](auto v)
                {
                    using T = std::decay_t<decltype(v)>;
                    return v * static_cast<const T*>(f)[0];
                });

            emitted_ += 2;
        }

        void Lowering::record_unary_inplace(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            // Copies input to output before in-place evaluation to preserve primal tensor.
            engine_->blas().copy(x, y);

            if (node.op == Ops::K_EXP.id)          engine_->elementwise().exp(y);
            else if (node.op == Ops::K_RELU.id)    engine_->nonlinear().relu(y);
            else if (node.op == Ops::K_GELU.id)    engine_->nonlinear().gelu(y);
            else if (node.op == Ops::K_SIGMOID.id) engine_->nonlinear().sigmoid(y);
            else if (node.op == Ops::K_TANH.id)    engine_->nonlinear().tanh(y);
            else
                SA_THROW_IF(true,
                            "No in-place kernel for operation {}. Add it to "
                            "Lowering::record_unary_inplace or stop routing it there.",
                            Ops::name_of(node.op));

            emitted_ += 2;
        }

        void Lowering::record_activation_backward(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& primal = input(node, 1);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            if (node.op == Ops::K_RELU_BACKWARD.id)         engine_->nonlinear().relu_backward(dy, primal, dx);
            else if (node.op == Ops::K_GELU_BACKWARD.id)    engine_->nonlinear().gelu_backward(dy, primal, dx);
            else if (node.op == Ops::K_SIGMOID_BACKWARD.id) engine_->nonlinear().sigmoid_backward(dy, primal, dx);
            else if (node.op == Ops::K_TANH_BACKWARD.id)    engine_->nonlinear().tanh_backward(dy, primal, dx);
            else
                SA_THROW_IF(true,
                            "No backward kernel for operation {}. Add it to "
                            "Lowering::record_activation_backward or stop routing it there.",
                            Ops::name_of(node.op));

            ++emitted_;
        }

        void Lowering::record_leaky_relu(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            // Copies input to output to preserve primal values for backward pass.
            engine_->blas().copy(x, y);
            engine_->nonlinear().leaky_relu(y, static_cast<float>(node.attributes.get(0)));

            emitted_ += 2;
        }

        void Lowering::record_leaky_relu_backward(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& primal = input(node, 1);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            // Reads negative slope attribute forwarded from primal operation.
            engine_->nonlinear().leaky_relu_backward(dy, primal, dx,
                static_cast<float>(node.attributes.get(0)));

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

            SushiBLAS::Tensor& row_max   = pool_->scratch(id, 0, Shape{rows}, dtype);
            SushiBLAS::Tensor& shifted   = pool_->scratch(id, 1, x_shape, dtype);
            SushiBLAS::Tensor& exponents = pool_->scratch(id, 2, x_shape, dtype);
            SushiBLAS::Tensor& row_sums  = pool_->scratch(id, 3, Shape{rows}, dtype);

            SushiBLAS::Tensor& ones_cols = pool_->ones(Shape{cols}, dtype);

            // Computes per-row maximum for numerically stable log-softmax normalization.
            engine_->reductions().max_along(x, 1, row_max);

            // Subtracts row maximum via outer product to bound exponents non-positive.
            engine_->blas().copy(x, shifted);
            engine_->blas().ger(row_max, ones_cols, shifted, -1.0f);

            // Evaluates row-wise sum of exponents using matrix-vector product.
            engine_->blas().copy(shifted, exponents);
            engine_->elementwise().exp(exponents);
            engine_->blas().gemv(exponents, ones_cols, row_sums, false, 1.0f, 0.0f);

            // 4. out = shifted - log(rowsum) broadcast back over the columns.
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

            // Computes column sums via matrix-vector multiplication with transpose flag.
            engine_->blas().gemv(x, ones_rows, sums, true, 1.0f, 0.0f);
            ++emitted_;
        }

        void Lowering::record_accumulate(const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "ACCUMULATE takes two operands and one result");

            // Aliases accumulator output directly onto input buffer prior to axpy.
            pool_->alias(node.outputs[0], node.inputs[0]);

            SushiBLAS::Tensor& accumulator = pool_->tensor(node.inputs[0]);
            const SushiBLAS::Tensor& delta = pool_->tensor(node.inputs[1]);

            engine_->blas().axpy(1.0f, delta, accumulator);
            ++emitted_;
        }

        void Lowering::record_zero(const Graph& graph, const Node& node)
        {
            // Emits constant zero initialization step to support packable buffer reuse.
            SushiBLAS::Tensor& zeros = output(graph, node, 0);
            engine_->random().constant(zeros, 0.0);
            ++emitted_;
        }

        void Lowering::record_cast(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "CAST (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& source = input(node, 0);
            SushiBLAS::Tensor& destination = output(graph, node, 0);

            // Dispatches precision conversion with explicit region dependency tracking.
            engine_->elementwise().convert(source, destination);
            ++emitted_;
        }

        void Lowering::record_reshape(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "RESHAPE (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& source = input(node, 0);
            SushiBLAS::Tensor& destination = output(graph, node, 0);

            SA_THROW_IF(!source.is_contiguous(),
                        "RESHAPE (node {}) needs a contiguous operand; the value it reads was "
                        "produced as a strided view, which reshape's reinterpretation would "
                        "read wrong", id);

            // Copies contiguous storage into destination under reinterpreted shape extents.
            const Shape& out_shape = graph.value(node.outputs[0]).shape;
            const SushiBLAS::Tensor reshaped(
                source.storage,
                SushiRuntime::span<const int64_t>(out_shape.data(),
                                                  static_cast<std::size_t>(out_shape.rank())),
                source.dtype, source.storage_offset, source.layout);

            engine_->blas().copy(reshaped, destination);
            ++emitted_;
        }

        void Lowering::record_layout_convert(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "LAYOUT_CONVERT (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& source = input(node, 0);
            SushiBLAS::Tensor& destination = output(graph, node, 0);

            const TensorLayout source_layout = graph.value(node.inputs[0]).layout;
            const TensorLayout dest_layout = graph.value(node.outputs[0]).layout;

            // Converts spatial channel layouts between NCHW and NHWC representations.
            const SushiBLAS::Tensor source_view = (source_layout == TensorLayout::Nchw)
                ? source.transpose(1, 2).transpose(2, 3) : source;
            SushiBLAS::Tensor dest_view = (dest_layout == TensorLayout::Nchw)
                ? destination.transpose(1, 2).transpose(2, 3) : destination;

            engine_->spatial().layout_convert(source_view, dest_view);
            ++emitted_;
        }

        namespace
        {
            /** @brief Reads CV-5's eight slots back into a SushiBLAS geometry. */
            SushiBLAS::Spatial::ConvGeometry blas_geometry_of(const Node& node) noexcept
            {
                SushiBLAS::Spatial::ConvGeometry g;
                g.kernel_h   = static_cast<int64_t>(node.attributes.get(0));
                g.kernel_w   = static_cast<int64_t>(node.attributes.get(1));
                g.stride_h   = static_cast<int64_t>(node.attributes.get(2));
                g.stride_w   = static_cast<int64_t>(node.attributes.get(3));
                g.dilation_h = static_cast<int64_t>(node.attributes.get(4));
                g.dilation_w = static_cast<int64_t>(node.attributes.get(5));
                g.pad_top    = static_cast<int64_t>(node.attributes.get(6));
                g.pad_left   = static_cast<int64_t>(node.attributes.get(7));
                return g;
            }

            /**
             * @brief Extracts output window grid extents from attribute slots 8 and 9.
             * @param node Graph node carrying spatial attributes.
             * @param id Node identifier for error reporting.
             * @param what Operation name string.
             * @param out_h Output height extent.
             * @param out_w Output width extent.
             */
            void windowed_extents_of(const Node& node, NodeId id, const char* what,
                                     int64_t& out_h, int64_t& out_w)
            {
                out_h = static_cast<int64_t>(node.attributes.get(8));
                out_w = static_cast<int64_t>(node.attributes.get(9));

                SA_THROW_IF(out_h <= 0 || out_w <= 0,
                            "{} (node {}) carries a {}x{} window grid; its emitter did "
                            "not write the output extents into attribute slots 8 and 9",
                            what, id, out_h, out_w);
            }
        }

        void Lowering::record_im2col(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "IM2COL (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& cols = output(graph, node, 0);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            int64_t out_h = 0;
            int64_t out_w = 0;
            windowed_extents_of(node, id, "IM2COL", out_h, out_w);

            engine_->spatial().im2col(in, cols, g, out_h, out_w);
            ++emitted_;
        }

        void Lowering::record_col2im(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "COL2IM (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& cols = input(node, 0);
            SushiBLAS::Tensor& image = output(graph, node, 0);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            int64_t out_h = 0;
            int64_t out_w = 0;
            windowed_extents_of(node, id, "COL2IM", out_h, out_w);

            engine_->spatial().col2im(cols, image, g, out_h, out_w);
            ++emitted_;
        }

        void Lowering::record_maxpool2d(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 2,
                        "MAXPOOL2D (node {}) takes one operand and two results "
                        "(the pooled value and its argmax indices)", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);
            SushiBLAS::Tensor& indices = output(graph, node, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            engine_->spatial().pool_max(in, out, indices, g);
            ++emitted_;
        }

        void Lowering::record_maxpool2d_backward(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 2 || node.outputs.size() != 1,
                        "MAXPOOL2D_BACKWARD (node {}) takes a gradient and the forward's "
                        "indices, and produces one result", id);

            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& indices = input(node, 1);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            engine_->spatial().pool_max_backward(dy, indices, dx, g);
            ++emitted_;
        }

        void Lowering::record_avgpool2d(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "AVGPOOL2D (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            engine_->spatial().pool_avg(in, out, g);
            ++emitted_;
        }

        void Lowering::record_avgpool2d_backward(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "AVGPOOL2D_BACKWARD (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& dy = input(node, 0);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(node);

            engine_->spatial().pool_avg_backward(dy, dx, g);
            ++emitted_;
        }

        void Lowering::record_upsample2d(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "UPSAMPLE2D (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            const int64_t scale_h = static_cast<int64_t>(node.attributes.get(0));
            const int64_t scale_w = static_cast<int64_t>(node.attributes.get(1));

            engine_->spatial().upsample_nearest(in, out, scale_h, scale_w);
            ++emitted_;
        }

        void Lowering::record_upsample2d_backward(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "UPSAMPLE2D_BACKWARD (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& dy = input(node, 0);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            const int64_t scale_h = static_cast<int64_t>(node.attributes.get(0));
            const int64_t scale_w = static_cast<int64_t>(node.attributes.get(1));

            engine_->spatial().upsample_nearest_backward(dy, dx, scale_h, scale_w);
            ++emitted_;
        }

        void Lowering::record_spatial_mean(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "SPATIAL_MEAN (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            engine_->reductions().mean_spatial(in, out);
            ++emitted_;
        }

        void Lowering::record_spatial_mean_backward(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "SPATIAL_MEAN_BACKWARD (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& dy = input(node, 0);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            // Derives spatial normalization factor from destination gradient dimensions.
            const Shape& dx_shape = shape_of(graph, node.outputs[0]);
            SA_THROW_IF(dx_shape.rank() != 4,
                        "SPATIAL_MEAN_BACKWARD (node {}) needs a rank-4 [N,H,W,C] result to read "
                        "H and W from; got {}", id, dx_shape.to_string());

            const double scale = 1.0 / static_cast<double>(dx_shape[1] * dx_shape[2]);

            engine_->spatial().broadcast_spatial(dy, dx, scale);
            ++emitted_;
        }

        void Lowering::record_concat_channels(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.empty() || node.outputs.size() != 1,
                        "CONCAT_CHANNELS (node {}) takes at least one operand and one result",
                        id);

            SushiBLAS::Tensor& out = output(graph, node, 0);

            // Appends channel slices sequentially while tracking cumulative offset.
            int64_t offset = 0;
            for (std::size_t i = 0; i < node.inputs.size(); ++i)
            {
                const SushiBLAS::Tensor& src = input(node, i);
                engine_->spatial().concat_channels(src, out, offset);
                offset += shape_of(graph, node.inputs[i])[3];
            }

            // One copy per operand, not one for the node.
            emitted_ += node.inputs.size();
        }

        void Lowering::record_slice_channels(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "SLICE_CHANNELS (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            const int64_t offset = static_cast<int64_t>(node.attributes.get(0));

            engine_->spatial().slice_channels(in, offset, out);
            ++emitted_;
        }

        namespace
        {
            /**
             * @brief Creates zero-copy `[N*H*W, C]` rank-2 view of an NHWC tensor.
             * @param t Source NHWC tensor.
             * @param rows Flattened spatial-batch dimension.
             * @param cols Channel dimension.
             * @return Rank-2 tensor sharing underlying storage.
             */
            SushiBLAS::Tensor channel_rows(const SushiBLAS::Tensor& t, int64_t rows, int64_t cols)
            {
                const int64_t extents[2] = {rows, cols};
                return SushiBLAS::Tensor(
                    t.storage, SushiRuntime::span<const int64_t>(extents, 2),
                    t.dtype, t.storage_offset, t.layout);
            }
        }

        void Lowering::record_channel_affine(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 3 || node.outputs.size() != 1,
                        "CHANNEL_AFFINE (node {}) takes three operands and one result", id);

            const SushiBLAS::Tensor& in = input(node, 0);
            const SushiBLAS::Tensor& gamma = input(node, 1);
            const SushiBLAS::Tensor& beta = input(node, 2);
            SushiBLAS::Tensor& out = output(graph, node, 0);

            engine_->spatial().channel_affine(in, gamma, beta, out);
            ++emitted_;
        }

        void Lowering::record_batchnorm2d(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 3 || node.outputs.size() != 3,
                        "BATCHNORM2D (node {}) takes three operands and three results - the "
                        "normalised value and the two statistics its own backward reads", id);

            const Shape& x_shape = shape_of(graph, node.inputs[0]);
            SA_THROW_IF(x_shape.rank() != 4,
                        "BATCHNORM2D (node {}) needs a rank-4 [N,H,W,C] operand; got {}",
                        id, x_shape.to_string());

            const SushiBLAS::Tensor& x = input(node, 0);
            SA_THROW_IF(!x.is_contiguous(),
                        "BATCHNORM2D (node {}) folds its statistics over a rank-2 view of its "
                        "operand, which a strided view would read wrong", id);

            const SushiBLAS::Tensor& gamma = input(node, 1);
            const SushiBLAS::Tensor& beta = input(node, 2);

            SushiBLAS::Tensor& y = output(graph, node, 0);
            SushiBLAS::Tensor& mean = output(graph, node, 1);
            SushiBLAS::Tensor& inv_std = output(graph, node, 2);

            const DType dtype = dtype_of(graph, node.inputs[0]);
            const int64_t channels = x_shape[3];
            const int64_t rows = x_shape[0] * x_shape[1] * x_shape[2];
            const double inverse_rows = 1.0 / static_cast<double>(rows);
            const double epsilon = node.attributes.get(0);

            const Shape per_channel({channels});
            SushiBLAS::Tensor& ones = pool_->ones(per_channel, dtype);
            SushiBLAS::Tensor& negated_mean = pool_->scratch(id, 0, per_channel, dtype);
            SushiBLAS::Tensor& centred = pool_->scratch(id, 1, x_shape, dtype);
            SushiBLAS::Tensor& squared = pool_->scratch(id, 2, x_shape, dtype);
            SushiBLAS::Tensor& variance = pool_->scratch(id, 3, per_channel, dtype);
            SushiBLAS::Tensor& scale = pool_->scratch(id, 4, per_channel, dtype);

            // Computes batch channel mean via column reduction over flattened spatial view.
            const SushiBLAS::Tensor x_rows = channel_rows(x, rows, channels);
            engine_->reductions().sum_along(x_rows, 0, mean);
            engine_->blas().scal(inverse_rows, mean);

            // Computes channel variance using two-pass numerically stable formulation.
            engine_->blas().copy(mean, negated_mean);
            engine_->blas().scal(-1.0, negated_mean);
            engine_->spatial().channel_affine(x, ones, negated_mean, centred);

            engine_->elementwise().mul(centred, centred, squared);
            const SushiBLAS::Tensor squared_rows = channel_rows(squared, rows, channels);
            engine_->reductions().sum_along(squared_rows, 0, variance);
            engine_->blas().scal(inverse_rows, variance);

            // Adds variance epsilon via axpy with shared ones vector.
            engine_->blas().axpy(epsilon, ones, variance);

            engine_->blas().copy(variance, inv_std);
            engine_->elementwise().sqrt(inv_std);
            engine_->elementwise().reciprocal(inv_std);

            // Scales normalized activations with learnable affine parameters.
            engine_->elementwise().mul(gamma, inv_std, scale);
            engine_->spatial().channel_affine(centred, scale, beta, y);

            // Dispatches composite forward batch-normalization operations.
            emitted_ += 14;
        }

        void Lowering::record_batchnorm2d_backward(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 5 || node.outputs.size() != 3,
                        "BATCHNORM2D_BACKWARD (node {}) takes the output gradient, the forward's "
                        "input and gamma, and the two saved statistics, and produces three "
                        "gradients", id);

            const Shape& x_shape = shape_of(graph, node.inputs[1]);
            SA_THROW_IF(x_shape.rank() != 4,
                        "BATCHNORM2D_BACKWARD (node {}) needs a rank-4 [N,H,W,C] input; got {}",
                        id, x_shape.to_string());

            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& x = input(node, 1);
            const SushiBLAS::Tensor& gamma = input(node, 2);
            const SushiBLAS::Tensor& mean = input(node, 3);
            const SushiBLAS::Tensor& inv_std = input(node, 4);

            SA_THROW_IF(!dy.is_contiguous() || !x.is_contiguous(),
                        "BATCHNORM2D_BACKWARD (node {}) folds over rank-2 views of its two "
                        "rank-4 operands, which a strided view would read wrong", id);

            SushiBLAS::Tensor& dx = output(graph, node, 0);
            SushiBLAS::Tensor& dgamma = output(graph, node, 1);
            SushiBLAS::Tensor& dbeta = output(graph, node, 2);

            const DType dtype = dtype_of(graph, node.inputs[1]);
            const int64_t channels = x_shape[3];
            const int64_t rows = x_shape[0] * x_shape[1] * x_shape[2];
            const double inverse_rows = 1.0 / static_cast<double>(rows);

            const Shape per_channel({channels});
            SushiBLAS::Tensor& shift = pool_->scratch(id, 0, per_channel, dtype);
            SushiBLAS::Tensor& x_hat = pool_->scratch(id, 1, x_shape, dtype);
            SushiBLAS::Tensor& product = pool_->scratch(id, 2, x_shape, dtype);
            SushiBLAS::Tensor& direct = pool_->scratch(id, 3, per_channel, dtype);
            SushiBLAS::Tensor& hat_term = pool_->scratch(id, 4, per_channel, dtype);
            SushiBLAS::Tensor& mean_term = pool_->scratch(id, 5, per_channel, dtype);
            SushiBLAS::Tensor& correction = pool_->scratch(id, 6, x_shape, dtype);
            SushiBLAS::Tensor& zero = pool_->scratch(id, 7, per_channel, dtype);

            // Reconstructs normalized activations using saved forward statistics.
            engine_->elementwise().mul(mean, inv_std, shift);
            engine_->blas().scal(-1.0, shift);
            engine_->spatial().channel_affine(x, inv_std, shift, x_hat);

            // Computes parameter gradients through column reductions.
            const SushiBLAS::Tensor dy_rows = channel_rows(dy, rows, channels);
            engine_->reductions().sum_along(dy_rows, 0, dbeta);

            engine_->elementwise().mul(dy, x_hat, product);
            const SushiBLAS::Tensor product_rows = channel_rows(product, rows, channels);
            engine_->reductions().sum_along(product_rows, 0, dgamma);

            // Formulates input gradient evaluation via affine channel combination.
            engine_->elementwise().mul(gamma, inv_std, direct);

            engine_->elementwise().mul(direct, dgamma, hat_term);
            engine_->blas().scal(-inverse_rows, hat_term);

            engine_->elementwise().mul(direct, dbeta, mean_term);
            engine_->blas().scal(-inverse_rows, mean_term);

            engine_->random().constant(zero, 0.0);
            engine_->spatial().channel_affine(dy, direct, zero, dx);
            engine_->spatial().channel_affine(x_hat, hat_term, mean_term, correction);
            engine_->blas().axpy(1.0, correction, dx);

            // Dispatches composite backward batch-normalization operations.
            emitted_ += 15;
        }

    } // namespace Graph
} // namespace SushiAI
