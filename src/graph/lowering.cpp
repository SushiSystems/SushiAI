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
            // template was instantiated with it, and these five instantiations
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

            /**
             * @brief Adds a per-column bias, **spills it**, then SushiBLAS's GELU.
             *
             * cuBLASLt calls this @c GELU_AUX_BIAS, and the auxiliary store is
             * not an optimisation - it is the only reason this composition is
             * allowed to exist. @c gelu_backward differentiates at the GELU's
             * *input*; the other three epilogues may throw that tensor away
             * because their gradients read the output instead. This one writes
             * it out on the way past, so the value the backward rule reads is
             * the value the unfused @c BIAS_ADD would have left there - the same
             * graph value, under the same id. See graph/fusion.hpp's `spill`
             * section and @c Ops::GEMM_BIAS_GELU.
             *
             * WHY INDEXING OFF row AND col IS EXACTLY-ONCE AND IN RANGE
             *
             * Both SYCL variants of @c Kernels::SYCL::gemm skip an out-of-range
             * element *before* calling the epilogue
             * (`if (global_row >= m) continue;` and the same for the column),
             * and @c Kernels::Reference::gemm walks `i < m, j < n` directly. So
             * the functor is invoked once per element of C and never outside
             * it, and the store below lands on each element of the spill
             * exactly once.
             *
             * The one thing @c row and @c col do *not* carry is a batch index -
             * the reference kernel's own documentation says so, and the SYCL
             * side does not supply one either, so every matrix of a batch would
             * be spilled on top of the last. That is safe here because
             * @c record_fused_gemm refuses a result that is not rank 2, which
             * makes @c execute_gemm's `batch_size` exactly 1. The refusal is
             * what this comment depends on; do not relax one without the other.
             *
             * The strides come from the aux tensor rather than being assumed,
             * because the spill may be a view into a planned arena and may be
             * column-major: @c Tensor::strides is what @c Tensor lays a shape
             * out with, so `row * strides[0] + col * strides[1]` is the element
             * under either layout, and @c Tensor::data() has already folded in
             * @c storage_offset.
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

                    // SushiBLAS's own gelu expression, copied rather than
                    // paraphrased - the same reason the three above are. A
                    // re-derivation that groups the terms differently would be
                    // correct mathematics and a different float, and fusion
                    // promises to move no bits.
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
            selections_.clear();

            // The last gate, and the only unconditional one. finalise() proves
            // CV-4 over the forward graph, and fuse() and insert_casts() each
            // re-prove it over the graph they rebuild - but a caller is free
            // to differentiate and lower directly, and the nodes autograd
            // emits would then be proved nowhere. Lowering is the single
            // funnel every path goes through, so the check belongs here too:
            // it makes the pass-level proofs belt-and-braces rather than
            // load-bearing, and it is O(nodes) against a walk already
            // O(nodes).
            //
            // It matters here more than anywhere else, because this is where
            // the tag stops being bookkeeping: record_layout_convert reads it
            // on both sides to decide whether to relabel one into [N,H,W,C],
            // and a wrong tag becomes a transpose that does not happen.
            graph.prove_layout_uniformity();

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

            // "A shorter entry of the same family" is exactly one entry -
            // GEMM_BIAS, the only one that folds no activation - and it is
            // spelled out rather than approximated by "both are GEMM
            // epilogues". That looser test let a selector answer
            // gemm+bias+tanh to a gemm+bias+relu match, which passes every
            // shape check and applies the wrong activation, and once a form
            // could spill it also let one answer gemm+bias+gelu to a node with
            // no auxiliary output for the epilogue to write.
            const bool narrowed_within_family =
                is_gemm_epilogue(query.matched) && choice.form == FusedForm::GEMM_BIAS;

            SA_THROW_IF(choice.form != FusedForm::NONE && choice.form != query.matched
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
            if (node.op == Ops::DYNAMIC_SCALE.id)          { record_dynamic_scale(graph, id, node); return; }
            if (node.op == Ops::EXP.id
                || node.op == Ops::RELU.id
                || node.op == Ops::GELU.id
                || node.op == Ops::SIGMOID.id
                || node.op == Ops::TANH.id)                { record_unary_inplace(graph, node); return; }
            if (node.op == Ops::LEAKY_RELU.id)             { record_leaky_relu(graph, node); return; }
            if (node.op == Ops::RELU_BACKWARD.id
                || node.op == Ops::GELU_BACKWARD.id
                || node.op == Ops::SIGMOID_BACKWARD.id
                || node.op == Ops::TANH_BACKWARD.id)       { record_activation_backward(graph, node); return; }
            if (node.op == Ops::LEAKY_RELU_BACKWARD.id)    { record_leaky_relu_backward(graph, node); return; }
            if (node.op == Ops::LOG_SOFTMAX.id)            { record_log_softmax(graph, id, node); return; }
            if (node.op == Ops::NLL_LOSS.id)               { record_nll_loss(graph, id, node); return; }
            if (node.op == Ops::SUM_ROWS.id)               { record_sum_rows(graph, node); return; }
            if (node.op == Ops::ACCUMULATE.id)             { record_accumulate(node); return; }
            if (node.op == Ops::ZERO.id)                   { record_zero(graph, node); return; }
            if (node.op == Ops::CAST.id)                   { record_cast(graph, id, node); return; }
            if (node.op == Ops::RESHAPE.id)                { record_reshape(graph, id, node); return; }
            if (node.op == Ops::LAYOUT_CONVERT.id)         { record_layout_convert(graph, id, node); return; }
            if (node.op == Ops::IM2COL.id)                 { record_im2col(graph, id, node); return; }
            if (node.op == Ops::COL2IM.id)                 { record_col2im(graph, id, node); return; }
            if (node.op == Ops::MAXPOOL2D.id)              { record_maxpool2d(graph, id, node); return; }
            if (node.op == Ops::MAXPOOL2D_BACKWARD.id)     { record_maxpool2d_backward(graph, id, node); return; }
            if (node.op == Ops::AVGPOOL2D.id)              { record_avgpool2d(graph, id, node); return; }
            if (node.op == Ops::AVGPOOL2D_BACKWARD.id)     { record_avgpool2d_backward(graph, id, node); return; }
            if (node.op == Ops::UPSAMPLE2D.id)             { record_upsample2d(graph, id, node); return; }
            if (node.op == Ops::UPSAMPLE2D_BACKWARD.id)    { record_upsample2d_backward(graph, id, node); return; }
            if (node.op == Ops::SPATIAL_MEAN.id)           { record_spatial_mean(graph, id, node); return; }
            if (node.op == Ops::SPATIAL_MEAN_BACKWARD.id)  { record_spatial_mean_backward(graph, id, node); return; }
            if (node.op == Ops::CONCAT_CHANNELS.id)        { record_concat_channels(graph, id, node); return; }
            if (node.op == Ops::SLICE_CHANNELS.id)         { record_slice_channels(graph, id, node); return; }
            if (node.op == Ops::CHANNEL_AFFINE.id)         { record_channel_affine(graph, id, node); return; }
            if (node.op == Ops::BATCHNORM2D.id)            { record_batchnorm2d(graph, id, node); return; }
            if (node.op == Ops::BATCHNORM2D_BACKWARD.id)   { record_batchnorm2d_backward(graph, id, node); return; }

            const FusedForm form = fused_form_of(node.op);
            if (is_gemm_epilogue(form))       { record_fused_gemm(graph, id, node, form); return; }
            if (form == FusedForm::EXP_SUB_SCALE)
            {
                record_fused_exp_sub_scale(graph, id, node, form);
                return;
            }

            // Three declared identities reach this throw on purpose:
            // SGD_UPDATE and ADAMW_UPDATE, which src/optim/{sgd,adamw}.cpp
            // record directly into the step, and BATCHNORM_UPDATE_STATS,
            // which CV-7 puts in the same class - a BatchNorm's running
            // statistics are written beside the graph, not through it.
            //
            // None of the three is a missing case. A node carrying one is a
            // graph that was built wrong, and this is the correct answer. See
            // Integration_Resilience.BatchNormUpdateStatsIsNotAnIrOperationAndSaysSo,
            // which pins that for the one of the three a graph builder could
            // plausibly emit by mistake.
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
            // Two results for the one entry that spills its pre-activation, one
            // for the rest. Checked against the catalog rather than merely
            // "one or two", because a node arriving with the wrong number is a
            // fusion-pass defect and the message should say which shape was
            // expected.
            const bool spills = fused_form_spills(matched);
            const std::size_t results = spills ? 2u : 1u;

            SA_THROW_IF(node.inputs.size() != 3 || node.outputs.size() != results,
                        "{} (node {}) takes three operands - A, B and the bias - and {} result{}",
                        Ops::name_of(node.op), id, results, results == 1 ? "" : "s");

            const Shape& out_shape = shape_of(graph, node.outputs[0]);

            // Rank 2 is a correctness constraint and not tidiness for the
            // spilling arm: execute_gemm derives `batch_size` from every
            // dimension of C before the trailing two, and the epilogue is
            // handed a (row, col) inside one matrix with no batch index - so a
            // batched spill would write every matrix over the last one. See
            // BiasGeluSpillEpilogue.
            SA_THROW_IF(out_shape.rank() != 2,
                        "{} (node {}) needs a rank-2 result; got {}",
                        Ops::name_of(node.op), id, out_shape.to_string());

            const SushiBLAS::Tensor& a = input(node, 0);
            const SushiBLAS::Tensor& b = input(node, 1);
            const SushiBLAS::Tensor& bias = input(node, 2);
            SushiBLAS::Tensor& c = output(graph, node, 0);

            // Materialised before the switch, because both the fused arm and
            // every declining one have to write it - a spill the selector
            // talked us out of is still a spill the backward pass reads.
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
                const void* const values = bias.data();

                // `.reads(bias)` is the whole of what used to be a separate
                // host-side ordering node. SushiBLAS declares A, B and C and
                // cannot see inside the functor, so the bias the epilogue
                // dereferences has to be declared by whoever built the functor
                // around it - which is here. It is not optional bookkeeping:
                // the bias is a parameter the optimizer rewrites inside this
                // same compiled plan, so without the declaration the update is
                // free to land before the forward pass has read the old value,
                // and the run still trains.
                //
                // Declared per arm rather than once above the switch on
                // purpose: the access set belongs to the spelling. An arm that
                // spilled a pre-activation would declare a *write* the others
                // do not, and hoisting the declaration out of the arm that
                // knows it is how the two drift apart.
                //
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
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasReluEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GEMM_BIAS_SIGMOID:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_SIGMOID.name.data(), Ops::GEMM_BIAS_SIGMOID.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasSigmoidEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GEMM_BIAS_TANH:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_TANH.name.data(), Ops::GEMM_BIAS_TANH.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasTanhEpilogue{values}).reads(bias));
                        break;

                    case FusedForm::GEMM_BIAS_GELU:
                        // The arm the paragraph above is about: `.writes(aux)`
                        // is the declaration no other arm makes, and without it
                        // the backward pass that reads the spill races the GEMM
                        // that fills it - which resolves the right way often
                        // enough to look like training.
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS_GELU.name.data(), Ops::GEMM_BIAS_GELU.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasGeluSpillEpilogue{values, aux->data(),
                                                      aux->strides[0], aux->strides[1]})
                                .reads(bias)
                                .writes(*aux));
                        break;

                    default:
                        SushiBLAS::Dispatch::execute_gemm(
                            *engine_, a, b, c, transpose_a, transpose_b, 1.0, 0.0,
                            Ops::GEMM_BIAS.name.data(), Ops::GEMM_BIAS.id,
                            SushiBLAS::Dispatch::gemm_epilogue(
                                BiasEpilogue{values}).reads(bias));
                        break;
                }
                ++emitted_;
            }

            // A spelling that did not fold the activation left the
            // pre-activation sitting in C, which is precisely what the spill is
            // supposed to contain - so one copy is the whole of it, and it has
            // to happen here, before the trailing activation overwrites C in
            // place. This is what makes a selector refusal cost launches and
            // change no number: the aux buffer holds the same bytes either way.
            if (spills && !fused_form_spills(choice.form))
            {
                engine_->blas().copy(c, *aux);
                ++emitted_;
            }

            // Whatever activation the chosen spelling did not fold in still has
            // to run, in place on the result. Correct for every activation in
            // Ops::ACTIVATIONS, fused or not: for the three whose gradient
            // reads their output the fused node's result *is* the activation's
            // output value, and for GELU the pre-activation the gradient wants
            // has just been spilled above.
            const OpID activation = fused_form_activation(matched);
            if (activation != Ops::LEAF.id && fused_form_activation(choice.form) != activation)
            {
                if (activation == Ops::RELU.id)         engine_->nonlinear().relu(c);
                else if (activation == Ops::SIGMOID.id) engine_->nonlinear().sigmoid(c);
                else if (activation == Ops::GELU.id)    engine_->nonlinear().gelu(c);
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

            // Copy first, then multiply in place, which is exactly the shape
            // record_scale() uses - so the two spellings differ in one thing
            // only, where the factor comes from.
            engine_->blas().copy(x, y);

            // The factor's *address* is captured, never its value. That is the
            // whole point of this operation: a captured float would be frozen
            // into the recorded task and changing it would force a recompile,
            // which is what Attributes' warning is about and what
            // CompiledStep::compile_count() == 1 is asserted against.
            //
            // The buffer is not declared as a read of this task, and that is
            // sound for exactly the access pattern optim/step_scalars.hpp
            // argues for and no other: it is written only from the host,
            // between replays, at a point where the previous replay has been
            // waited on and the next has not been dispatched. A *device* writer
            // would race, because there is no edge to order it.
            constexpr const char* TASK_NAME = "sushiai.dynamic_scale";
            static_assert(Ops::DYNAMIC_SCALE.name == std::string_view(TASK_NAME),
                          "the task name must be the string its OpID hashes");

            SushiBLAS::Dispatch::execute_unary_inplace(*engine_, y,
                TASK_NAME, Ops::DYNAMIC_SCALE.id,
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

        void Lowering::record_leaky_relu(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& x = input(node, 0);
            SushiBLAS::Tensor& y = output(graph, node, 0);

            // Copy first, same as record_unary_inplace: the forward input
            // survives for the backward rule, which is what makes this
            // differentiable without a separate save-for-backward mechanism.
            engine_->blas().copy(x, y);
            engine_->nonlinear().leaky_relu(y, static_cast<float>(node.attributes.get(0)));

            emitted_ += 2;
        }

        void Lowering::record_leaky_relu_backward(const Graph& graph, const Node& node)
        {
            const SushiBLAS::Tensor& dy = input(node, 0);
            const SushiBLAS::Tensor& primal = input(node, 1);
            SushiBLAS::Tensor& dx = output(graph, node, 0);

            // The slope isn't carried on this node - it's read out of the
            // forward LEAKY_RELU node's attribute slot 0 by
            // GraphBuilder::activation_backward, which forwards the whole
            // Attributes through unchanged, so it lands here in the same slot.
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

            // 1. Each row's OWN maximum, not the batch's.
            //
            //    This used to be a whole-tensor `reductions().max`, broadcast to
            //    every row, on the argument that log-softmax is invariant to
            //    subtracting any single scalar from a row. The argument is sound
            //    in real arithmetic and false in binary32, because what destroys
            //    the invariance is not the subtraction but the exponential below
            //    it: a row sitting far under the batch maximum has every
            //    `exp(x - batch_max)` collapse through the subnormals to zero, so
            //    its row sum is exactly zero, `log(0)` is `-inf`, and step 4
            //    returns `+inf` for the whole row. Measured on this project's
            //    OpenCL CPU device over a row whose true log-probabilities are
            //    around -0.41 to -2.41, as the worst absolute deviation from the
            //    answer the same code gives at a gap of zero: 2.15e-06 at a gap
            //    of 90, 2.70e-02 at 100, and `+inf` from 104 - which then makes
            //    the loss NaN and the gradient non-finite, so the run trains on
            //    NaN and reports success. Shifting by the row's own maximum takes
            //    the gap out of the arithmetic entirely rather than moving the
            //    threshold, so that deviation is exactly zero at every gap.
            //    tests/functional/regression/test_log_softmax_stability.cpp pins
            //    every one of those numbers.
            //
            //    It also costs one recorded call less than the global shift did,
            //    because a per-row vector needs no broadcast back out to a row.
            engine_->reductions().max_along(x, 1, row_max);

            // 2. shifted = X - (row_max) 1^T; every exponent is now at most zero,
            //    and every row has at least one exponent that is exactly zero.
            engine_->blas().copy(x, shifted);
            engine_->blas().ger(row_max, ones_cols, shifted, -1.0f);

            // 3. row_sums = rowwise sum of exp(shifted). Still a gemv against a
            //    ones vector rather than the new `sum_along`: both are correct
            //    here - every summand is in (0, 1] after step 2, so there is
            //    nothing for a compensated fold to recover - and leaving it alone
            //    keeps this change to the one line that was wrong.
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

        void Lowering::record_cast(const Graph& graph, NodeId id, const Node& node)
        {
            SA_THROW_IF(node.inputs.size() != 1 || node.outputs.size() != 1,
                        "CAST (node {}) takes one operand and one result", id);

            const SushiBLAS::Tensor& source = input(node, 0);
            SushiBLAS::Tensor& destination = output(graph, node, 0);

            // `elementwise().convert`, which is the one SushiBLAS operation
            // whose two operands may disagree on dtype - `blas().copy` refuses
            // a mismatch, and every other family dispatches on a single shared
            // element type. It declares `src.region()` and `dst.region()` to the
            // dependency tracker rather than bare pointers, which is what makes
            // this correct for two arena views: the fp16 copy of a parameter is
            // read by a GEMM and the parameter itself is rewritten by the
            // optimizer inside the same compiled plan, so both edges have to
            // land or the update could overtake the conversion and still train.
            //
            // The planner cannot overlap these two buffers even though the
            // source dies exactly where the destination is born: `precedes()`
            // asks whether every use of the earlier buffer *strictly* reaches
            // the later one's first write, and a node never reaches itself.
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

            // A reshape moves no data - it is the same bytes read under
            // different extents - so the "copy" here is `blas().copy` walking
            // the same linear order twice, not a permutation of any kind.
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

            // `SpatialOps::layout_convert` reads its direction purely from the
            // two operands' strides: both must be handed in shaped [N, H, W,
            // C], so whichever side is physically NCHW ([N, C, H, W]) is
            // relabelled into that order first. Two transposes rather than one
            // because NCHW -> NHWC is the 3-cycle (C, H, W) -> (H, W, C), not
            // a single swap: transpose(1, 2) gives [N, H, C, W], and
            // transpose(2, 3) on that gives [N, H, W, C] with the strides
            // carried along at each step, so no data moves until the kernel
            // itself runs.
            const SushiBLAS::Tensor source_view = (source_layout == TensorLayout::NCHW)
                ? source.transpose(1, 2).transpose(2, 3) : source;
            SushiBLAS::Tensor dest_view = (dest_layout == TensorLayout::NCHW)
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
             * @brief Reads the window grid a windowed node carries in slots 8-9.
             *
             * Read, never re-derived. Deriving the extents from the geometry
             * assumes symmetric padding, and an asymmetric grid can derive to
             * its own transpose with the same window count - which the row
             * check and the origin checks both accept, so the gather returns
             * wrong numbers and nothing throws.
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

            // The divisor is read from dx's own shape - the [N,H,W,C] gradient
            // this node is about to fill - rather than carried as a separate
            // attribute, so the forward's mean_spatial and this broadcast can
            // never disagree about what H*W was.
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

            // concat_channels copies one operand at a time; the running offset
            // is the running channel count of the operands already placed, in
            // the same order the forward's shape check walked them.
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
             * @brief The zero-copy `[N*H*W, C]` view of an NHWC tensor.
             *
             * Channels innermost is what makes this free: the rank-4 and the
             * rank-2 readings are the same bytes in the same order, so a
             * per-channel fold is a column sum of this view and needs no new
             * kernel. Built the way record_reshape builds its own view, from
             * the storage and offset rather than from a pointer, so an operand
             * living inside a planned arena is addressed correctly.
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

            // The batch mean: a column sum of the [N*H*W, C] view, which is
            // the fold the design says NHWC already spells with no new kernel.
            const SushiBLAS::Tensor x_rows = channel_rows(x, rows, channels);
            engine_->reductions().sum_along(x_rows, 0, mean);
            engine_->blas().scal(inverse_rows, mean);

            // Two passes rather than E[x*x] - E[x]*E[x]. The one-pass form is a
            // subtraction of two numbers that are equal to several digits when
            // the channel's mean is large next to its spread, and it loses
            // exactly those digits - which shows up as a variance that is
            // slightly wrong, or negative, and a normalisation that trains
            // anyway. The second pass costs one traversal and is exact.
            engine_->blas().copy(mean, negated_mean);
            engine_->blas().scal(-1.0, negated_mean);
            engine_->spatial().channel_affine(x, ones, negated_mean, centred);

            engine_->elementwise().mul(centred, centred, squared);
            const SushiBLAS::Tensor squared_rows = channel_rows(squared, rows, channels);
            engine_->reductions().sum_along(squared_rows, 0, variance);
            engine_->blas().scal(inverse_rows, variance);

            // epsilon added through the shared ones vector rather than a
            // constant fill, so nothing here allocates a buffer to hold one
            // repeated number.
            engine_->blas().axpy(epsilon, ones, variance);

            engine_->blas().copy(variance, inv_std);
            engine_->elementwise().sqrt(inv_std);
            engine_->elementwise().reciprocal(inv_std);

            // y = gamma * x_hat + beta, and x_hat = centred * inv_std, so the
            // whole apply half is one channel_affine over the centred value.
            engine_->elementwise().mul(gamma, inv_std, scale);
            engine_->spatial().channel_affine(centred, scale, beta, y);

            // Fourteen: two sum_along, four scal, two copy, two channel_affine,
            // two mul, one axpy, sqrt and reciprocal. pool_->ones() is not
            // counted, matching record_bias_add - the vector is shared and its
            // fill is not this node's work.
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

            // x_hat = (x - mean) * inv_std, rebuilt from the statistics the
            // forward saved rather than from a fresh reduction over x: the
            // forward normalised by *these* numbers, and a second reduction
            // over a buffer a later node may have written would differentiate
            // a different function.
            engine_->elementwise().mul(mean, inv_std, shift);
            engine_->blas().scal(-1.0, shift);
            engine_->spatial().channel_affine(x, inv_std, shift, x_hat);

            // The two reductions that make this not an elementwise scale.
            // dbeta is the plain sum of dy; dgamma is the sum of dy * x_hat.
            const SushiBLAS::Tensor dy_rows = channel_rows(dy, rows, channels);
            engine_->reductions().sum_along(dy_rows, 0, dbeta);

            engine_->elementwise().mul(dy, x_hat, product);
            const SushiBLAS::Tensor product_rows = channel_rows(product, rows, channels);
            engine_->reductions().sum_along(product_rows, 0, dgamma);

            // dx = (gamma * inv_std / M) * (M * dy - dbeta - x_hat * dgamma),
            // regrouped per channel as
            //     dx = direct * dy + hat_term * x_hat + mean_term
            // so that the whole thing is two channel_affine passes and an axpy
            // rather than a broadcast of each reduction in turn.
            engine_->elementwise().mul(gamma, inv_std, direct);

            engine_->elementwise().mul(direct, dgamma, hat_term);
            engine_->blas().scal(-inverse_rows, hat_term);

            engine_->elementwise().mul(direct, dbeta, mean_term);
            engine_->blas().scal(-inverse_rows, mean_term);

            engine_->random().constant(zero, 0.0);
            engine_->spatial().channel_affine(dy, direct, zero, dx);
            engine_->spatial().channel_affine(x_hat, hat_term, mean_term, correction);
            engine_->blas().axpy(1.0, correction, dx);

            // Fifteen: two sum_along, five mul, three scal, three
            // channel_affine, one constant fill and one axpy.
            emitted_ += 15;
        }

    } // namespace Graph
} // namespace SushiAI
