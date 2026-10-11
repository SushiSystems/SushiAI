/**************************************************************************/
/* context.hpp                                                            */
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
 * @file context.hpp
 * @brief Declares the LoweringContext that a lowering row reads its node through.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Maps a bool transpose flag to the SushiBLAS transpose enumerator.
         * @param transposed True for Transpose::Trans, false for Transpose::NoTrans.
         * @return The matching SushiBLAS transpose value.
         */
        [[nodiscard]] constexpr SushiBLAS::Core::Transpose transpose_of(bool transposed) noexcept
        {
            return transposed ? SushiBLAS::Core::Transpose::Trans
                              : SushiBLAS::Core::Transpose::NoTrans;
        }

        /**
         * @class LoweringContext
         * @brief Gives one lowering row the recorder, the pool and the node it lowers.
         *
         * Holds references to the state of one Lowering::record pass for the duration of
         * one row call. It owns none of that state and is not copyable or movable.
         */
        class LoweringContext final
        {
            public:
                /**
                 * @brief Binds the context to one node of one recording pass.
                 * @param id A node of @p graph.
                 * @param selections The pass's list of fusion decisions; select() appends to it.
                 * @param emitted The pass's count of recorded tasks; count() adds to it.
                 * @pre Every reference argument outlives the context.
                 */
                LoweringContext(SushiBLAS::TaskRecorder& recorder, SushiBLAS::RNGStream& stream,
                                TensorPool& pool, const Graph& graph, NodeID id,
                                const FusionSelector& selector,
                                std::vector<FusionSelection>& selections,
                                std::size_t& emitted) noexcept;

                /**
                 * @brief Binds the context as the constructor above does, and to the pass's
                 *        list of replay preludes.
                 * @param preludes The list before_each_replay() appends to.
                 * @pre Every reference argument outlives the context.
                 */
                LoweringContext(SushiBLAS::TaskRecorder& recorder, SushiBLAS::RNGStream& stream,
                                TensorPool& pool, const Graph& graph, NodeID id,
                                const FusionSelector& selector,
                                std::vector<FusionSelection>& selections,
                                std::size_t& emitted,
                                std::vector<ReplayPrelude>& preludes) noexcept;

                LoweringContext(const LoweringContext&) = delete;
                LoweringContext& operator=(const LoweringContext&) = delete;
                LoweringContext(LoweringContext&&) = delete;
                LoweringContext& operator=(LoweringContext&&) = delete;
                ~LoweringContext() = default;

                /** @brief The recorder the row records its SushiBLAS tasks on. */
                [[nodiscard]] SushiBLAS::TaskRecorder& recorder() const noexcept;

                /** @brief The random stream that random operations draw from. */
                [[nodiscard]] SushiBLAS::RNGStream& stream() const noexcept;

                /** @brief The graph being lowered. */
                [[nodiscard]] const Graph& graph() const noexcept;

                /** @brief The id of the node this context describes. */
                [[nodiscard]] NodeID id() const noexcept;

                /**
                 * @brief The node this context describes.
                 * @return A reference into the graph, valid while the graph is unchanged.
                 */
                [[nodiscard]] const Node& node() const;

                /**
                 * @brief The declared name of the node's operation.
                 * @return A static string from the operation table, used in diagnostics.
                 */
                [[nodiscard]] std::string_view name() const;

                /**
                 * @brief Stops the process unless the node has exactly the given operand counts.
                 * @param inputs The required number of operands.
                 * @param outputs The required number of results.
                 */
                void require_arity(std::size_t inputs, std::size_t outputs) const;

                /**
                 * @brief Stops the process unless the node has at least the given operand counts.
                 * @param inputs The minimum number of operands.
                 * @param outputs The minimum number of results.
                 */
                void require_arity_at_least(std::size_t inputs, std::size_t outputs) const;

                /**
                 * @brief Stops the process unless the node has at least the given number of
                 *        operands and exactly the given number of results.
                 * @param inputs The minimum number of operands.
                 * @param outputs The required number of results.
                 */
                void require_variadic_arity(std::size_t inputs, std::size_t outputs) const;

                /**
                 * @brief The graph description of one operand.
                 * @param index Position in the node's operand list; must be in range.
                 * @return A reference into the graph, valid while the graph is unchanged.
                 */
                [[nodiscard]] const Value& input_value(std::size_t index) const;

                /**
                 * @brief The graph description of one result.
                 * @param index Position in the node's result list; must be in range.
                 * @return A reference into the graph, valid while the graph is unchanged.
                 */
                [[nodiscard]] const Value& output_value(std::size_t index) const;

                /**
                 * @brief The tensor behind one operand.
                 * @param index Position in the node's operand list; must be in range.
                 * @return A reference stable for the pool's lifetime.
                 */
                [[nodiscard]] const SushiBLAS::Tensor& input(std::size_t index) const;

                /**
                 * @brief The tensor behind one operand, for a row that writes it.
                 * @param index Position in the node's operand list; must be in range.
                 * @return A reference stable for the pool's lifetime.
                 */
                [[nodiscard]] SushiBLAS::Tensor& mutable_input(std::size_t index) const;

                /**
                 * @brief The tensor behind one result, allocated on first request.
                 * @param index Position in the node's result list; must be in range.
                 * @return A reference stable for the pool's lifetime, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> output(std::size_t index) const;

                /**
                 * @brief Whether one result already has a tensor behind it.
                 * @param index Position in the node's result list; must be in range.
                 * @return True when the pool holds a tensor for that result.
                 */
                [[nodiscard]] bool output_is_bound(std::size_t index) const;

                /**
                 * @brief Makes one result share the storage of one operand.
                 * @param result_index Position of the result in the node's result list.
                 * @param operand_index Position of the operand in the node's operand list.
                 * @return Success, or StatusCode::OutOfMemory when the host allocation fails.
                 */
                [[nodiscard]] Status alias_output(std::size_t result_index, std::size_t operand_index) const;

                /**
                 * @brief A scratch tensor private to this node and slot, reused across replays.
                 * @param slot The scratch slot of the node.
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return A reference stable for the pool's lifetime, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> scratch(int slot, const Shape& shape,
                                                                 DType dtype) const;

                /**
                 * @brief A shared tensor filled with ones, reused across nodes.
                 * @param shape Extents of the tensor.
                 * @param dtype Element data type.
                 * @return A reference stable for the pool's lifetime, or the allocation failure.
                 */
                [[nodiscard]] Result<SushiBLAS::Tensor&> ones(const Shape& shape,
                                                              DType dtype) const;

                /**
                 * @brief Adds to the pass's count of recorded tasks.
                 * @param tasks The number of tasks the row recorded.
                 */
                void count(std::size_t tasks) const noexcept;

                /** @brief The pass's running count of recorded tasks. */
                [[nodiscard]] std::size_t emitted() const noexcept;

                /**
                 * @brief Asks the fusion selector which form to spell, and records the decision.
                 * @param query The matched node, as the selector reasons about it.
                 * @return The form the selector chose.
                 * @pre The selector answers a form that is the match, a GEMM narrowing, or None.
                 */
                [[nodiscard]] FusionChoice select(const FusionQuery& query) const;

                /**
                 * @brief Sets the task count of the decision select() recorded last.
                 * @param emitted_before The pass's task count before the row started recording.
                 * @pre select() has been called on this context.
                 */
                void close_selection(std::size_t emitted_before) const noexcept;

                /**
                 * @brief Asks the step to run @p prelude on its own thread before each replay.
                 * @param prelude Holds a target; an empty one is fatal.
                 * @return Success, StatusCode::Unsupported when the context was bound without a
                 *         list of preludes, or StatusCode::OutOfMemory when the list cannot grow.
                 */
                [[nodiscard]] Status before_each_replay(ReplayPrelude prelude) const;

            private:
                /** @brief Stops the process unless @p index is a valid operand of the node. */
                void check_input(std::size_t index) const;

                /** @brief Stops the process unless @p index is a valid result of the node. */
                void check_output(std::size_t index) const;

                SushiBLAS::TaskRecorder& recorder_;
                SushiBLAS::RNGStream& stream_;
                TensorPool& pool_;
                const Graph& graph_;
                NodeID id_;
                const FusionSelector& selector_;
                std::vector<FusionSelection>& selections_;
                std::size_t& emitted_;

                /// @brief The pass's list of replay preludes, or null when it keeps none.
                std::vector<ReplayPrelude>* preludes_;
        };
    } // namespace Graph
} // namespace SushiAI
