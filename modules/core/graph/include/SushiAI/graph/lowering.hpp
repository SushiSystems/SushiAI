/**************************************************************************/
/* lowering.hpp                                                           */
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
 * @file lowering.hpp
 * @brief Declares the driver that records an IR graph through the lowering table.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Collects the scratch requests of every node of @p graph, in node order,
         *        through the scratch rules of the table in graph/lowering/all.hpp.
         * @return The requests; MemoryPlan::build() places them.
         */
        [[nodiscard]] std::vector<ScratchRequest> scratch_of(const Graph& graph);

        /**
         * @class Lowering
         * @brief Records the SushiBLAS work of every node of an IR graph, in node order, each
         *        node through its row of the table in graph/lowering/all.hpp.
         */
        class Lowering final
        {
            public:
                /**
                 * @brief Binds the lowering to the recorder, stream, pool and selector of one step.
                 * @pre Every argument outlives the lowering.
                 */
                Lowering(SushiBLAS::TaskRecorder& recorder, SushiBLAS::RNGStream& stream,
                         TensorPool& pool, const FusionSelector& selector) noexcept
                    : recorder_(&recorder), stream_(&stream), pool_(&pool), selector_(&selector)
                {

                }

                /**
                 * @brief Records the nodes of @p graph that @p nodes names, in node order.
                 * @return Success, or the first refusal, annotated with the node's operation;
                 *         InvalidDescription for a range that ends before it starts or
                 *         behind the last node.
                 */
                [[nodiscard]] Status record(const Graph& graph, NodeRange nodes = {});

                /** @brief Returns the number of tasks recorded by the last record(). */
                [[nodiscard]] std::size_t emitted_operations() const noexcept
                {
                    return emitted_;
                }

                /** @brief Returns the fusion decisions of the last record(), in execution order. */
                [[nodiscard]] SushiRuntime::span<const FusionSelection> selections() const noexcept
                {
                    return SushiRuntime::span<const FusionSelection>(selections_.data(),
                                                                     selections_.size());
                }

                /**
                 * @brief Hands over the replay preludes of the last record(), in node order.
                 * @return The list, which whoever replays the recorded work calls before each
                 *         replay; the lowering keeps none of it.
                 */
                [[nodiscard]] std::vector<ReplayPrelude> take_preludes() noexcept
                {
                    return std::move(preludes_);
                }

            private:
                /** @brief The recorder every row records on. */
                SushiBLAS::TaskRecorder* recorder_;

                /** @brief The random stream handed to every row. */
                SushiBLAS::RNGStream* stream_;

                /** @brief The pool that holds the tensor of each value. */
                TensorPool* pool_;

                /** @brief The selector that picks the form of a fusible node. */
                const FusionSelector* selector_;

                /** @brief The fusion decisions of the last record(), in execution order. */
                std::vector<FusionSelection> selections_;

                /** @brief The number of tasks the last record() recorded. */
                std::size_t emitted_ = 0;

                /** @brief The replay preludes the rows of the last record() registered. */
                std::vector<ReplayPrelude> preludes_;
        };

    } // namespace Graph
} // namespace SushiAI
