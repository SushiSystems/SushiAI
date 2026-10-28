/**************************************************************************/
/* trace_summary.hpp                                                      */
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
 * @file trace_summary.hpp
 * @brief Declares TraceSummary, what one training trace of a model graph left behind.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Lists the operation of every node of a graph, in order. */
        using OpList = std::vector<decltype(Graph::Node::op)>;

        /** @brief Holds the batch size trace_summary traces over. */
        inline constexpr int64_t K_TRACE_BATCH = 2;

        /** @brief Holds what one training trace of a model graph left behind. */
        struct TraceSummary
        {
            OpList ops;                       /**< The operation of every traced node. */
            std::vector<std::string> outputs; /**< Each published output as "name [extents]". */
            std::vector<std::string> state;   /**< Each piece of state, as state_record() wrote it. */
        };

        /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
        [[nodiscard]] inline std::string state_record(const std::string& name, const Shape& shape,
                                                      bool is_buffer) noexcept
        {
            return name + " " + shape.to_string() + (is_buffer ? " buffer" : " parameter");
        }

        /** @brief Lists every piece of state of @p model under @p prefix, in walk order. */
        [[nodiscard]] inline std::vector<std::string> state_records(NN::ModelGraph& model,
                                                                    std::string_view prefix) noexcept
        {
            std::vector<std::string> records;
            const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
            {
                records.push_back(state_record(name, state.shape(), false));
            };
            const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
            {
                records.push_back(state_record(name, state.shape(), true));
            };
            const NN::StateVisitor visitor(on_parameter, on_buffer);
            model.walk_state(prefix, visitor);

            return records;
        }

        /**
         * @brief Traces @p model in Training mode over K_TRACE_BATCH samples and summarises it.
         * @param prefix The prefix the state names are listed under.
         */
        [[nodiscard]] inline Result<TraceSummary> trace_summary(NN::ModelGraph& model,
                                                                const Shape& sample,
                                                                std::string_view prefix) noexcept
        {
            std::vector<int64_t> extents{K_TRACE_BATCH};
            for (int32_t axis = 0; axis < sample.rank(); ++axis)
                extents.push_back(sample[axis]);

            Graph::Graph graph;
            const Graph::GraphBuilder builder(graph);
            const Graph::ValueID x = builder.input(
                Shape(SushiRuntime::span<const int64_t>(extents.data(), extents.size())), "x");
            NN::BufferUpdateList updates;

            SA_TRY_ASSIGN(const NN::ModelOutputs outputs,
                          model.trace(NN::Trace::training(builder, updates), x));

            TraceSummary summary;
            for (const Graph::Node& node : graph.nodes())
                summary.ops.push_back(node.op);

            for (const NN::ModelOutputs::Entry& output : outputs.all())
                summary.outputs.push_back(output.first + " " +
                                          graph.value(output.second).shape.to_string());

            summary.state = state_records(model, prefix);
            return summary;
        }

    } // namespace Test
} // namespace SushiAI
