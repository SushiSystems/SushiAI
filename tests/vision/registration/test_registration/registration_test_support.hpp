/**************************************************************************/
/* registration_test_support.hpp                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file registration_test_support.hpp
 * @brief Declares the helpers the tests of the registration module share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/registration/registration.hpp>

namespace SushiAI::Test
{
    /** @brief Builds a registry from the layer rows, then the vision node rows. */
    inline Result<NN::NodeRegistry> composed()
    {
        NN::NodeRegistry registry;
        SA_TRY(NN::register_layers(registry));
        SA_TRY(Vision::register_nodes(registry));

        return Result<NN::NodeRegistry>(std::move(registry));
    }

    /** @brief Traces @p model over a batch of one and lists its outputs with their extents. */
    inline std::vector<Train::OutputShape> trace_outputs(NN::ModelGraph& model, const Shape& sample)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID input =
            builder.input(Shape{1, sample[0], sample[1], sample[2]}, "x");
        NN::BufferUpdateList updates;

        const NN::ModelOutputs outputs =
            model.trace(NN::Trace::training(builder, updates), input).value();

        std::vector<Train::OutputShape> shapes;
        for (const NN::ModelOutputs::Entry& output : outputs.all())
            shapes.push_back(Train::OutputShape{output.first, graph.value(output.second).shape});
        return shapes;
    }

    /** @brief Returns the number of elements the parameters of each row of @p model hold. */
    inline std::vector<std::int64_t> count_row_parameters(NN::ModelGraph& model)
    {
        std::vector<std::int64_t> rows;
        for (std::size_t row = 0; row < model.size(); ++row)
        {
            std::int64_t total = 0;
            auto on_parameter = [&total](const std::string&, NN::Parameter& parameter)
            {
                total += parameter.shape().num_elements();
            };
            auto on_buffer = [](const std::string&, NN::Buffer&)
            {

            };
            const NN::StateVisitor visitor(on_parameter, on_buffer);
            model.node(row).walk_state("", visitor);
            rows.push_back(total);
        }
        return rows;
    }

    /** @brief Returns the number of elements the parameters of @p model hold. */
    inline std::int64_t count_parameters(NN::ModelGraph& model)
    {
        std::int64_t total = 0;
        auto on_parameter = [&total](const std::string&, NN::Parameter& parameter)
        {
            total += parameter.shape().num_elements();
        };
        auto on_buffer = [](const std::string&, NN::Buffer&)
        {

        };
        const NN::StateVisitor visitor(on_parameter, on_buffer);
        model.walk_state("", visitor);
        return total;
    }
} // namespace SushiAI::Test
