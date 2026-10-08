/**************************************************************************/
/* node_registry.cpp                                                      */
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
 * @file node_registry.cpp
 * @brief Implements make_node over both registries and build_model over a parsed model.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/node_registry.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiBLAS/core/format.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Returns the row of @p table whose type is @p type, or null when none is. */
            template<typename Entry, std::size_t N>
            [[nodiscard]] const Entry* find_type(const std::array<Entry, N>& table,
                                                 std::string_view type) noexcept
            {
                for (const Entry& entry : table)
                {
                    if (entry.type == type)
                        return &entry;
                }

                return nullptr;
            }

            /** @brief Joins the types of @p table into "'a', 'b'" for a diagnostic. */
            template<typename Entry, std::size_t N>
            [[nodiscard]] std::string list_types(const std::array<Entry, N>& table)
            {
                std::string listed;
                for (const Entry& entry : table)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(entry.type).append("'");
                }

                return listed;
            }

            /** @brief Returns the text that names row @p index of type @p type in a diagnostic. */
            [[nodiscard]] std::string row_context(std::size_t index, std::string_view type)
            {
                std::string context = "layer " + std::to_string(index) + " '";
                context.append(type).append("'");
                return context;
            }

            /**
             * @brief Builds the layer @p entry names and presents it as a one-port node.
             * @return InvalidDescription unless @p inputs holds one shape the factory accepts.
             */
            [[nodiscard]] Result<std::unique_ptr<Node>> make_layer_node(
                const LayerEntry& entry, ShapeSpan inputs, const Config::Value& attributes,
                std::string_view context)
            {
                SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads {} values, but a '{}' layer reads one", context,
                           inputs.size(), entry.type);

                SA_TRY_ASSIGN(std::unique_ptr<ILayer> layer, Detail::bridge_throwing([&]()
                {
                    return entry.make(inputs[0], attributes, context);
                }));

                return std::unique_ptr<Node>(std::make_unique<LayerNode>(std::move(layer)));
            }

            /**
             * @brief Resolves one source of a row to the port it names.
             * @param rows Port 0 of every row above the one being built, by row index.
             * @return InvalidDescription when the row is not above, or has no such port.
             */
            [[nodiscard]] Result<NodeRef> resolve_source(const ModelGraph& graph,
                                                         const std::vector<NodeRef>& rows,
                                                         const Config::SourceSpec& source,
                                                         std::string_view context)
            {
                const bool reads_input = source.node == Config::K_MODEL_INPUT;
                SA_FAIL_IF(!reads_input &&
                               (source.node < 0 ||
                                source.node >= static_cast<std::int64_t>(rows.size())),
                           StatusCode::InvalidDescription,
                           "{} reads row {}, which is not above it; a layer reads only the "
                           "model input and the rows above it",
                           context, source.node);

                const NodeRef whole =
                    reads_input ? graph.input() : rows[static_cast<std::size_t>(source.node)];
                if (source.port.empty())
                    return whole;

                Result<NodeRef> selected = graph.port(whole, source.port);
                SA_TRY(SA_ANNOTATE(selected.status(), "{}", context));

                return selected;
            }

            /**
             * @brief Builds row @p index of a model and appends it to @p graph under its index.
             * @param rows Port 0 of every row above @p index, by row index.
             * @return Port 0 of the new node, or the failure of a source, the factory or add().
             */
            [[nodiscard]] Result<NodeRef> add_row(ModelGraph& graph,
                                                  const std::vector<NodeRef>& rows,
                                                  const Config::LayerSpec& spec,
                                                  std::size_t index)
            {
                const std::string context = row_context(index, spec.type);
                SA_FAIL_IF(spec.from.empty(), StatusCode::InvalidDescription,
                           "{} lists no source, so it would read nothing", context);

                std::vector<NodeRef> sources;
                std::vector<Shape> shapes;
                sources.reserve(spec.from.size());
                shapes.reserve(spec.from.size());
                for (const Config::SourceSpec& source : spec.from)
                {
                    SA_TRY_ASSIGN(const NodeRef ref,
                                  resolve_source(graph, rows, source, context));
                    sources.push_back(ref);
                    shapes.push_back(graph.shape_of(ref));
                }

                SA_TRY_ASSIGN(std::unique_ptr<Node> node,
                              make_node(spec.type, ShapeSpan(shapes.data(), shapes.size()),
                                        spec.attributes, context));

                return graph.add(std::to_string(index), std::move(node),
                                 SushiRuntime::span<const NodeRef>(sources.data(),
                                                                   sources.size()));
            }

            /**
             * @brief Publishes entry @p index of a model's outputs: a whole row, or one port.
             * @param rows Port 0 of every row of the model, by row index.
             * @return InvalidDescription when the row, the port or the name does not fit.
             */
            [[nodiscard]] Status publish(ModelGraph& graph, const std::vector<NodeRef>& rows,
                                         const Config::OutputSpec& output, std::size_t index)
            {
                SA_FAIL_IF(output.node < 0 ||
                               output.node >= static_cast<std::int64_t>(rows.size()),
                           StatusCode::InvalidDescription,
                           "output {} names row {}, but the model has rows 0 to {}", index,
                           output.node, rows.size() - 1);

                const NodeRef row = rows[static_cast<std::size_t>(output.node)];
                if (output.port.empty())
                {
                    SA_FAIL_IF(!output.name.empty(), StatusCode::InvalidDescription,
                               "output {} states the name '{}' without a port; a row published "
                               "whole keeps its ports' own names",
                               index, output.name);

                    return SA_ANNOTATE(graph.output_ports(row), "output {}", index);
                }

                Result<NodeRef> selected = graph.port(row, output.port);
                SA_TRY(SA_ANNOTATE(selected.status(), "output {}", index));

                return SA_ANNOTATE(
                    graph.output(output.name.empty() ? output.port : output.name,
                                 selected.value()),
                    "output {}", index);
            }

        } // namespace

        Result<std::unique_ptr<Node>> make_node(std::string_view type, ShapeSpan inputs,
                                                const Config::Value& attributes,
                                                std::string_view context)
        {
            const NodeEntry* const node = find_type(K_NODE_REGISTRY, type);
            if (node != nullptr)
                return node->make(inputs, attributes, context);

            const LayerEntry* const layer = find_type(K_LAYER_REGISTRY, type);
            if (layer != nullptr)
                return make_layer_node(*layer, inputs, attributes, context);

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return SushiBLAS::Core::format(
                    "{} has the type '{}', which this build does not have; the node types "
                    "are: {}; the layer types are: {}",
                    context, type, list_types(K_NODE_REGISTRY), list_types(K_LAYER_REGISTRY));
            });
        }

        Result<ModelGraph> build_model(const Config::ModelSpec& model)
        {
            SA_FAIL_IF(model.layers.empty(), StatusCode::InvalidDescription,
                       "the model declares no layers, so it describes no model");

            ModelGraph graph(model.input_shape);

            std::vector<NodeRef> rows;
            rows.reserve(model.layers.size());
            for (std::size_t index = 0; index < model.layers.size(); ++index)
            {
                SA_TRY_ASSIGN(const NodeRef row,
                              add_row(graph, rows, model.layers[index], index));
                rows.push_back(row);
            }

            if (model.outputs.empty())
                SA_TRY(graph.output_ports(rows.back()));

            for (std::size_t index = 0; index < model.outputs.size(); ++index)
                SA_TRY(publish(graph, rows, model.outputs[index], index));

            return Result<ModelGraph>(std::move(graph));
        }

    } // namespace NN
} // namespace SushiAI
