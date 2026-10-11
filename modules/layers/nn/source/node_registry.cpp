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
 * @brief Implements NodeRegistry, make_node over its rows and build_model over a parsed model.
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
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Returns the row of @p rows whose type is @p type, or null when none is. */
            template<typename Entry>
            [[nodiscard]] const Entry* find_type(const std::vector<Entry>& rows,
                                                 std::string_view type) noexcept
            {
                for (const Entry& entry : rows)
                {
                    if (entry.type == type)
                        return &entry;
                }

                return nullptr;
            }

            /** @brief Joins the types of @p rows into "'a', 'b'" for a diagnostic. */
            template<typename Entry>
            [[nodiscard]] std::string list_types(SushiRuntime::span<const Entry> rows)
            {
                std::string listed;
                for (const Entry& entry : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(entry.type).append("'");
                }

                return listed;
            }

            /**
             * @brief Checks that @p row may join @p registry.
             * @param row An empty type or a null factory is fatal.
             * @return InvalidDescription when @p registry holds the type of @p row.
             */
            template<typename Entry>
            [[nodiscard]] Status require_new_type(const NodeRegistry& registry, const Entry& row)
            {
                SA_FATAL_IF(row.type.empty(), "a registry row has an empty type");
                SA_FATAL_IF(row.make == nullptr, "the registry row '{}' has no factory",
                            row.type);
                SA_FAIL_IF(registry.find_node(row.type) != nullptr ||
                               registry.find_linked(row.type) != nullptr ||
                               registry.find_layer(row.type) != nullptr,
                           StatusCode::InvalidDescription,
                           "the type '{}' is already in the registry; a type has one row",
                           row.type);

                return Status();
            }

            /** @brief Joins the types of the node rows and the linked node rows of @p registry. */
            [[nodiscard]] std::string list_node_types(const NodeRegistry& registry)
            {
                std::string listed = list_types(registry.nodes());
                if (!listed.empty() && !registry.linked().empty())
                    listed.append(", ");

                return listed.append(list_types(registry.linked()));
            }

            /** @brief Appends @p row to @p rows, or returns the failure of the allocation. */
            template<typename Entry>
            [[nodiscard]] Status append_row(std::vector<Entry>& rows, const Entry& row) noexcept
            {
                return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    rows.push_back(row);
                    return Status();
                });
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

                SA_TRY_ASSIGN(std::unique_ptr<Layer> layer, Detail::bridge_throwing([&]()
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
            [[nodiscard]] Result<NodeRef> add_row(const NodeRegistry& registry,
                                                  ModelGraph& graph,
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

                // Node i of the graph is row i, so the graph is the rows above this one.
                SA_TRY_ASSIGN(std::unique_ptr<Node> node,
                              make_node(registry, spec.type,
                                        ShapeSpan(shapes.data(), shapes.size()),
                                        spec.attributes, context, BuiltRows(graph)));

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

        Status NodeRegistry::add(const NodeEntry& row)
        {
            SA_TRY(require_new_type(*this, row));

            return append_row(nodes_, row);
        }

        Status NodeRegistry::add(const LayerEntry& row)
        {
            SA_TRY(require_new_type(*this, row));

            return append_row(layers_, row);
        }

        Status NodeRegistry::add(const LinkedNodeEntry& row)
        {
            SA_TRY(require_new_type(*this, row));

            return append_row(linked_, row);
        }

        const NodeEntry* NodeRegistry::find_node(std::string_view type) const noexcept
        {
            return find_type(nodes_, type);
        }

        const LinkedNodeEntry* NodeRegistry::find_linked(std::string_view type) const noexcept
        {
            return find_type(linked_, type);
        }

        const LayerEntry* NodeRegistry::find_layer(std::string_view type) const noexcept
        {
            return find_type(layers_, type);
        }

        SushiRuntime::span<const NodeEntry> NodeRegistry::nodes() const noexcept
        {
            return SushiRuntime::span<const NodeEntry>(nodes_.data(), nodes_.size());
        }

        SushiRuntime::span<const LinkedNodeEntry> NodeRegistry::linked() const noexcept
        {
            return SushiRuntime::span<const LinkedNodeEntry>(linked_.data(), linked_.size());
        }

        SushiRuntime::span<const LayerEntry> NodeRegistry::layers() const noexcept
        {
            return SushiRuntime::span<const LayerEntry>(layers_.data(), layers_.size());
        }

        Result<std::unique_ptr<Node>> make_node(const NodeRegistry& registry,
                                                std::string_view type, ShapeSpan inputs,
                                                const Config::Value& attributes,
                                                std::string_view context, const BuiltRows& above)
        {
            const NodeEntry* const node = registry.find_node(type);
            if (node != nullptr)
                return node->make(inputs, attributes, context);

            const LinkedNodeEntry* const linked = registry.find_linked(type);
            if (linked != nullptr)
                return linked->make(inputs, attributes, context, above);

            const LayerEntry* const layer = registry.find_layer(type);
            if (layer != nullptr)
                return make_layer_node(*layer, inputs, attributes, context);

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return SushiBLAS::Core::format(
                    "{} has the type '{}', which this build does not have; the node types "
                    "are: {}; the layer types are: {}",
                    context, type, list_node_types(registry),
                    list_types(registry.layers()));
            });
        }

        Result<ModelGraph> build_model(const NodeRegistry& registry,
                                       const Config::ModelSpec& model)
        {
            SA_FAIL_IF(model.layers.empty(), StatusCode::InvalidDescription,
                       "the model declares no layers, so it describes no model");

            ModelGraph graph(model.input_shape);

            std::vector<NodeRef> rows;
            rows.reserve(model.layers.size());
            for (std::size_t index = 0; index < model.layers.size(); ++index)
            {
                SA_TRY_ASSIGN(const NodeRef row,
                              add_row(registry, graph, rows, model.layers[index], index));
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
