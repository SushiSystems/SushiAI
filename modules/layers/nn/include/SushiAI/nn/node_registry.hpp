/**************************************************************************/
/* node_registry.hpp                                                      */
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
 * @file node_registry.hpp
 * @brief Declares NodeRegistry, make_node and the builder of a model graph from a file.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Holds the node, linked node and layer rows a model file may name, by type. */
        class NodeRegistry final
        {
            public:
                /**
                 * @brief Appends a node row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            factory is fatal.
                 * @return InvalidDescription when a node row or a layer row holds the type.
                 */
                [[nodiscard]] Status add(const NodeEntry& row) noexcept;

                /**
                 * @brief Appends a layer row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            factory is fatal.
                 * @return InvalidDescription when a node row or a layer row holds the type.
                 */
                [[nodiscard]] Status add(const LayerEntry& row) noexcept;

                /**
                 * @brief Appends a linked node row, whose factory reads the rows above its own.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            factory is fatal.
                 * @return InvalidDescription when a row of any kind holds the type.
                 */
                [[nodiscard]] Status add(const LinkedNodeEntry& row) noexcept;

                /** @brief Returns the node row of @p type, or null when none holds it. */
                [[nodiscard]] const NodeEntry* find_node(std::string_view type) const noexcept;

                /** @brief Returns the linked node row of @p type, or null when none holds it. */
                [[nodiscard]] const LinkedNodeEntry* find_linked(
                    std::string_view type) const noexcept;

                /** @brief Returns the layer row of @p type, or null when none holds it. */
                [[nodiscard]] const LayerEntry* find_layer(std::string_view type) const noexcept;

                /**
                 * @brief Returns the node rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const NodeEntry> nodes() const noexcept;

                /**
                 * @brief Returns the linked node rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const LinkedNodeEntry> linked() const noexcept;

                /**
                 * @brief Returns the layer rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const LayerEntry> layers() const noexcept;

            private:
                /** @brief Holds the node rows in insertion order. */
                std::vector<NodeEntry> nodes_;

                /** @brief Holds the linked node rows in insertion order. */
                std::vector<LinkedNodeEntry> linked_;

                /** @brief Holds the layer rows in insertion order. */
                std::vector<LayerEntry> layers_;
        };

        /**
         * @brief Builds a node of @p type: a node row of @p registry, else a linked node row,
         *        else a layer row wrapped in a LayerNode.
         * @param inputs The shapes of the row's sources; a layer row reads one.
         * @param above The rows a linked node row may name; the default views none.
         * @return The node, or InvalidDescription when the factory refuses the row or
         *         @p registry holds no @p type; the second message lists every row.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_node(const NodeRegistry& registry,
                                                              std::string_view type,
                                                              ShapeSpan inputs,
                                                              const Config::Value& attributes,
                                                              std::string_view context,
                                                              const BuiltRows& above = {}) noexcept;

        /**
         * @brief Builds the graph @p model describes, naming each row by its index.
         * @param registry The rows a type of @p model may name.
         * @param model A parsed model, or a spec built by hand, which is checked the same
         *              way; an empty outputs list publishes every port of the last row.
         * @return The graph with its outputs published, or InvalidDescription naming the
         *         first row or output that does not fit.
         */
        [[nodiscard]] Result<ModelGraph> build_model(const NodeRegistry& registry,
                                                     const Config::ModelSpec& model) noexcept;

    } // namespace NN
} // namespace SushiAI
