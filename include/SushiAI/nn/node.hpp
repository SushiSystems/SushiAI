/**************************************************************************/
/* node.hpp                                                               */
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
 * @file node.hpp
 * @brief Declares Node, the seam every node of a model graph is traced and walked through.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Config
    {
        class Value;
    } // namespace Config

    namespace NN
    {
        class StateVisitor;
        class Trace;

        /** @brief Views the values a node reads, in the order its sources were named. */
        using ValueSpan = SushiRuntime::span<const Graph::ValueId>;

        /** @brief Views the per-sample shapes a node is built over, one per input. */
        using ShapeSpan = SushiRuntime::span<const Shape>;

        /** @brief Names the port of a node that has one result. */
        inline constexpr std::string_view K_SINGLE_PORT_NAME = "output";

        /** @brief Describes one result of a node. */
        struct Port
        {
            /** @brief Holds K_SINGLE_PORT_NAME for a single result, else the role name. */
            std::string name;

            /** @brief Holds the per-sample shape, batch axis excluded. */
            Shape shape;
        };

        /** @brief Defines one node of a model graph: several values in, several values out. */
        class Node
        {
            public:
                /** @brief Releases the node through its base. */
                virtual ~Node() = default;

                /**
                 * @brief Traces the node and appends one id per entry of ports(), in order.
                 * @param inputs The values read; input_count() of them.
                 * @param outputs Grows by ports().size() on success and stays as it was on
                 *                a failure this node reports.
                 * @return InvalidDescription when @p inputs does not fit the node.
                 */
                [[nodiscard]] virtual Status forward(const Trace& trace, ValueSpan inputs,
                                                     std::vector<Graph::ValueId>& outputs) = 0;

                /**
                 * @brief Hands every declared parameter and defined buffer to @p visitor.
                 * @param prefix Dotted prefix the names of this node's state hang under.
                 */
                virtual void walk_state(std::string_view prefix, const StateVisitor& visitor) = 0;

                /**
                 * @brief Returns how many values forward() reads; at least one.
                 * @return A count fixed when the node was built, so ModelGraph::add() checks it.
                 */
                [[nodiscard]] virtual std::size_t input_count() const noexcept = 0;

                /**
                 * @brief Lists what forward() appends.
                 * @return A view that lives as long as the node.
                 */
                [[nodiscard]] virtual SushiRuntime::span<const Port> ports() const noexcept = 0;
        };

        /**
         * @brief Builds one node from a row of a model file.
         * @param inputs The per-sample shapes of the row's sources, batch axis excluded.
         * @param attributes The row's own keys; the factory refuses any it does not know.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'concat'".
         */
        using NodeFactory = Result<std::unique_ptr<Node>> (*)(ShapeSpan inputs,
                                                              const Config::Value& attributes,
                                                              std::string_view context);

        /** @brief Holds one row of the node registry. */
        struct NodeEntry
        {
            /** @brief Holds the "type" a model file spells. */
            std::string_view type;

            /** @brief Points at the factory that builds the type. */
            NodeFactory make;
        };

        namespace Detail
        {
            /** @brief Returns InvalidDescription unless @p inputs holds @p expected values. */
            [[nodiscard]] inline Status require_input_count(ValueSpan inputs, std::size_t expected)
            {
                SA_FAIL_IF(inputs.size() != expected, StatusCode::InvalidDescription,
                           "the node's input count is {}; the count it was given is {}",
                           expected, inputs.size());

                return Status();
            }

        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
