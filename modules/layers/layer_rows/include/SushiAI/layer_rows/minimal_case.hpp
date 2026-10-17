/**************************************************************************/
/* minimal_case.hpp                                                       */
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
 * @file minimal_case.hpp
 * @brief Declares MinimalCase, the smallest object of a node registry row, and its builder.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/model_text/model_text.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Describes one node registry row's minimal object and the shapes around it. */
        struct MinimalCase
        {
            std::string_view type;       /**< The registry key. */
            std::string_view attributes; /**< The row without its "type" and "from". */
            std::vector<Shape> inputs;   /**< The per-sample shapes the row reads. */
            Shape output;                /**< The per-sample shape of the node's one port. */
        };

        /** @brief Views @p shapes as the inputs of one row. */
        [[nodiscard]] inline NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
        {
            return NN::ShapeSpan(shapes.data(), shapes.size());
        }

        /** @brief Builds the node @p minimal describes from @p registry, as "layer 0 '<type>'". */
        [[nodiscard]] inline Result<std::unique_ptr<NN::Node>> make_minimal(
            const NN::NodeRegistry& registry, const MinimalCase& minimal,
            std::string_view attributes)
        {
            const std::string context = "layer 0 '" + std::string(minimal.type) + "'";
            return NN::make_node(registry, minimal.type, span_of(minimal.inputs),
                                 attributes_from(attributes), context);
        }

    } // namespace Test
} // namespace SushiAI
