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
 * @brief Declares the node registry, make_node and the builder of a model graph from a file.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <memory>
#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/blocks/bottleneck.hpp>
#include <SushiAI/nn/blocks/c3k.hpp>
#include <SushiAI/nn/blocks/c3k2.hpp>
#include <SushiAI/nn/blocks/conv_norm.hpp>
#include <SushiAI/nn/blocks/sppf.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Lists the node types LAYER_REGISTRY cannot hold: several ports, or a mode. */
        inline constexpr std::array<NodeEntry, 7> K_NODE_REGISTRY = {
            NodeEntry{"batchnorm2d", &make_batch_norm2d_node},
            NodeEntry{"concat",      &make_concat_node},
            NodeEntry{"conv",        &make_conv_node},
            NodeEntry{"bottleneck",  &make_bottleneck_node},
            NodeEntry{"c3k",         &make_c3k_node},
            NodeEntry{"c3k2",        &make_c3k2_node},
            NodeEntry{"sppf",        &make_sppf_node},
        };

        /**
         * @brief Builds a node of @p type: a K_NODE_REGISTRY row, else a LAYER_REGISTRY row
         *        wrapped in a LayerNode.
         * @param inputs The shapes of the row's sources; a LAYER_REGISTRY row reads one.
         * @param context What to call the row in a diagnostic, e.g. "layer 3 'concat'".
         * @return The node, or InvalidDescription when the factory refuses the row or
         *         @p type is in neither table; the second message lists both tables.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_node(std::string_view type,
                                                              ShapeSpan inputs,
                                                              const Config::Value& attributes,
                                                              std::string_view context);

        /**
         * @brief Builds the graph @p model describes, naming each row by its index.
         * @param model A parsed model, or a spec built by hand, which is checked the same
         *              way; an empty outputs list publishes every port of the last row.
         * @return The graph with its outputs published, or InvalidDescription naming the
         *         first row or output that does not fit.
         */
        [[nodiscard]] Result<ModelGraph> build_model(const Config::ModelSpec& model);

    } // namespace NN
} // namespace SushiAI
