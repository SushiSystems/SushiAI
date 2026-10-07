/**************************************************************************/
/* layer_node.hpp                                                         */
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
 * @file layer_node.hpp
 * @brief Declares LayerNode, which presents a run-time ILayer as a one-in one-out node.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Presents a run-time ILayer as a one-in one-out node. */
        class LayerNode final : public Node
        {
            public:
                /**
                 * @brief Takes ownership of @p layer and declares the one port it produces.
                 * @param layer The layer to present; a null pointer is fatal.
                 */
                explicit LayerNode(std::unique_ptr<ILayer> layer);

                /**
                 * @brief Traces the layer through the trace's builder and appends its result.
                 * @return InvalidDescription unless @p inputs holds exactly one value the layer
                 *         accepts; @p outputs is unchanged then.
                 */
                [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs,
                                             std::vector<Graph::ValueId>& outputs) override;

                /** @brief Returns one, the number of values a layer reads. */
                [[nodiscard]] std::size_t input_count() const noexcept override;

                /** @brief Hands the layer's declared parameters to @p visitor; it has no buffer. */
                void walk_state(std::string_view prefix, const StateVisitor& visitor) override;

                /** @brief Lists the single port, named K_SINGLE_PORT_NAME. */
                [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;

                /** @brief Returns the wrapped layer. */
                [[nodiscard]] ILayer& layer() noexcept;

                /** @brief Returns the wrapped layer, read-only. */
                [[nodiscard]] const ILayer& layer() const noexcept;

            private:
                /** @brief Holds the wrapped layer; never null. */
                std::unique_ptr<ILayer> layer_;

                /** @brief Holds the one port forward() appends to. */
                Port port_;
        };

    } // namespace NN
} // namespace SushiAI
