/**************************************************************************/
/* layer_stack.hpp                                                        */
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
 * @file layer_stack.hpp
 * @brief Declares runtime sequential layer container executing Layer instances.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @class LayerStack
         * @brief Holds a runtime sequential list of polymorphic layers.
         */
        class LayerStack
        {
            public:
                LayerStack() = default;
                LayerStack(LayerStack&&) noexcept = default;
                LayerStack& operator=(LayerStack&&) noexcept = default;
                LayerStack(const LayerStack&) = delete;
                LayerStack& operator=(const LayerStack&) = delete;

                /**
                 * @brief Appends @p layer to the end of the stack.
                 * @param layer The layer; ownership is taken.
                 * @throws Error If @p layer is null - a null entry would trace
                 *         as a silent identity rather than failing.
                 */
                void push(std::unique_ptr<Layer> layer);

                /** @brief No parameters of its own; the layers own them all. */
                static constexpr auto fields() { return std::tuple<>{}; }

                /**
                 * @brief Traces every layer in order.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The last layer's result.
                 * @throws Error If the stack is empty, which would otherwise
                 *         make the model the identity function.
                 */
                Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x);

                /**
                 * @brief Applies visitor to every declared parameter with index prefix.
                 * @param prefix Parameter name prefix for hierarchy traversal.
                 * @param visitor Callback invoked once per declared parameter.
                 */
                void walk_parameters(std::string_view prefix, const ParameterVisitor& visitor);

                /** @brief How many layers the stack holds. */
                std::size_t size() const noexcept { return layers_.size(); }

                /** @brief Whether the stack holds no layers at all. */
                bool empty() const noexcept { return layers_.empty(); }

                /**
                 * @brief How many features a sample carries after the last layer.
                 * @return The last layer's output width.
                 * @throws Error If the stack is empty.
                 */
                int64_t output_features() const;

                /**
                 * @brief Returns the last layer's per-sample output shape, batch axis excluded.
                 * @pre The stack holds at least one layer.
                 */
                Shape output_shape() const;

                /**
                 * @brief Returns output_shape(), in the form a layer type gives it.
                 * @param input The shape the stack was built over; the layers check it when
                 *              they are traced, and the stack does not hold it.
                 * @pre The stack holds at least one layer.
                 */
                Shape output_shape(const Shape& input) const;

                /**
                 * @brief The layer at @p index, for a test that wants to inspect it.
                 * @param index Which layer, in [0, size()).
                 * @return The layer; owned by the stack.
                 * @throws Error If @p index is out of range.
                 */
                Layer& layer(std::size_t index) const;

            private:
                std::vector<std::unique_ptr<Layer>> layers_;
        };

        // Invariant: Verifies LayerStack module and parameter walk contracts at declaration scope.
        static_assert(Module<LayerStack>,
                      "LayerStack must satisfy the Module contract so every "
                      "template written over the parameter walk accepts it");
        static_assert(Detail::HasOutputShapeOfInput<LayerStack>::value,
                      "LayerStack must map an input shape to its output shape to be a node");
        static_assert(Detail::HasParameterWalk<LayerStack>::value,
                      "LayerStack must be enumerated through its own walk");
        static_assert(!ModuleContainer<LayerStack>,
                      "LayerStack must not also look like a container: its children are "
                      "all Layer&, so a container walk over it would find no parameters "
                      "and report no error");

    } // namespace NN
} // namespace SushiAI
