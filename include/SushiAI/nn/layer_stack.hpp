/**************************************************************************/
/* layer_stack.hpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file layer_stack.hpp
 * @brief Sequential's runtime twin: a layer list that came from data.
 *
 * @section parity Naming parity with Sequential is the whole point
 *
 * @c Sequential names its children by index **including the gaps a
 * parameterless activation leaves**, so a three-layer MLP is "0.weight",
 * "0.bias", "2.weight". @c LayerStack applies the same rule, and it does so by
 * going through the same @c for_each_parameter walk rather than by
 * reimplementing the join.
 *
 * That is not cosmetic. It means a checkpoint written by a hand-written
 * @c Sequential loads into a @c LayerStack built from the equivalent
 * model.json, and the reverse - so the equivalence of the compile-time and
 * runtime paths is provable *by the checkpoint itself* rather than by a second
 * mechanism whose own correctness would then be the open question.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @class LayerStack
         * @brief Applies a runtime list of layers in order, naming them by index.
         *
         * Satisfies @ref Module, so everything written as
         * @c template<typename @c M> over the parameter walk - @c NN::initialize,
         * @c Train::build_graph, @c NN::parameter_elements, the checkpoint I/O -
         * consumes it with no change at all. That is what the walk's third arm
         * bought (see module.hpp's @c runtime section).
         *
         * Non-copyable, because it owns its layers through @c unique_ptr;
         * movable, because a factory has to return one.
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
                void push(std::unique_ptr<ILayer> layer);

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
                 * @brief Applies @p visitor to every declared parameter, index-prefixed.
                 *
                 * The index is the layer's position in the stack, gaps and all,
                 * which is exactly what @c Sequential's @c visit_children
                 * produces.
                 *
                 * @param prefix The name prefix; empty at the root.
                 * @param visitor Called once per declared parameter.
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
                 * @brief The layer at @p index, for a test that wants to inspect it.
                 * @param index Which layer, in [0, size()).
                 * @return The layer; owned by the stack.
                 * @throws Error If @p index is out of range.
                 */
                ILayer& layer(std::size_t index) const;

            private:
                std::vector<std::unique_ptr<ILayer>> layers_;
        };

        // Checked here, where a mistake names LayerStack, rather than at the
        // first template that tried to walk one.
        static_assert(Module<LayerStack>,
                      "LayerStack must satisfy the Module contract so every "
                      "template written over the parameter walk accepts it");
        static_assert(Detail::HasParameterWalk<LayerStack>::value,
                      "LayerStack must be enumerated through its own walk");
        static_assert(!ModuleContainer<LayerStack>,
                      "LayerStack must not also look like a container: its children are "
                      "all ILayer&, so a container walk over it would find no parameters "
                      "and report no error");

    } // namespace NN
} // namespace SushiAI
