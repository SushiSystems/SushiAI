/**************************************************************************/
/* ilayer.hpp                                                             */
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
 * @file ilayer.hpp
 * @brief A layer whose identity is decided at run time.
 *
 * @section cost Why a virtual call costs nothing here
 *
 * @c NN::Sequential decides its layer list at compile time: it holds a
 * @c std::tuple<Ls...>, its @c fields() is @c static @c constexpr, and the
 * parameter walk over it is a template walk. A JSON file decides that list at
 * run time. The two cannot be reconciled - one is a type, the other is data -
 * so a second, runtime-polymorphic path is required rather than optional.
 *
 * In an eager framework that would mean a virtual call per layer per step, and
 * it would be a real cost. Here it is not. @c forward() runs **once**, at trace
 * time, to emit IR; the graph is then compiled once and replayed
 * (docs/ARCHITECTURE.md §1). A @c std::vector<std::unique_ptr<ILayer>>
 * therefore costs exactly nothing in the steady state - the graph-native
 * architecture is what pays for this feature, and the same design would be
 * indefensible in an eager one.
 *
 * @section adapter Why there is an adapter rather than a parallel layer set
 *
 * Every layer here is one of @c nn/layers.hpp's, wrapped. Writing a second,
 * runtime-shaped @c Linear would mean two definitions of one layer's
 * mathematics, two parameter namings, and a checkpoint that moves between them
 * only by convention. @ref LayerAdapter delegates the parameter walk to
 * @c for_each_parameter - the very walk @c Sequential's children go through -
 * so the two paths cannot drift.
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <cstdint>
#include <string_view>
#include <utility>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @class ILayer
         * @brief One layer of a model whose architecture came from a file.
         */
        class ILayer
        {
            public:
                virtual ~ILayer() = default;

                /**
                 * @brief Traces the layer, declaring its parameters on first call.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The result's id.
                 */
                virtual Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                               Graph::ValueId x) = 0;

                /**
                 * @brief Applies @p visitor to every declared parameter this layer owns.
                 * @param prefix The name prefix this layer's parameters hang under.
                 * @param visitor Called once per declared parameter.
                 */
                virtual void walk_parameters(std::string_view prefix,
                                             const ParameterVisitor& visitor) = 0;

                /**
                 * @brief How many features a sample carries after this layer.
                 *
                 * What lets a config chain layer widths: a layer's input width
                 * is its predecessor's output width, so a mismatch between two
                 * adjacent layers is not expressible in the file at all.
                 *
                 * @return The output feature count.
                 */
                virtual int64_t output_features() const noexcept = 0;
        };

        /**
         * @class LayerAdapter
         * @brief Presents any compile-time @ref Module as an @ref ILayer.
         *
         * The parameter walk delegates to @c for_each_parameter, so a
         * @c LayerStack's parameter names are @c Sequential's names by
         * construction rather than by convention. That equivalence is what
         * makes a checkpoint written by one loadable by the other, and it is
         * proved by a round-trip test rather than asserted in a comment.
         *
         * @tparam L The wrapped layer type; must satisfy @ref Module.
         */
        template<typename L>
        class LayerAdapter final : public ILayer
        {
            public:
                static_assert(Module<L>,
                              "LayerAdapter's argument must satisfy the Module contract: "
                              "a static fields() listing its parameters, and a "
                              "forward(const Graph::GraphBuilder&, Graph::ValueId) returning "
                              "exactly Graph::ValueId.");

                /**
                 * @brief Wraps @p layer.
                 *
                 * The output width is supplied rather than derived: a layer
                 * knows its own mathematics but not, in general, how to name
                 * its output width without being traced first, and the config
                 * layer needs the number before anything is traced.
                 *
                 * @param layer The layer; moved in.
                 * @param output_features How many features a sample carries after it.
                 */
                LayerAdapter(L layer, int64_t output_features)
                    : layer_(std::move(layer)), output_features_(output_features) {}

                Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                       Graph::ValueId x) override
                {
                    return layer_.forward(builder, x);
                }

                void walk_parameters(std::string_view prefix,
                                     const ParameterVisitor& visitor) override
                {
                    // Takes for_each_parameter's leaf or container arm, never
                    // its runtime arm, because L is a compile-time layer - so
                    // this does not recurse back into this function.
                    for_each_parameter(layer_, prefix, visitor);
                }

                int64_t output_features() const noexcept override { return output_features_; }

                /** @brief The wrapped layer, for a test that wants to inspect it. */
                L& layer() noexcept { return layer_; }

                /** @brief The wrapped layer, read-only. */
                const L& layer() const noexcept { return layer_; }

            private:
                L layer_;
                int64_t output_features_ = 0;
        };

    } // namespace NN
} // namespace SushiAI
