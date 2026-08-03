/**************************************************************************/
/* layers.hpp                                                             */
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
 * @file layers.hpp
 * @brief The layers themselves: aggregates that trace into IR.
 *
 * Every layer here is an aggregate, so it is written the way it reads -
 * @c Linear{784, 128} - and it declares its own parameters the first time it
 * traces. Nothing registers, nothing inherits, and each layer's whole contract
 * is the two members below its data: @c fields() and @c forward().
 *
 * Host lane: no SushiBLAS, no SYCL.
 */

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @struct Linear
         * @brief An affine map: Y = X * W^T + b.
         *
         * The weight is stored [out_features, in_features], matching PyTorch,
         * so a checkpoint moves between the two without a transpose. The
         * product is expressed with @c MATMUL's transpose @e flag rather than a
         * transposed view - SushiBLAS derives a leading dimension from a
         * tensor's shape, so a view would be read with the wrong stride and
         * produce a plausible-looking wrong answer.
         */
        struct Linear
        {
            /** @brief Input features per sample. */
            int64_t in_features = 0;

            /** @brief Output features per sample. */
            int64_t out_features = 0;

            /** @brief Whether to add a learnable bias. */
            bool use_bias = true;

            /** @brief The [out_features, in_features] weight. */
            Parameter weight{};

            /** @brief The [out_features] bias; undeclared when use_bias is false. */
            Parameter bias{};

            /** @brief This layer's parameters, for the generic walk. */
            static constexpr auto fields()
            {
                return std::tuple{
                    field("weight", &Linear::weight),
                    field("bias", &Linear::bias),
                };
            }

            /**
             * @brief Traces the layer, declaring its parameters on first call.
             * @param builder Where the nodes go.
             * @param x An [N, in_features] input.
             * @return The [N, out_features] result's id.
             * @throws Error If @p x's inner extent is not in_features.
             */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x);
        };

        /**
         * @struct ReLU
         * @brief max(0, x), elementwise.
         */
        struct ReLU
        {
            /** @brief No parameters. */
            static constexpr auto fields() { return std::tuple<>{}; }

            /** @brief Traces the activation. @param builder Where the node goes. @param x The input. @return The result's id. */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const
            {
                return builder.relu(x);
            }
        };

        /**
         * @struct GELU
         * @brief The Gaussian error linear unit, tanh approximation.
         */
        struct GELU
        {
            /** @brief No parameters. */
            static constexpr auto fields() { return std::tuple<>{}; }

            /** @brief Traces the activation. @param builder Where the node goes. @param x The input. @return The result's id. */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const
            {
                return builder.gelu(x);
            }
        };

        /**
         * @struct Tanh
         * @brief The hyperbolic tangent, elementwise.
         */
        struct Tanh
        {
            /** @brief No parameters. */
            static constexpr auto fields() { return std::tuple<>{}; }

            /** @brief Traces the activation. @param builder Where the node goes. @param x The input. @return The result's id. */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const
            {
                return builder.tanh(x);
            }
        };

        /**
         * @struct Sigmoid
         * @brief The logistic sigmoid, elementwise.
         */
        struct Sigmoid
        {
            /** @brief No parameters. */
            static constexpr auto fields() { return std::tuple<>{}; }

            /** @brief Traces the activation. @param builder Where the node goes. @param x The input. @return The result's id. */
            Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const
            {
                return builder.sigmoid(x);
            }
        };

        /**
         * @class Sequential
         * @brief Applies its layers in order, and names their parameters by index.
         *
         * Deduces its own type, so the model reads as a list:
         * @code
         * Sequential model{Linear{784, 128}, ReLU{}, Linear{128, 10}};
         * @endcode
         *
         * Its children are visited by index, which is what produces the
         * "0.weight" / "2.bias" naming PyTorch uses - including the gaps where
         * a parameterless activation sits, so the indices keep matching the
         * layer positions rather than the parameter positions.
         *
         * @tparam Ls The layer types, in application order.
         */
        template<typename... Ls>
        class Sequential
        {
            public:
                // The pack is checked here rather than spelled `template<Module... Ls>`,
                // which is a C++20 constraint and this library is C++17. The contract is
                // the same and it is still stated at the top of the class, so a malformed
                // layer is reported against Sequential's own instantiation - naming the
                // pack - rather than 200 lines down inside std::apply. What is lost is
                // that Sequential<Bad> is now a *valid* specialization that fails when
                // instantiated, instead of one that never matched: it cannot be SFINAE'd
                // away by an overload set. Nothing in this library overloads on it.
                static_assert((Module<Ls> && ...),
                              "Every type in a Sequential must satisfy the Module contract: "
                              "a static fields() listing its parameters, and a "
                              "forward(const Graph::GraphBuilder&, Graph::ValueId) returning "
                              "exactly Graph::ValueId. Check the layer named in the "
                              "instantiation below, not this line.");

                /**
                 * @brief Constructs from the layers themselves.
                 * @param layers The layers, in application order; moved in.
                 */
                constexpr explicit Sequential(Ls... layers) : layers_(std::move(layers)...) {}

                /** @brief No parameters of its own; the children own them all. */
                static constexpr auto fields() { return std::tuple<>{}; }

                /**
                 * @brief Traces every layer in order.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The last layer's result.
                 */
                Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
                {
                    std::apply([&](auto&... layer) { ((x = layer.forward(builder, x)), ...); },
                               layers_);
                    return x;
                }

                /**
                 * @brief Applies @p visitor to each child with its index.
                 * @param visitor Called as visitor(std::size_t index, Layer& child).
                 */
                template<typename Fn>
                void visit_children(Fn&& visitor)
                {
                    visit_children_impl(visitor, std::index_sequence_for<Ls...>{});
                }

                /** @brief How many layers this container holds. */
                static constexpr std::size_t size() noexcept { return sizeof...(Ls); }

                /** @brief Direct access to the @p Index-th layer. */
                template<std::size_t Index>
                auto& layer() noexcept { return std::get<Index>(layers_); }

            private:
                template<typename Fn, std::size_t... Indices>
                void visit_children_impl(Fn& visitor, std::index_sequence<Indices...>)
                {
                    (visitor(Indices, std::get<Indices>(layers_)), ...);
                }

                std::tuple<Ls...> layers_;
        };

        /** @brief Deduces Sequential's layer pack from its constructor arguments. */
        template<typename... Ls>
        Sequential(Ls...) -> Sequential<Ls...>;

        // The contract is checked here, where a mistake names the layer, rather
        // than at the first use, where it would name a template instantiation.
        static_assert(Module<Linear>, "Linear must satisfy the Module contract");
        static_assert(Module<ReLU>, "ReLU must satisfy the Module contract");
        static_assert(Module<GELU>, "GELU must satisfy the Module contract");
        static_assert(Module<Tanh>, "Tanh must satisfy the Module contract");
        static_assert(Module<Sigmoid>, "Sigmoid must satisfy the Module contract");
        static_assert(Module<Sequential<Linear, ReLU>>,
                      "Sequential must itself satisfy the Module contract so it can nest");
        static_assert(ModuleContainer<Sequential<Linear, ReLU>>,
                      "Sequential must be walkable as a container");
        static_assert(!ModuleContainer<Linear>,
                      "A leaf layer must not look like a container");

    } // namespace NN
} // namespace SushiAI
