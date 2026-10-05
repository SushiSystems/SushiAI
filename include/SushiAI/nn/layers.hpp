/****************************************************************/
/* layers.hpp                                                   */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file layers.hpp
 * @brief Declares core neural network layers that trace into graph IR.
 * @author Mustafa Garip
 */

#pragma once

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
         * @brief Represents an affine transformation: Y = X * W^T + b.
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
         * @brief Applies layers in sequence and indexes parameter paths.
         * @tparam Ls Layer types in forward application order.
         */
        template<typename... Ls>
        class Sequential
        {
            public:
                // Enforces Module concept constraints across parameter pack in C++17.
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

        // Invariant: Module contracts are verified eagerly at declaration scope.
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

        // Invariant: Compile-time layers use fields() or container visits rather than runtime walks.
        static_assert(!Detail::HasParameterWalk<Linear>::value,
                      "A compile-time leaf must be enumerated by its fields(), "
                      "not by a runtime walk");
        static_assert(!Detail::HasParameterWalk<Sequential<Linear, ReLU>>::value,
                      "Sequential must be enumerated as a container, not by a runtime walk");

    } // namespace NN
} // namespace SushiAI
