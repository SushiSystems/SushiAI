/**************************************************************************/
/* layers.hpp                                                             */
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
 * @file layers.hpp
 * @brief Declares core neural network layers that trace into graph IR.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>

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

            /** @brief The fill of the weight; fan-scaled unless a model states another. */
            ParameterInit weight_init = ParameterInit::fan_scaled();

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
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x);

            /**
             * @brief Returns the per-sample shape forward() produces, [out_features].
             * @param input The per-sample input shape; must be [in_features].
             */
            Shape output_shape(const Shape& input) const;
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
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
            {
                return builder.relu(x);
            }

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const
            {
                return input;
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
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
            {
                return builder.gelu(x);
            }

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const
            {
                return input;
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
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
            {
                return builder.tanh(x);
            }

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const
            {
                return input;
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
            Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x) const
            {
                return builder.sigmoid(x);
            }

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const
            {
                return input;
            }
        };

        /**
         * @class Sequential
         * @brief Applies layers in sequence and indexes parameter paths.
         * @tparam Ls Module types in forward application order.
         */
        template<typename... Ls>
        class Sequential
        {
            public:
                static_assert((!Detail::HasHalfModalPair<Ls> && ...),
                              "A type in a Sequential has one of trace_training and "
                              "trace_evaluation and lacks the other. A layer whose emission "
                              "depends on the mode states both. Check the layer named in the "
                              "instantiation below, not this line.");

                static_assert((!Detail::HasModalMemberAndForward<Ls> && ...),
                              "A type in a Sequential has trace_training or trace_evaluation "
                              "and a forward as well, so which of them traces it is unstated. "
                              "A modal layer has the pair and no forward. Check the layer named "
                              "in the instantiation below, not this line.");

                static_assert((Traceable<Ls> && ...),
                              "Every type in a Sequential needs a static fields() listing its "
                              "state and one of three forms: "
                              "forward(const Graph::GraphBuilder&, Graph::ValueID), "
                              "forward(const Trace&, Graph::ValueID), or the pair "
                              "trace_training and trace_evaluation over "
                              "(const Trace&, Graph::ValueID), each returning exactly "
                              "Graph::ValueID. Check the layer named in the instantiation "
                              "below, not this line.");

                /**
                 * @brief Constructs from the layers themselves.
                 * @param layers The layers, in application order; moved in.
                 */
                constexpr explicit Sequential(Ls... layers) : layers_(std::move(layers)...) {}

                /** @brief No parameters of its own; the children own them all. */
                static constexpr auto fields() { return std::tuple<>{}; }

                /**
                 * @brief Traces every layer in order; exists only when every layer is a Module.
                 * @tparam Enabled Leave at its default; false removes the overload.
                 * @param builder Where the nodes go.
                 * @param x The input value.
                 * @return The last layer's result.
                 */
                template<bool Enabled = true,
                         std::enable_if_t<Enabled && (Module<Ls> && ...), int> = 0>
                Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x)
                {
                    std::apply([&](auto&... layer) { ((x = layer.forward(builder, x)), ...); },
                               layers_);
                    return x;
                }

                /**
                 * @brief Traces every layer in order, each in the form trace_layer() picks.
                 * @param trace Passed to every layer unread; the mode is not consulted here.
                 * @param x The input value.
                 * @return The last layer's result.
                 */
                Graph::ValueID forward(const Trace& trace, Graph::ValueID x)
                {
                    std::apply([&](auto&... layer)
                    {
                        ((x = trace_child(layer, trace, x)), ...);
                    }, layers_);
                    return x;
                }

                /**
                 * @brief Returns the per-sample shape forward() produces over @p input.
                 * @tparam Enabled Leave at its default; false removes the overload.
                 * @param input The per-sample shape the first layer reads, batch axis excluded.
                 */
                template<bool Enabled = true,
                         std::enable_if_t<
                             Enabled && (Detail::HasOutputShapeOfInput<Ls>::value && ...), int> = 0>
                Shape output_shape(const Shape& input) const
                {
                    Shape shape = input;
                    std::apply([&](const auto&... layer)
                    {
                        ((shape = layer.output_shape(shape)), ...);
                    }, layers_);
                    return shape;
                }

                /**
                 * @brief Applies @p visitor to each child with its index.
                 * @param visitor Called as visitor(std::size_t index, auto& child).
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
                /** @brief Traces one child and throws the failure it recorded on @p trace. */
                template<typename L>
                static Graph::ValueID trace_child(L& layer, const Trace& trace, Graph::ValueID x)
                {
                    const Graph::ValueID traced = trace_layer(layer, trace, x);
                    SA_THROW_IF(!trace.status().is_ok(), "{}", trace.status().message());
                    return traced;
                }

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
        static_assert(TraceModule<Sequential<Linear, ReLU>>,
                      "Sequential must take a trace whatever its layers are");
        static_assert(Detail::HasOutputShapeOfInput<Sequential<Linear, ReLU>>::value,
                      "Sequential must map an input shape to its output shape to be a node");
        static_assert(!ModalModule<Sequential<Linear, ReLU>>,
                      "Sequential never reads the mode, so it is not a modal layer");
        static_assert(Module<Sequential<Sequential<Linear, ReLU>, Linear>>,
                      "A Sequential of Modules must stay a Module when it nests");
        static_assert(!TraceModule<Linear> && !ModalModule<Linear>,
                      "A builder-only leaf must be traced through its builder forward");
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
