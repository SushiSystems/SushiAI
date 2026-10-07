/**************************************************************************/
/* ilayer.hpp                                                             */
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
 * @file ilayer.hpp
 * @brief Declares runtime-polymorphic layer interfaces and adapters.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <utility>
#include <SushiAI/core/shape.hpp>
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

                /** @brief Returns the per-sample shape after this layer, batch axis excluded. */
                virtual Shape output_shape() const noexcept = 0;

                /** @brief Returns the element count of output_shape(). */
                int64_t output_features() const noexcept { return output_shape().num_elements(); }
        };

        /**
         * @class LayerAdapter
         * @brief Adapts a compile-time Module to the runtime ILayer interface.
         * @tparam L Wrapped layer type satisfying the Module contract.
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
                 * @brief Wraps a module instance with the per-sample shape it produces.
                 * @param output_shape Must equal what @p layer traces, batch axis excluded.
                 */
                LayerAdapter(L layer, Shape output_shape)
                    : layer_(std::move(layer)), output_shape_(output_shape) {}

                /**
                 * @brief Wraps a module instance that produces a rank-1 feature vector.
                 * @param output_features Must be positive; the shape becomes [output_features].
                 */
                LayerAdapter(L layer, int64_t output_features)
                    : LayerAdapter(std::move(layer), Shape{output_features}) {}

                Graph::ValueId forward(const Graph::GraphBuilder& builder,
                                       Graph::ValueId x) override
                {
                    return layer_.forward(builder, x);
                }

                void walk_parameters(std::string_view prefix,
                                     const ParameterVisitor& visitor) override
                {
                    // Delegates to compile-time parameter traversal without recursion.
                    for_each_parameter(layer_, prefix, visitor);
                }

                Shape output_shape() const noexcept override { return output_shape_; }

                /** @brief The wrapped layer, for a test that wants to inspect it. */
                L& layer() noexcept { return layer_; }

                /** @brief The wrapped layer, read-only. */
                const L& layer() const noexcept { return layer_; }

            private:
                L layer_;
                Shape output_shape_{};
        };

    } // namespace NN
} // namespace SushiAI
