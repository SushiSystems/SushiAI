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
                 * @brief Returns the feature count per sample after this layer.
                 * @return Output feature count.
                 */
                virtual int64_t output_features() const noexcept = 0;
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
                 * @brief Wraps an existing module instance with its output feature count.
                 * @param layer Layer moved into the adapter.
                 * @param output_features Output dimension produced by this layer.
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
                    // Delegates to compile-time parameter traversal without recursion.
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
