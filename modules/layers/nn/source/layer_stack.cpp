/**************************************************************************/
/* layer_stack.cpp                                                        */
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
 * @file layer_stack.cpp
 * @brief Implements runtime sequential layer container execution and parameter walks.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/failure/check.hpp>

#include <utility>

namespace SushiAI
{
    namespace NN
    {
        void LayerStack::push(std::unique_ptr<Layer> layer) noexcept
        {
            SA_FATAL_IF(layer == nullptr,
                        "a null layer was pushed onto a LayerStack at index {}",
                        layers_.size());
            layers_.push_back(std::move(layer));
        }

        Graph::Traced LayerStack::forward(const Graph::GraphBuilder& builder,
                                          const Graph::Traced& input) noexcept
        {
            SA_TRY(input);
            SA_REFUSE_IF(builder, layers_.empty(),
                         "this LayerStack holds no layers, so it would trace as the "
                         "identity function rather than as a model");

            Graph::Traced traced = input;
            for (const std::unique_ptr<Layer>& layer : layers_)
            {
                traced = layer->forward(builder, traced);
                SA_TRY(traced);
            }
            return traced;
        }

        void LayerStack::walk_parameters(std::string_view prefix, const ParameterVisitor& visitor) noexcept
        {
            // Invariant: Traverses parameters by index matching Sequential naming.
            for (std::size_t i = 0; i < layers_.size(); ++i)
                layers_[i]->walk_parameters(Detail::join(prefix, std::to_string(i)), visitor);
        }

        int64_t LayerStack::output_features() const noexcept
        {
            SA_FATAL_IF(layers_.empty(),
                        "this LayerStack holds no layers, so it has no output width");
            return layers_.back()->output_features();
        }

        Shape LayerStack::output_shape() const noexcept
        {
            SA_FATAL_IF(layers_.empty(),
                        "this LayerStack holds no layers, so it has no output shape");
            return layers_.back()->output_shape();
        }

        Result<Shape> LayerStack::output_shape(const Shape&) const noexcept
        {
            return output_shape();
        }

        Layer& LayerStack::layer(std::size_t index) const noexcept
        {
            SA_FATAL_IF(index >= layers_.size(),
                        "layer {} is out of range; this LayerStack holds {}",
                        index, layers_.size());
            return *layers_[index];
        }

    } // namespace NN
} // namespace SushiAI
