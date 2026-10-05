/****************************************************************/
/* layer_stack.cpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file layer_stack.cpp
 * @brief Implements runtime sequential layer container execution and parameter walks.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_stack.hpp>

#include <utility>

namespace SushiAI
{
    namespace NN
    {
        void LayerStack::push(std::unique_ptr<ILayer> layer)
        {
            SA_THROW_IF(layer == nullptr,
                        "a null layer was pushed onto a LayerStack at index {}",
                        layers_.size());
            layers_.push_back(std::move(layer));
        }

        Graph::ValueId LayerStack::forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
        {
            SA_THROW_IF(layers_.empty(),
                        "this LayerStack holds no layers, so it would trace as the "
                        "identity function rather than as a model");

            for (const std::unique_ptr<ILayer>& layer : layers_)
                x = layer->forward(builder, x);
            return x;
        }

        void LayerStack::walk_parameters(std::string_view prefix, const ParameterVisitor& visitor)
        {
            // Invariant: Traverses parameters by index matching Sequential naming.
            for (std::size_t i = 0; i < layers_.size(); ++i)
                layers_[i]->walk_parameters(Detail::join(prefix, std::to_string(i)), visitor);
        }

        int64_t LayerStack::output_features() const
        {
            SA_THROW_IF(layers_.empty(),
                        "this LayerStack holds no layers, so it has no output width");
            return layers_.back()->output_features();
        }

        ILayer& LayerStack::layer(std::size_t index) const
        {
            SA_THROW_IF(index >= layers_.size(),
                        "layer {} is out of range; this LayerStack holds {}",
                        index, layers_.size());
            return *layers_[index];
        }

    } // namespace NN
} // namespace SushiAI
