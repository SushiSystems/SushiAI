/**************************************************************************/
/* layer_node.cpp                                                         */
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
 * @file layer_node.cpp
 * @brief Implements LayerNode: the trace, the state walk and the port of a run-time layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_node.hpp>

#include <cstddef>
#include <string>
#include <utility>

#include <SushiAI/core/check.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Holds the number of values a layer reads. */
            constexpr std::size_t K_INPUT_COUNT = 1;

            /** @brief Returns the port @p layer produces; a null layer is fatal. */
            Port port_of(const ILayer* layer)
            {
                SA_FATAL_IF(layer == nullptr, "a LayerNode was built over a null layer");

                return Port{std::string(K_SINGLE_PORT_NAME), layer->output_shape()};
            }

        } // namespace

        LayerNode::LayerNode(std::unique_ptr<ILayer> layer)
            : layer_(std::move(layer)), port_(port_of(layer_.get()))
        {

        }

        Status LayerNode::forward(const Trace& trace, ValueSpan inputs,
                                  std::vector<Graph::ValueId>& outputs)
        {
            SA_TRY(Detail::require_input_count(inputs, K_INPUT_COUNT));
            SA_TRY_ASSIGN(const Graph::ValueId traced,
                          Detail::bridge_throwing([this, &trace, &inputs]()
            {
                return layer_->forward(trace.builder(), inputs[0]);
            }));

            outputs.push_back(traced);
            return Status();
        }

        std::size_t LayerNode::input_count() const noexcept
        {
            return K_INPUT_COUNT;
        }

        void LayerNode::walk_state(std::string_view prefix, const StateVisitor& visitor)
        {
            const auto hand_on = [&visitor](const std::string& name, Parameter& parameter)
            {
                visitor.parameter(name, parameter);
            };
            const ParameterVisitor parameters(hand_on);
            layer_->walk_parameters(prefix, parameters);
        }

        SushiRuntime::span<const Port> LayerNode::ports() const noexcept
        {
            return SushiRuntime::span<const Port>(&port_, 1);
        }

        ILayer& LayerNode::layer() noexcept
        {
            return *layer_;
        }

        const ILayer& LayerNode::layer() const noexcept
        {
            return *layer_;
        }

    } // namespace NN
} // namespace SushiAI
