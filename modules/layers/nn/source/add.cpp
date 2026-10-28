/**************************************************************************/
/* add.cpp                                                                */
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
 * @file add.cpp
 * @brief Implements Add: the shape check at construction, the trace and the factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/add.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Holds the number of operands an addition takes. */
            constexpr std::size_t K_OPERAND_COUNT = 2;

            /** @brief Names the rank of a per-sample [H, W, C] feature map. */
            constexpr int32_t K_FEATURE_MAP_RANK = 3;

        } // namespace

        Result<std::unique_ptr<Add>> Add::create(ShapeSpan inputs) noexcept
        {
            SA_FAIL_IF(inputs.size() != K_OPERAND_COUNT, StatusCode::InvalidDescription,
                       "add takes {} inputs and was given {}", K_OPERAND_COUNT, inputs.size());

            for (std::size_t index = 0; index < inputs.size(); ++index)
            {
                const Shape& input = inputs[index];
                SA_FAIL_IF(input.rank() != K_FEATURE_MAP_RANK, StatusCode::InvalidDescription,
                           "add input {} has the per-sample shape {}; add sums rank-3 [H, W, C] "
                           "inputs",
                           index, input.to_string());
            }

            SA_FAIL_IF(inputs[0] != inputs[1], StatusCode::InvalidDescription,
                       "add inputs must share one shape; input 0 is {} and input 1 is {}",
                       inputs[0].to_string(), inputs[1].to_string());

            return std::unique_ptr<Add>(new Add(inputs[0]));
        }

        Add::Add(const Shape& shape)
            : port_{std::string(K_SINGLE_PORT_NAME), shape}
        {

        }

        Status Add::forward(const Trace& trace, ValueSpan inputs,
                            std::vector<Graph::ValueID>& outputs)
        {
            SA_TRY(Detail::require_input_count(inputs, K_OPERAND_COUNT));
            SA_TRY_ASSIGN(const Graph::ValueID sum, trace.builder().add(inputs[0], inputs[1]));

            outputs.push_back(sum);
            return Status();
        }

        std::size_t Add::input_count() const noexcept
        {
            return K_OPERAND_COUNT;
        }

        void Add::walk_state(std::string_view, const StateVisitor&)
        {

        }

        SushiRuntime::span<const Port> Add::ports() const noexcept
        {
            return SushiRuntime::span<const Port>(&port_, 1);
        }

        Result<std::unique_ptr<Node>> make_add_node(ShapeSpan inputs,
                                                    const Config::Value& attributes,
                                                    std::string_view context) noexcept
        {
            SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                       "{} has the unknown key '{}'; an add row takes no key", context,
                       attributes.keys().front());

            Result<std::unique_ptr<Add>> made = Add::create(inputs);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<Node>(std::move(made).value());
        }

    } // namespace NN
} // namespace SushiAI
