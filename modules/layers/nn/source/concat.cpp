/**************************************************************************/
/* concat.cpp                                                             */
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
 * @file concat.cpp
 * @brief Implements Concat: the shape check at construction, the trace and the factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/concat.hpp>

#include <cstdint>
#include <string>
#include <utility>

#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Names the rank of a per-sample [H, W, C] feature map. */
            constexpr int32_t K_FEATURE_MAP_RANK = 3;

        } // namespace

        Result<std::unique_ptr<Concat>> Concat::create(ShapeSpan inputs)
        {
            SA_FAIL_IF(inputs.empty(), StatusCode::InvalidDescription,
                       "concat needs at least one input and was given none");

            int64_t channels = 0;
            for (std::size_t index = 0; index < inputs.size(); ++index)
            {
                const Shape& input = inputs[index];
                SA_FAIL_IF(input.rank() != K_FEATURE_MAP_RANK, StatusCode::InvalidDescription,
                           "concat input {} has the per-sample shape {}; concat joins rank-3 "
                           "[H, W, C] inputs along C",
                           index, input.to_string());
                SA_FAIL_IF(input[0] != inputs[0][0] || input[1] != inputs[0][1],
                           StatusCode::InvalidDescription,
                           "concat input {} is {} and input 0 is {}; the inputs must agree on "
                           "H and W",
                           index, input.to_string(), inputs[0].to_string());

                channels += input[2];
            }

            const Shape joined{inputs[0][0], inputs[0][1], channels};
            return std::unique_ptr<Concat>(new Concat(inputs.size(), joined));
        }

        Concat::Concat(std::size_t input_count, const Shape& joined)
            : input_count_(input_count), port_{std::string(K_SINGLE_PORT_NAME), joined}
        {

        }

        Status Concat::forward(const Trace& trace, ValueSpan inputs,
                               std::vector<Graph::ValueID>& outputs)
        {
            SA_TRY(Detail::require_input_count(inputs, input_count_));
            SA_TRY_ASSIGN(const Graph::ValueID joined, Detail::bridge_throwing([&trace, &inputs]()
            {
                return trace.builder().concat_channels(inputs);
            }));

            outputs.push_back(joined);
            return Status();
        }

        std::size_t Concat::input_count() const noexcept
        {
            return input_count_;
        }

        void Concat::walk_state(std::string_view, const StateVisitor&)
        {

        }

        SushiRuntime::span<const Port> Concat::ports() const noexcept
        {
            return SushiRuntime::span<const Port>(&port_, 1);
        }

        Result<std::unique_ptr<Node>> make_concat_node(ShapeSpan inputs,
                                                       const Config::Value& attributes,
                                                       std::string_view context)
        {
            SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                       "{} has the unknown key '{}'; a concat row takes no key", context,
                       attributes.keys().front());

            Result<std::unique_ptr<Concat>> made = Concat::create(inputs);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<Node>(std::move(made).value());
        }

    } // namespace NN
} // namespace SushiAI
