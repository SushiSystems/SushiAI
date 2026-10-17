/**************************************************************************/
/* dropout.cpp                                                            */
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
 * @file dropout.cpp
 * @brief Implements Dropout's two trace forms, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/dropout.hpp>

#include <cstdint>
#include <utility>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/nn/node_attributes.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Names the key that holds a row's drop rate. */
            constexpr std::string_view K_RATE_KEY = "rate";

            /** @brief Lists every key a "dropout" row accepts. */
            constexpr std::string_view K_DROPOUT_KEYS[] = {K_RATE_KEY};

            /** @brief Returns InvalidDescription unless @p rate lies in [0, 1). */
            [[nodiscard]] Status check_rate(double rate)
            {
                SA_FAIL_IF(!(rate >= 0.0 && rate < 1.0), StatusCode::InvalidDescription,
                           "Dropout needs a rate in [0, 1); got {}", rate);

                return Status();
            }

            /** @brief Returns @p x, recording on @p trace a rate that does not fit. */
            [[nodiscard]] Graph::ValueID pass_through(const Dropout& layer, const Trace& trace,
                                                      Graph::ValueID x)
            {
                const Status checked = check_rate(layer.rate);
                if (!checked.is_ok())
                    trace.fail(checked);

                return x;
            }
        } // namespace

        Graph::ValueID Dropout::trace_training(const Trace& trace, Graph::ValueID x)
        {
            const Status checked = check_rate(rate);
            if (!checked.is_ok())
            {
                trace.fail(checked);
                return x;
            }

            if (rate == 0.0)
                return x;

            const std::uint64_t seed =
                Graph::dropout_site_seed(trace.seed(), trace.claim_random_site());
            const Result<Graph::DropoutResult> masked = trace.builder().dropout(x, rate, seed);
            if (!masked.has_value())
            {
                trace.fail(masked.status());
                return x;
            }

            return masked.value().output;
        }

        Graph::ValueID Dropout::trace_evaluation(const Trace& trace, Graph::ValueID x)
        {
            return pass_through(*this, trace, x);
        }

        Shape Dropout::output_shape(const Shape& input) const
        {
            return input;
        }

        Result<std::unique_ptr<Node>> make_dropout_node(ShapeSpan inputs,
                                                        const Config::Value& attributes,
                                                        std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_DROPOUT_KEYS),
                                               context));
            SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                       "{} reads one input, but it was given {}", context, inputs.size());

            Dropout layer{};
            SA_TRY_ASSIGN(layer.rate, Detail::read_real(attributes, K_RATE_KEY, context));
            SA_TRY(SA_ANNOTATE(check_rate(layer.rate), "{} '{}'", context, K_RATE_KEY));

            return Detail::wrap_block(std::move(layer), inputs[0]);
        }

    } // namespace NN
} // namespace SushiAI
