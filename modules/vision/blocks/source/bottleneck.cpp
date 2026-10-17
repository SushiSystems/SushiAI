/**************************************************************************/
/* bottleneck.cpp                                                         */
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
 * @file bottleneck.cpp
 * @brief Implements Bottleneck's trace and the factory that builds its node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/bottleneck.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "bottleneck" row accepts. */
            constexpr std::string_view K_BOTTLENECK_KEYS[] = {
                "filters", "hidden", "shortcut", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the step of both convolutions. */
            constexpr int64_t K_UNIT_STRIDE = 1;

        } // namespace

        Bottleneck::Bottleneck(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
                               bool shortcut, NN::NormOptions norm, int64_t kernel1,
                               int64_t kernel2)
            : cv1(in_channels, hidden_channels, kernel1, K_UNIT_STRIDE, norm),
              cv2(hidden_channels, out_channels, kernel2, K_UNIT_STRIDE, norm),
              adds_input(shortcut && in_channels == out_channels)
        {

        }

        Graph::ValueID Bottleneck::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            const Graph::ValueID y = NN::trace_layer(cv2, trace, NN::trace_layer(cv1, trace, x));
            if (!adds_input)
                return y;

            return trace.builder().add(x, y);
        }

        Shape Bottleneck::output_shape(const Shape& input) const
        {
            return cv2.output_shape(cv1.output_shape(input));
        }

        Result<std::unique_ptr<NN::Node>> make_bottleneck_node(NN::ShapeSpan inputs,
                                                               const Config::Value& attributes,
                                                               std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_BOTTLENECK_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, "hidden", context));
            SA_TRY_ASSIGN(const bool shortcut,
                          Detail::read_bool_or(attributes, "shortcut", context,
                                               Detail::K_DEFAULT_SHORTCUT));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            return Detail::wrap_block(Bottleneck(input[2], filters, hidden, shortcut, norm), input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
