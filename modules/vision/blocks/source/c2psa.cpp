/**************************************************************************/
/* c2psa.cpp                                                              */
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
 * @file c2psa.cpp
 * @brief Implements C2PSA's constructor, its trace, its shape rule and the factory of its node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/c2psa.hpp>

#include <cstddef>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "c2psa" row accepts. */
            constexpr std::string_view K_C2PSA_KEYS[] = {
                "filters", "hidden", "repeats", "heads", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

        } // namespace

        C2PSA::C2PSA(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
                     int64_t repeats, int64_t heads, NN::NormOptions norm)
            : hidden(hidden_channels),
              cv1(in_channels, 2 * hidden_channels, Detail::K_POINTWISE, Detail::K_POINTWISE,
                  norm),
              cv2(2 * hidden_channels, out_channels, Detail::K_POINTWISE, Detail::K_POINTWISE,
                  norm)
        {
            for (int64_t index = 0; index < repeats; ++index)
                m.push_back(PSABlock(hidden_channels, heads, norm));
        }

        Graph::ValueId C2PSA::forward(const NN::Trace& trace, Graph::ValueId x)
        {
            const Graph::GraphBuilder& builder = trace.builder();
            const Graph::ValueId stem = NN::trace_layer(cv1, trace, x);

            const Graph::ValueId kept = builder.slice_channels(stem, 0, hidden);
            Graph::ValueId chained = builder.slice_channels(stem, hidden, hidden);
            for (PSABlock& block : m)
                chained = NN::trace_layer(block, trace, chained);

            const Graph::ValueId parts[] = {kept, chained};
            return NN::trace_layer(
                cv2, trace,
                builder.concat_channels(SushiRuntime::span<const Graph::ValueId>(parts)));
        }

        Shape C2PSA::output_shape(const Shape& input) const
        {
            // cv1 refuses an input that is not [H, W, in_channels] and keeps H and W.
            const Shape stem = cv1.output_shape(input);

            return cv2.output_shape(Shape{stem[0], stem[1], cv2.conv.in_channels});
        }

        Result<std::unique_ptr<NN::Node>> make_c2psa_node(NN::ShapeSpan inputs,
                                                          const Config::Value& attributes,
                                                          std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_C2PSA_KEYS), context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, "hidden", context));
            SA_TRY_ASSIGN(const int64_t repeats,
                          Detail::read_positive_int_or(attributes, "repeats", context,
                                                       Detail::K_DEFAULT_REPEATS));
            SA_TRY_ASSIGN(const int64_t heads,
                          Detail::read_positive_int_or(attributes, "heads", context,
                                                       Detail::default_head_count(hidden)));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            SA_FAIL_IF(input[2] != filters, StatusCode::InvalidDescription,
                       "{} 'filters' must equal the input width, because C2PSA keeps it; it is {}, "
                       "and the input has {} channels",
                       context, filters, input[2]);
            SA_FAIL_IF(hidden % heads != 0 || hidden / heads < Detail::K_SMALLEST_HEAD_WIDTH,
                       StatusCode::InvalidDescription,
                       "{} 'hidden' must split into {} parts of at least {} channels; it is {}",
                       context, heads, Detail::K_SMALLEST_HEAD_WIDTH, hidden);

            return Detail::wrap_block(C2PSA(input[2], filters, hidden, repeats, heads, norm),
                                      input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
