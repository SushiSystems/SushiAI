/**************************************************************************/
/* segment_head.cpp                                                       */
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
 * @file segment_head.cpp
 * @brief Implements SegmentHead's trace and the factory that builds its node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/segment_head.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>

#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "segment_head" row accepts. */
            constexpr std::string_view K_SEGMENT_HEAD_KEYS[] = {
                "classes", "hidden", "scale", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the side of the window of the first convolution. */
            constexpr int64_t K_FEATURE_KERNEL = 3;

            /** @brief Holds the upsample factor of a row that states none. */
            constexpr int64_t K_DEFAULT_SCALE = 1;

            /** @brief Builds the biased, unpadded 1x1 convolution that scores the classes. */
            [[nodiscard]] NN::Conv2d biased_pointwise(int64_t in_channels, int64_t out_channels)
            {
                NN::Conv2d layer;
                layer.in_channels = in_channels;
                layer.out_channels = out_channels;

                return layer;
            }
        } // namespace

        SegmentHead::SegmentHead(int64_t in_channels, int64_t hidden, int64_t classes,
                                 int64_t scale, NN::NormOptions norm)
            : cv1(in_channels, hidden, K_FEATURE_KERNEL, Detail::K_POINTWISE, norm),
              cv2(biased_pointwise(hidden, classes)),
              up{scale}
        {

        }

        Graph::ValueID SegmentHead::forward(const NN::Trace& trace, Graph::ValueID x)
        {
            x = NN::trace_layer(cv1, trace, x);
            x = NN::trace_layer(cv2, trace, x);

            return NN::trace_layer(up, trace, x);
        }

        Shape SegmentHead::output_shape(const Shape& input) const
        {
            return up.output_shape(cv2.output_shape(cv1.output_shape(input)));
        }

        Result<std::unique_ptr<NN::Node>> make_segment_head_node(NN::ShapeSpan inputs,
                                                                 const Config::Value& attributes,
                                                                 std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_SEGMENT_HEAD_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t classes,
                          Detail::read_positive_int(attributes, "classes", context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int_or(attributes, "hidden", context, input[2]));
            SA_TRY_ASSIGN(const int64_t scale,
                          Detail::read_positive_int_or(attributes, "scale", context,
                                                       K_DEFAULT_SCALE));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            return Detail::wrap_block(SegmentHead(input[2], hidden, classes, scale, norm), input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
