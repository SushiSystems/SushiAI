/**************************************************************************/
/* channel_slice.cpp                                                      */
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
 * @file channel_slice.cpp
 * @brief Implements the tracing and shape rule of the channel slicing layer.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/channel_slice.hpp>

#include <cstdint>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /**
             * @brief Checks that @p layer keeps a non-empty range of the channels of @p sample.
             * @param layer The layer whose offset and count are checked.
             * @param sample A per-sample shape; anything but an [H, W, C] shape throws, and so
             *        does a range that does not fit its channels.
             */
            void check_range(const ChannelSlice& layer, const Shape& sample)
            {
                SA_THROW_IF(sample.rank() != 3,
                            "ChannelSlice needs a per-sample shape of [H, W, C]; got {}",
                            sample.to_string());

                const int64_t channels = sample[2];
                // offset <= channels is tested first, so channels - offset cannot overflow.
                SA_THROW_IF(layer.offset < 0 || layer.count <= 0 || layer.offset > channels
                                || layer.count > channels - layer.offset,
                            "ChannelSlice needs 0 <= offset and 0 < count inside the channels; "
                            "got offset={}, count={} for {} channels",
                            layer.offset, layer.count, channels);
            }
        } // namespace

        Graph::ValueID ChannelSlice::forward(const Graph::GraphBuilder& builder,
                                             Graph::ValueID x) const
        {
            const Shape& image = builder.graph().value(x).shape;
            SA_THROW_IF(image.rank() != 4,
                        "ChannelSlice needs a rank-4 [N, H, W, C] input; got {}",
                        image.to_string());
            (void)output_shape(Shape{image[1], image[2], image[3]});

            return builder.slice_channels(x, offset, count);
        }

        Shape ChannelSlice::output_shape(const Shape& input) const
        {
            check_range(*this, input);

            return Shape{input[0], input[1], count};
        }

    } // namespace NN
} // namespace SushiAI
