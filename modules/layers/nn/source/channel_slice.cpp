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
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /**
             * @brief Checks that @p layer keeps a non-empty range of the channels of @p sample.
             * @param layer The layer whose offset and count are checked.
             * @param sample A per-sample shape.
             * @return InvalidDescription for anything but an [H, W, C] shape, or for a range
             *         that does not fit its channels.
             */
            [[nodiscard]] Status check_range(const ChannelSlice& layer,
                                             const Shape& sample) noexcept
            {
                SA_FAIL_IF(sample.rank() != 3, StatusCode::InvalidDescription,
                           "ChannelSlice needs a per-sample shape of [H, W, C]; got {}",
                           sample.to_string());

                const int64_t channels = sample[2];
                // offset <= channels is tested first, so channels - offset cannot overflow.
                SA_FAIL_IF(layer.offset < 0 || layer.count <= 0 || layer.offset > channels
                               || layer.count > channels - layer.offset,
                           StatusCode::InvalidDescription,
                           "ChannelSlice needs 0 <= offset and 0 < count inside the channels; "
                           "got offset={}, count={} for {} channels",
                           layer.offset, layer.count, channels);

                return Status();
            }
        } // namespace

        Graph::Traced ChannelSlice::forward(const Graph::GraphBuilder& builder,
                                            const Graph::Traced& input) const
        {
            SA_TRY_ASSIGN(const Graph::ValueID x, input);

            const Shape& image = builder.graph().value(x).shape;
            SA_REFUSE_IF(builder, image.rank() != 4,
                         "ChannelSlice needs a rank-4 [N, H, W, C] input; got {}",
                         image.to_string());
            SA_TRY(builder.observe(check_range(*this, Shape{image[1], image[2], image[3]})));

            return builder.slice_channels(x, offset, count);
        }

        Result<Shape> ChannelSlice::output_shape(const Shape& input) const
        {
            SA_TRY(check_range(*this, input));

            return Shape{input[0], input[1], count};
        }

    } // namespace NN
} // namespace SushiAI
