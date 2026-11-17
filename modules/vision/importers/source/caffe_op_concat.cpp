/**************************************************************************/
/* caffe_op_concat.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file caffe_op_concat.cpp
 * @brief Implements the translator of a Caffe Concat layer.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/axis_translation.hpp>
#include <SushiAI/interchange/caffe_net.hpp>

#include "caffe_parameters.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Holds the rank of a Caffe image blob, which the axis is counted over. */
                constexpr std::int64_t K_CAFFE_IMAGE_RANK = 4;

                /** @brief Holds the position of the channel axis once the layout is channels-last. */
                constexpr std::int64_t K_CHANNELS_LAST_CHANNEL_AXIS = 3;

                /**
                 * @brief Returns Unsupported unless @p layer joins the channels.
                 *
                 * The legacy concat_dim counts as the axis when axis is absent.
                 */
                [[nodiscard]] Status expect_channel_axis(const Interchange::CaffeLayer& layer)
                {
                    const ParameterReader parameters(layer, "concat_param");
                    SA_TRY_ASSIGN(const std::int64_t legacy, parameters.read_integer("concat_dim", 1));
                    SA_TRY_ASSIGN(const std::int64_t stated, parameters.read_integer("axis", legacy));
                    SA_TRY_ASSIGN(const std::int64_t axis,
                                  Interchange::channels_last_axis(stated, K_CAFFE_IMAGE_RANK));
                    SA_FAIL_IF(axis != K_CHANNELS_LAST_CHANNEL_AXIS, StatusCode::Unsupported,
                               "{} joins along the axis {}; the concat row joins the channels",
                               describe_layer(layer), stated);
                    return Status();
                }
            } // namespace

            Status translate_concat(const Interchange::CaffeLayer& layer, std::int64_t index,
                                    TranslationProgress& progress)
            {
                SA_FAIL_IF(layer.bottoms.empty() || layer.tops.size() != 1,
                           StatusCode::InvalidDescription,
                           "{} has {} bottoms and {} tops; this type takes one or more and 1",
                           describe_layer(layer), layer.bottoms.size(), layer.tops.size());
                SA_TRY(expect_channel_axis(layer));

                std::vector<std::int64_t> sources;
                BlobValue joined;
                for (std::size_t position = 0; position < layer.bottoms.size(); ++position)
                {
                    SA_TRY_ASSIGN(const BlobValue part, read_bottom(progress, layer, position));
                    SA_TRY(expect_image(layer, part));
                    SA_FAIL_IF(position != 0 &&
                                   (part.height != joined.height || part.width != joined.width),
                               StatusCode::InvalidDescription,
                               "{} joins a {} by {} blob to a {} by {} one", describe_layer(layer),
                               part.height, part.width, joined.height, joined.width);
                    if (position == 0)
                    {
                        joined = part;
                        joined.channels = 0;
                    }
                    joined.channels += part.channels;
                    sources.push_back(part.row);
                }

                // One bottom joins nothing: the top reads the row of the bottom, held by joined.
                if (sources.size() > 1)
                    joined.row = append_row(
                        progress, "concat", make_attributes({}, {}),
                        SushiRuntime::span<const std::int64_t>(sources.data(), sources.size()));
                write_blob(progress, layer.tops.front(), joined, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
