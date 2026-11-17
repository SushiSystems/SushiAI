/**************************************************************************/
/* caffe_op_softmax.cpp                                                   */
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
 * @file caffe_op_softmax.cpp
 * @brief Implements the translator of a Caffe Softmax layer into a softmax row.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
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
                /** @brief Holds the rank of an image blob, [N, C, H, W]. */
                constexpr std::int64_t K_IMAGE_BLOB_RANK = 4;

                /** @brief Holds the rank of a vector blob, [N, K]. */
                constexpr std::int64_t K_VECTOR_BLOB_RANK = 2;

                /**
                 * @brief Returns the per-sample axis of the model that a Caffe axis names.
                 * @return The position among the axes of one sample, batch axis excluded;
                 *         Unsupported for the batch axis, InvalidDescription for an axis outside
                 *         the blob.
                 */
                [[nodiscard]] Result<std::int64_t> sample_axis(
                    const Interchange::CaffeLayer& layer, std::int64_t axis, std::int64_t rank)
                {
                    const Result<std::int64_t> placed = Interchange::channels_last_axis(axis, rank);
                    SA_TRY(SA_ANNOTATE(placed.status(), "{}", describe_layer(layer)));
                    const std::int64_t moved = placed.value();
                    SA_FAIL_IF(moved == 0, StatusCode::Unsupported,
                               "{} normalises over the batch axis, which a row cannot state",
                               describe_layer(layer));
                    return moved - 1;
                }
            } // namespace

            Status translate_softmax(const Interchange::CaffeLayer& layer, std::int64_t index,
                                     TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(const BlobValue source, read_bottom(progress, layer, 0));

                const ParameterReader parameters(layer, "softmax_param");
                SA_TRY_ASSIGN(const std::int64_t axis, parameters.read_integer("axis", 1));
                const std::int64_t rank = source.is_image ? K_IMAGE_BLOB_RANK : K_VECTOR_BLOB_RANK;
                SA_TRY_ASSIGN(const std::int64_t chosen, sample_axis(layer, axis, rank));

                // The sample's last axis is the row's default, so only another axis is stated.
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                if (chosen != rank - 2)
                {
                    keys.emplace_back("axis");
                    values.push_back(Config::Value::integer(chosen));
                }

                const std::int64_t sources[] = {source.row};
                BlobValue result = source;
                result.row = append_row(progress, "softmax",
                                        make_attributes(std::move(keys), std::move(values)),
                                        sources);
                write_blob(progress, layer.tops.front(), result, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
