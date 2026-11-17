/**************************************************************************/
/* caffe_op_batch_norm.cpp                                                */
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
 * @file caffe_op_batch_norm.cpp
 * @brief Implements the translator of a Caffe BatchNorm layer and of the Scale layer after it.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/shape/shape.hpp>

#include "caffe_parameters.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail::Caffe
        {
            namespace
            {
                /** @brief Holds the epsilon of a BatchNorm layer whose batch_norm_param states none. */
                constexpr double K_DEFAULT_EPSILON = 1e-5;

                /** @brief Holds the blobs of a BatchNorm layer: mean, variance, scale factor. */
                constexpr std::size_t K_BATCH_NORM_BLOBS = 3;

                /** @brief Holds what the Scale layer after a BatchNorm layer states. */
                struct ScaleAfter
                {
                    const Interchange::CaffeLayer* layer = nullptr; /**< The layer, or null. */
                    std::int64_t index = -1;                        /**< Its position, or -1. */
                    bool use_bias = false;                          /**< Whether it adds a bias. */
                };

                /**
                 * @brief Finds the Scale layer that reads the top of the BatchNorm layer @p layer.
                 *
                 * The Scale is the first layer to read that top, with one bottom and one top.
                 * @return The Scale, or an empty ScaleAfter when none follows; Unsupported for a
                 *         Scale that scales along another axis than the channels, or one whose top
                 *         differs from the BatchNorm's while another layer reads that top.
                 */
                [[nodiscard]] Result<ScaleAfter> find_scale(const Interchange::CaffeLayer& layer,
                                                            std::int64_t index,
                                                            const TranslationProgress& progress)
                {
                    ScaleAfter found;
                    const std::string& top = layer.tops.front();
                    const std::int64_t reader = find_reader(progress, top, index);
                    if (reader < 0)
                        return found;
                    const Interchange::CaffeLayer& next =
                        progress.net->layers[static_cast<std::size_t>(reader)];
                    if (next.type != "Scale" || next.bottoms.size() != 1 || next.tops.size() != 1)
                        return found;

                    const bool in_place = next.tops.front() == top;
                    SA_FAIL_IF(!in_place && find_reader(progress, top, reader) >= 0,
                               StatusCode::Unsupported,
                               "{} feeds the {} and another layer; the pair folds into one row "
                               "that has one result",
                               describe_layer(layer), describe_layer(next));

                    const ParameterReader parameters(next, "scale_param");
                    SA_TRY_ASSIGN(found.use_bias, parameters.read_bool("bias_term", false));
                    SA_TRY_ASSIGN(const std::int64_t axis, parameters.read_integer("axis", 1));
                    SA_TRY_ASSIGN(const std::int64_t axes, parameters.read_integer("num_axes", 1));
                    SA_FAIL_IF(axis != 1 || axes != 1, StatusCode::Unsupported,
                               "{} scales {} axes from axis {}; only the channels are translated",
                               describe_layer(next), axes, axis);
                    found.layer = &next;
                    found.index = reader;
                    return found;
                }

                /** @brief Appends the four tensors of the row @p row, folding what Caffe splits. */
                [[nodiscard]] Status store_weights(const Interchange::CaffeLayer& layer,
                                                   const ScaleAfter& scale, std::int64_t channels,
                                                   std::int64_t row, TranslationProgress& progress)
                {
                    SA_TRY(expect_blobs(progress, layer, K_BATCH_NORM_BLOBS));
                    if (scale.layer != nullptr)
                    {
                        SA_TRY(expect_blobs(progress, *scale.layer, scale.use_bias ? 2 : 1));
                    }
                    if (!progress.has_weights)
                        return Status();

                    const std::size_t count = static_cast<std::size_t>(channels);
                    SA_TRY_ASSIGN(std::vector<float> mean, read_blob(layer, 0, count));
                    SA_TRY_ASSIGN(std::vector<float> variance, read_blob(layer, 1, count));
                    SA_TRY_ASSIGN(const std::vector<float> factor, read_blob(layer, 2, 1));
                    const float inverse = factor.front() == 0.0f ? 0.0f : 1.0f / factor.front();
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        mean[i] *= inverse;
                        variance[i] *= inverse;
                    }

                    std::vector<float> gamma(count, 1.0f);
                    std::vector<float> beta(count, 0.0f);
                    if (scale.layer != nullptr)
                    {
                        SA_TRY_ASSIGN(gamma, read_blob(*scale.layer, 0, count));
                        if (scale.use_bias)
                        {
                            SA_TRY_ASSIGN(beta, read_blob(*scale.layer, 1, count));
                        }
                    }

                    const Shape per_channel{channels};
                    append_tensor(progress, entry_name(row, "weight"), per_channel, gamma);
                    append_tensor(progress, entry_name(row, "bias"), per_channel, beta);
                    append_tensor(progress, entry_name(row, "running_mean"), per_channel, mean);
                    append_tensor(progress, entry_name(row, "running_var"), per_channel, variance);
                    return Status();
                }
            } // namespace

            Status translate_batch_norm(const Interchange::CaffeLayer& layer, std::int64_t index,
                                        TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(BlobValue value, read_bottom(progress, layer, 0));
                SA_TRY(expect_image(layer, value));

                const ParameterReader parameters(layer, "batch_norm_param");
                SA_TRY_ASSIGN(const double epsilon,
                              parameters.read_real("eps", K_DEFAULT_EPSILON));
                SA_FAIL_IF(!(epsilon > 0.0), StatusCode::InvalidDescription,
                           "{} states the epsilon {}; it needs a positive one",
                           describe_layer(layer), epsilon);
                SA_TRY_ASSIGN(const ScaleAfter scale, find_scale(layer, index, progress));

                const std::int64_t sources[] = {value.row};
                value.row = append_row(
                    progress, "batchnorm2d",
                    make_attributes({"bn_epsilon"}, {Config::Value::real(epsilon)}), sources);
                SA_TRY(store_weights(layer, scale, value.channels, value.row, progress));

                if (scale.layer == nullptr)
                {
                    write_blob(progress, layer.tops.front(), value, index);
                    return Status();
                }
                progress.absorbed[static_cast<std::size_t>(scale.index)] = true;
                write_blob(progress, scale.layer->tops.front(), value, scale.index);
                return Status();
            }

            Status translate_scale(const Interchange::CaffeLayer& layer, std::int64_t,
                                   TranslationProgress&)
            {
                SA_FAIL_IF(true, StatusCode::Unsupported,
                           "{} does not read the top of a BatchNorm layer, and only that pair "
                           "has a row",
                           describe_layer(layer));
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
