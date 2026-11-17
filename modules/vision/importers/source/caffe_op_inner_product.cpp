/**************************************************************************/
/* caffe_op_inner_product.cpp                                             */
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
 * @file caffe_op_inner_product.cpp
 * @brief Implements the translator of a Caffe InnerProduct layer.
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
                /** @brief Holds what an inner_product_param states, defaults filled in. */
                struct InnerProduct
                {
                    std::int64_t units = 0; /**< Output features. */
                    bool use_bias = true;   /**< Whether the layer adds a bias. */
                };

                /**
                 * @brief Reads the inner_product_param of @p layer over its defaults.
                 * @return Unsupported for another axis or a transposed weight; InvalidDescription
                 *         for a num_output below 1.
                 */
                [[nodiscard]] Result<InnerProduct> read_inner_product(
                    const Interchange::CaffeLayer& layer)
                {
                    const ParameterReader parameters(layer, "inner_product_param");
                    InnerProduct read;
                    SA_TRY_ASSIGN(read.units, parameters.read_integer("num_output", 0));
                    SA_TRY_ASSIGN(read.use_bias, parameters.read_bool("bias_term", true));
                    SA_TRY_ASSIGN(const std::int64_t axis, parameters.read_integer("axis", 1));
                    SA_TRY_ASSIGN(const bool transposed, parameters.read_bool("transpose", false));

                    SA_FAIL_IF(read.units < 1, StatusCode::InvalidDescription,
                               "{} needs a num_output of 1 or more; got {}",
                               describe_layer(layer), read.units);
                    SA_FAIL_IF(axis != 1, StatusCode::Unsupported,
                               "{} states axis {}; only the axis 1 is translated",
                               describe_layer(layer), axis);
                    SA_FAIL_IF(transposed, StatusCode::Unsupported,
                               "{} stores its weight transposed, which linear does not read",
                               describe_layer(layer));
                    return read;
                }

                /** @brief Appends the weight and bias tensors of the row @p row. */
                [[nodiscard]] Status store_weights(const Interchange::CaffeLayer& layer,
                                                   const InnerProduct& read,
                                                   const BlobValue& source, std::int64_t row,
                                                   TranslationProgress& progress)
                {
                    SA_TRY(expect_blobs(progress, layer, read.use_bias ? 2 : 1));
                    if (!progress.has_weights)
                        return Status();

                    const std::int64_t inputs = source.is_image
                                                    ? source.height * source.width * source.channels
                                                    : source.channels;
                    SA_TRY_ASSIGN(std::vector<float> weights,
                                  read_blob(layer, 0, static_cast<std::size_t>(read.units * inputs)));
                    if (source.is_image)
                    {
                        SA_TRY_ASSIGN(weights,
                                      Interchange::flatten_weights_to_channels_last(
                                          SushiRuntime::span<const float>(weights.data(),
                                                                          weights.size()),
                                          read.units, source.channels, source.height,
                                          source.width));
                    }
                    append_tensor(progress, entry_name(row, "weight"), Shape{read.units, inputs},
                                  weights);
                    if (read.use_bias)
                    {
                        SA_TRY_ASSIGN(const std::vector<float> bias,
                                      read_blob(layer, 1, static_cast<std::size_t>(read.units)));
                        append_tensor(progress, entry_name(row, "bias"), Shape{read.units}, bias);
                    }
                    return Status();
                }
            } // namespace

            Status translate_inner_product(const Interchange::CaffeLayer& layer, std::int64_t index,
                                           TranslationProgress& progress)
            {
                SA_TRY(expect_shape(layer, 1));
                SA_TRY_ASSIGN(const BlobValue source, read_bottom(progress, layer, 0));
                SA_TRY_ASSIGN(const InnerProduct read, read_inner_product(layer));

                std::int64_t input_row = source.row;
                if (source.is_image)
                {
                    const std::int64_t flatten_sources[] = {source.row};
                    input_row = append_row(progress, "flatten", make_attributes({}, {}),
                                           flatten_sources);
                }

                const std::int64_t sources[] = {input_row};
                const std::int64_t row = append_row(
                    progress, "linear",
                    make_attributes({"units", "use_bias"},
                                    {Config::Value::integer(read.units),
                                     Config::Value::boolean(read.use_bias)}),
                    sources);
                SA_TRY(store_weights(layer, read, source, row, progress));

                BlobValue result;
                result.row = row;
                result.channels = read.units;
                write_blob(progress, layer.tops.front(), result, index);
                return Status();
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
