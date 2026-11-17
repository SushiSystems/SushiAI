/**************************************************************************/
/* caffe_op_input.cpp                                                     */
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
 * @file caffe_op_input.cpp
 * @brief Implements the model input a Caffe net declares, as an Input layer or at net level.
 * @author Mustafa Garip
 */

#include "caffe_operator_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <SushiAI/config/model_config.hpp>
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
                /** @brief Holds the extents of an image blob: batch, channels, height, width. */
                constexpr std::size_t K_IMAGE_RANK = 4;

                /** @brief Holds the extents of a vector blob: batch, features. */
                constexpr std::size_t K_VECTOR_RANK = 2;
            } // namespace

            Status declare_input(TranslationProgress& progress, const std::string& name,
                                 const std::vector<std::int64_t>& dimensions, std::int64_t writer)
            {
                SA_FAIL_IF(progress.has_input, StatusCode::Unsupported,
                           "the net declares a second input, '{}'; a model file has one input",
                           name);
                SA_FAIL_IF(dimensions.size() != K_IMAGE_RANK && dimensions.size() != K_VECTOR_RANK,
                           StatusCode::Unsupported,
                           "the input '{}' has {} extents; the importer reads [N, C, H, W] and "
                           "[N, K]",
                           name, dimensions.size());
                for (const std::int64_t extent : dimensions)
                {
                    SA_FAIL_IF(extent < 1, StatusCode::InvalidDescription,
                               "the input '{}' has the extent {}; every extent needs 1 or more",
                               name, extent);
                }

                BlobValue value;
                if (dimensions.size() == K_IMAGE_RANK)
                {
                    value.is_image = true;
                    value.channels = dimensions[1];
                    value.height = dimensions[2];
                    value.width = dimensions[3];
                    progress.spec.input_shape = Shape{value.height, value.width, value.channels};
                }
                else
                {
                    value.channels = dimensions[1];
                    progress.spec.input_shape = Shape{value.channels};
                }
                progress.has_input = true;
                write_blob(progress, name, value, writer);
                return Status();
            }

            Status translate_input(const Interchange::CaffeLayer& layer, std::int64_t index,
                                   TranslationProgress& progress)
            {
                SA_FAIL_IF(!layer.bottoms.empty() || layer.tops.size() != 1,
                           StatusCode::Unsupported,
                           "{} has {} bottoms and {} tops; the importer reads an Input layer "
                           "with one top",
                           describe_layer(layer), layer.bottoms.size(), layer.tops.size());
                const ParameterReader parameters(layer, "input_param");
                SA_TRY_ASSIGN(const std::vector<std::int64_t> dimensions,
                              parameters.read_nested_integers("shape", "dim"));
                return declare_input(progress, layer.tops.front(), dimensions, index);
            }
        } // namespace Detail::Caffe
    } // namespace Vision::Importers
} // namespace SushiAI
