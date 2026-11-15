/**************************************************************************/
/* ultralytics_tensors.cpp                                                */
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
 * @file ultralytics_tensors.cpp
 * @brief Defines the permutation of Ultralytics tensors into SushiAI's weight layouts.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/ultralytics_tensors.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/filter_layout.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the rank of a convolution weight. */
            constexpr std::int32_t K_FILTER_RANK = 4;

            /** @brief Returns whether @p name begins with @p prefix. */
            bool starts_with(const std::string& name, const std::string& prefix) noexcept
            {
                return name.compare(0, prefix.size(), prefix) == 0;
            }

            /** @brief Returns whether a dropped prefix of @p rules begins @p name. */
            bool is_dropped(const std::string& name, const TensorRules& rules) noexcept
            {
                for (const std::string& prefix : rules.dropped)
                {
                    if (starts_with(name, prefix))
                        return true;
                }
                return false;
            }

            /** @brief Returns @p name under the first rename of @p rules that matches it. */
            std::string renamed(const std::string& name, const TensorRules& rules)
            {
                for (const auto& rule : rules.renames)
                {
                    if (starts_with(name, rule.first))
                        return rule.second + name.substr(rule.first.size());
                }
                return name;
            }

            /** @brief Returns @p tensor as a record and its bytes in SushiAI's layout. */
            Result<IO::ImportedTensor> to_record(const UltralyticsTensor& tensor)
            {
                const Shape& shape = tensor.shape;
                std::vector<float> values;
                Shape laid_out = shape;
                if (shape.rank() != K_FILTER_RANK)
                {
                    values = tensor.values;
                }
                else if (tensor.groups == 1)
                {
                    const IO::FilterExtents extents{shape[0], shape[1], shape[2], shape[3]};
                    values.resize(tensor.values.size());
                    IO::copy_kernel_last(tensor.values.data(), extents, values.data());
                    laid_out = Shape{shape[0], shape[2], shape[3], shape[1]};
                }
                else
                {
                    SA_FAIL_IF(shape[1] != 1 || tensor.groups != shape[0], StatusCode::Unsupported,
                               "the weight '{}' belongs to a convolution of {} groups over {} "
                               "channels, and only depthwise grouping is imported",
                               tensor.name, tensor.groups, shape[0]);
                    const IO::FilterExtents extents{shape[0], shape[1], shape[2], shape[3]};
                    values.resize(tensor.values.size());
                    IO::copy_depthwise_kernel_last(tensor.values.data(), extents, values.data());
                    laid_out = Shape{shape[2], shape[3], shape[0]};
                }

                IO::ImportedTensor out;
                out.record.name = tensor.name;
                out.record.dtype = DType::Float32;
                out.record.shape = laid_out;
                out.payload.resize(values.size() * sizeof(float));
                std::memcpy(out.payload.data(), values.data(), out.payload.size());
                return out;
            }
        } // namespace

        Result<std::vector<IO::ImportedTensor>> list_ultralytics_tensors(
            SushiRuntime::span<const UltralyticsTensor> tensors, const TensorRules& rules) noexcept
        {
            std::vector<IO::ImportedTensor> out;
            out.reserve(tensors.size());
            for (const UltralyticsTensor& tensor : tensors)
            {
                if (is_dropped(tensor.name, rules))
                    continue;
                SA_TRY_ASSIGN(IO::ImportedTensor record, to_record(tensor));
                record.record.name = renamed(tensor.name, rules);
                out.push_back(std::move(record));
            }
            return out;
        }
    } // namespace Vision::Importers
} // namespace SushiAI
