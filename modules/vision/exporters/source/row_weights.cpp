/**************************************************************************/
/* row_weights.cpp                                                        */
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
 * @file row_weights.cpp
 * @brief Defines the readers of a row's weights that every exporter shares.
 * @author Mustafa Garip
 */

#include "row_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            std::string row_tensor_name(std::size_t row, std::string_view field)
            {
                return std::to_string(row) + "." + std::string(field);
            }

            const IO::ImportedTensor* find_weight(const IO::ImportedModel& model, std::size_t row,
                                                  std::string_view field) noexcept
            {
                const std::string name = row_tensor_name(row, field);
                for (const IO::ImportedTensor& tensor : model.weights)
                {
                    if (tensor.record.name == name)
                        return &tensor;
                }
                return nullptr;
            }

            Result<std::vector<float>> read_weight(const IO::ImportedModel& model,
                                                   std::size_t row, std::string_view field,
                                                   std::string_view float32_holder,
                                                   std::vector<std::int64_t>& extents)
            {
                const IO::ImportedTensor* tensor = find_weight(model, row, field);
                SA_FAIL_IF(tensor == nullptr, StatusCode::InvalidDescription,
                           "the weights hold no tensor '{}'", row_tensor_name(row, field));
                SA_FAIL_IF(tensor->record.dtype != DType::Float32, StatusCode::Unsupported,
                           "the weight '{}' is not float32, and {} float32", tensor->record.name,
                           float32_holder);
                const std::size_t count =
                    static_cast<std::size_t>(tensor->record.shape.num_elements());
                SA_FAIL_IF(tensor->payload.size() != count * sizeof(float),
                           StatusCode::InvalidDescription,
                           "the weight '{}' holds {} bytes for {} elements", tensor->record.name,
                           tensor->payload.size(), count);
                extents.clear();
                for (std::int32_t axis = 0; axis < tensor->record.shape.rank(); ++axis)
                    extents.push_back(tensor->record.shape[axis]);
                std::vector<float> values(count);
                if (count != 0)
                    std::memcpy(values.data(), tensor->payload.data(), tensor->payload.size());
                return values;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
