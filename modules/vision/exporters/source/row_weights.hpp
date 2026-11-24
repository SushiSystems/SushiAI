/**************************************************************************/
/* row_weights.hpp                                                        */
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
 * @file row_weights.hpp
 * @brief Declares the readers of a row's weights that every exporter shares.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Returns the name `<row>.<field>` of a tensor the row owns. */
            [[nodiscard]] std::string row_tensor_name(std::size_t row, std::string_view field);

            /** @brief Returns the weight `<row>.<field>` of @p model, or null when it has none. */
            [[nodiscard]] const IO::ImportedTensor* find_weight(const IO::ImportedModel& model,
                                                                std::size_t row,
                                                                std::string_view field) noexcept;

            /**
             * @brief Returns the float32 weight `<row>.<field>` as its values and extents.
             * @param float32_holder The words a refusal puts before "float32", such as
             *        "a caffemodel holds".
             * @return InvalidDescription when @p model holds no such weight or its bytes do not
             *         match its shape; Unsupported when it is not float32.
             */
            [[nodiscard]] Result<std::vector<float>> read_weight(
                const IO::ImportedModel& model, std::size_t row, std::string_view field,
                std::string_view float32_holder, std::vector<std::int64_t>& extents);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
