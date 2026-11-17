/**************************************************************************/
/* kept_rows.hpp                                                          */
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
 * @file kept_rows.hpp
 * @brief Declares KeptRows, which rows of a batch a row mask keeps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Holds which rows of a batch a loss keeps.
         *
         * The default keeps every row, whatever their count, so an objective without a row mask
         * counts like one whose mask keeps all.
         */
        class KeptRows final
        {
            public:
                /** @brief Builds the selection that keeps every row. */
                KeptRows() noexcept = default;

                /**
                 * @brief Builds the selection of @p flags, one per row.
                 * @param flags 1 for a kept row and 0 for a dropped one.
                 */
                explicit KeptRows(std::vector<std::uint8_t> flags) noexcept
                    : flags_(std::move(flags)), is_all_(false)
                {

                }

                /** @brief Returns true when row @p row is kept. */
                [[nodiscard]] bool keeps(std::int64_t row) const noexcept
                {
                    return is_all_ || flags_[static_cast<std::size_t>(row)] != 0;
                }

                /** @brief Returns the rows the selection states; 0 for the one that keeps all. */
                [[nodiscard]] std::int64_t size() const noexcept
                {
                    return static_cast<std::int64_t>(flags_.size());
                }

                /** @brief Returns how many of the stated rows are kept. */
                [[nodiscard]] std::int64_t count() const noexcept
                {
                    std::int64_t kept = 0;
                    for (const std::uint8_t flag : flags_)
                        kept += flag != 0 ? 1 : 0;

                    return kept;
                }

            private:
                /// @brief Holds 1 for a kept row and 0 for a dropped one; empty keeps all.
                std::vector<std::uint8_t> flags_;

                /// @brief Whether every row is kept.
                bool is_all_ = true;
        };

    } // namespace Train
} // namespace SushiAI
