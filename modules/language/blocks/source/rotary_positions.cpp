/**************************************************************************/
/* rotary_positions.cpp                                                   */
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
 * @file rotary_positions.cpp
 * @brief Implements the angles of rotary positions.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/rotary_positions.hpp>

#include <cmath>
#include <cstddef>
#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        double rotary_angle(std::int64_t position, std::int64_t pair,
                            std::int64_t pairs) noexcept
        {
            const double exponent = -static_cast<double>(pair) / static_cast<double>(pairs);
            return static_cast<double>(position) * std::pow(K_ROTARY_BASE, exponent);
        }

        Status check_rotary_row_spans(std::size_t cosines, std::size_t sines) noexcept
        {
            SA_FAIL_IF(cosines == 0 || cosines != sines, StatusCode::InvalidDescription,
                       "a rotary row was given {} cosines and {} sines; it writes one of each "
                       "per pair",
                       cosines, sines);
            return Status();
        }

        void write_rotary_row(std::int64_t position, SushiRuntime::span<float> cosines,
                              SushiRuntime::span<float> sines) noexcept
        {
            const Status checked = check_rotary_row_spans(cosines.size(), sines.size());
            SA_FATAL_IF(!checked.is_ok(), "{}", checked.message());

            const std::int64_t pairs = static_cast<std::int64_t>(cosines.size());
            for (std::int64_t pair = 0; pair < pairs; ++pair)
            {
                const double angle = rotary_angle(position, pair, pairs);
                cosines[static_cast<std::size_t>(pair)] = static_cast<float>(std::cos(angle));
                sines[static_cast<std::size_t>(pair)] = static_cast<float>(std::sin(angle));
            }
        }

    } // namespace Language::Blocks
} // namespace SushiAI
