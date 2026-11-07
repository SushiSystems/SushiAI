/**************************************************************************/
/* run_state.cpp                                                          */
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
 * @file run_state.cpp
 * @brief Implements the checks of the run state an optimizer restores.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/run_state.hpp>

#include <cstddef>
#include <variant>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief Counts the entries of @p scalars called @p name. */
            [[nodiscard]] std::size_t count_scalars(SushiRuntime::span<const StateScalar> scalars,
                                                    std::string_view name) noexcept
            {
                std::size_t count = 0;
                for (const StateScalar& scalar : scalars)
                    count += scalar.name == name ? 1 : 0;

                return count;
            }
        } // namespace

        const StateScalar* find_scalar(SushiRuntime::span<const StateScalar> scalars,
                                       std::string_view name) noexcept
        {
            for (const StateScalar& scalar : scalars)
            {
                if (scalar.name == name)
                    return &scalar;
            }

            return nullptr;
        }

        Status check_state_owner(std::string_view owner, std::string_view type) noexcept
        {
            SA_FAIL_IF(owner != type, StatusCode::Incompatible,
                       "the run state was written by a '{}' optimizer, and this optimizer is "
                       "'{}'", owner, type);
            return Status();
        }

        Status check_scalar_list(SushiRuntime::span<const StateScalar> kept,
                                 SushiRuntime::span<const StateScalar> given,
                                 std::string_view type) noexcept
        {
            for (const StateScalar& scalar : given)
            {
                const StateScalar* const expected = find_scalar(kept, scalar.name);
                SA_FAIL_IF(expected == nullptr, StatusCode::Incompatible,
                           "the run state holds the scalar '{}', which this '{}' optimizer "
                           "does not keep", scalar.name, type);
                SA_FAIL_IF(count_scalars(given, scalar.name) != 1, StatusCode::Incompatible,
                           "the run state holds the scalar '{}' of this '{}' optimizer more "
                           "than once", scalar.name, type);
                SA_FAIL_IF(expected->value.index() != scalar.value.index(),
                           StatusCode::Incompatible,
                           "the run state holds the scalar '{}' of this '{}' optimizer as {}",
                           scalar.name, type,
                           std::holds_alternative<double>(scalar.value)
                               ? "a real number where a count is kept"
                               : "a count where a real number is kept");
            }

            for (const StateScalar& expected : kept)
            {
                SA_FAIL_IF(find_scalar(given, expected.name) == nullptr,
                           StatusCode::Incompatible,
                           "the run state does not hold the scalar '{}' of this '{}' optimizer",
                           expected.name, type);
            }

            return Status();
        }

    } // namespace Optim
} // namespace SushiAI
