/**************************************************************************/
/* execution.cpp                                                          */
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
 * @file execution.cpp
 * @brief Implements create_runtime, which turns what SushiRuntime reports into a Status through
 *        the seam.
 * @author Mustafa Garip
 */

#include <SushiAI/execution/execution.hpp>

#include <utility>

namespace SushiAI
{
    namespace Detail
    {
        Result<SushiRuntime::Runtime> create_runtime(
            const SushiRuntime::RuntimeConfig& config) noexcept
        {
            try
            {
                SushiRuntime::Result<SushiRuntime::Runtime> created =
                    SushiRuntime::Runtime::create(config);
                if (!created)
                    return status_of_error(created.error(), SeamPhase::Allocate);

                return std::move(*created);
            }
            catch (...)
            {
                return status_of_current_exception(SeamPhase::Allocate);
            }
        }
    } // namespace Detail
} // namespace SushiAI
