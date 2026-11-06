/**************************************************************************/
/* run_progress.cpp                                                       */
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

/**
 * @file run_progress.cpp
 * @brief Implements the guarded call of a detection run's progress callback.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/run_progress.hpp>

#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Vision::Training
{
    Status notify_guarded(const std::function<void()>& notify) noexcept
    {
        if (!notify)
            return Status();

        return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute, [&]()
        {
            notify();
            return Status();
        });
    }

} // namespace SushiAI::Vision::Training
