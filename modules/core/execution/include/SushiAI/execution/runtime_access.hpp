/**************************************************************************/
/* runtime_access.hpp                                                     */
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
 * @file runtime_access.hpp
 * @brief Declares RuntimeAccess, the port over the SushiRuntime a backend runs on.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiRuntime/sushiruntime.hpp>

namespace SushiAI
{
    /**
     * @class RuntimeAccess
     * @brief Gives the adapters that exist only on a runtime (regions, clusters, placement) the
     *        runtime they run on.
     *
     * A host backend has no runtime and does not implement this port, so a class that needs one
     * asks for a RuntimeAccess, not for a concrete backend.
     */
    class RuntimeAccess
    {
        public:
            virtual ~RuntimeAccess() = default;

            /** @brief Returns the runtime; it outlives the object that gives it out. */
            [[nodiscard]] virtual SushiRuntime::Runtime& runtime() noexcept = 0;
    };
} // namespace SushiAI
