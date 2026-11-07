/**************************************************************************/
/* runtime_backend.hpp                                                    */
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
 * @file runtime_backend.hpp
 * @brief Declares RuntimeBackend, the port of a backend that runs on a SushiRuntime.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/execution/runtime_access.hpp>

namespace SushiAI
{
    /**
     * @class RuntimeBackend
     * @brief An ExecutionBackend that also gives out the runtime its engine is built on.
     *
     * A class that needs both the engine and the runtime takes one RuntimeBackend, so the two can
     * never name different objects.
     */
    class RuntimeBackend : public ExecutionBackend, public RuntimeAccess
    {
        public:
            ~RuntimeBackend() override = default;
    };
} // namespace SushiAI
