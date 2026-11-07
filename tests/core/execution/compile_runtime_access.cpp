/**************************************************************************/
/* compile_runtime_access.cpp                                             */
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
 * @file compile_runtime_access.cpp
 * @brief Checks at compile time that RuntimeAccess is a port, and that the runtime backend and
 *        no host backend implements it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/execution/runtime_access.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>

static_assert(std::is_abstract<SushiAI::RuntimeAccess>::value,
              "RuntimeAccess is a port: only its adapters are made");
static_assert(std::has_virtual_destructor<SushiAI::RuntimeAccess>::value,
              "an adapter of RuntimeAccess is destroyed through the port");
static_assert(std::is_base_of<SushiAI::RuntimeAccess, SushiAI::RuntimeExecutionBackend>::value,
              "the runtime backend gives out its runtime");
static_assert(!std::is_base_of<SushiAI::RuntimeAccess, SushiAI::HostExecutionBackend>::value,
              "a host backend has no runtime to give out");
