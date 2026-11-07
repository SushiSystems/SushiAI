/**************************************************************************/
/* compile_scope.cpp                                                         */
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
 * @file compile_scope.cpp
 * @brief Checks at compile time that a Scope guard can be neither copied nor moved.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/eager/scope.hpp>

static_assert(!std::is_copy_constructible<SushiAI::Eager::Scope>::value,
              "a Scope guard closes its scope once, so it cannot be copied");
static_assert(!std::is_move_constructible<SushiAI::Eager::Scope>::value,
              "a Scope guard closes its scope once, so it cannot be moved");
static_assert(!std::is_copy_assignable<SushiAI::Eager::Scope>::value,
              "a Scope guard closes its scope once, so it cannot be copy-assigned");
static_assert(!std::is_move_assignable<SushiAI::Eager::Scope>::value,
              "a Scope guard closes its scope once, so it cannot be move-assigned");
