/**************************************************************************/
/* compile_window_logits.cpp                                             */
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
 * @file compile_window_logits.cpp
 * @brief Checks at compile time that WindowLogits is a port and that the in-memory source
 *        implements it.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/language/generation/in_memory_window_logits.hpp>
#include <SushiAI/language/generation/window_logits.hpp>

static_assert(std::is_abstract<SushiAI::Language::Generation::WindowLogits>::value,
              "WindowLogits is a port: only its adapters are made");
static_assert(std::has_virtual_destructor<SushiAI::Language::Generation::WindowLogits>::value,
              "an adapter of WindowLogits is destroyed through the port");
static_assert(std::is_base_of<SushiAI::Language::Generation::WindowLogits,
                              SushiAI::Language::Generation::InMemoryWindowLogits>::value,
              "the in-memory source implements the port");
