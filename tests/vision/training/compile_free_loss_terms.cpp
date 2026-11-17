/**************************************************************************/
/* compile_free_loss_terms.cpp                                            */
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
 * @file compile_free_loss_terms.cpp
 * @brief Checks at compile time that FreeLossTerms is plain data a host oracle can copy.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/vision/training/free_loss_terms.hpp>

static_assert(std::is_trivially_copyable<SushiAI::Vision::Training::FreeLossTerms>::value,
              "the terms of a step are copied between the oracle, the loop and the report");
static_assert(std::is_default_constructible<SushiAI::Vision::Training::FreeLossTerms>::value,
              "a loop starts its totals from a default value");
