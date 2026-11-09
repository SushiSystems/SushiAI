/**************************************************************************/
/* compile_anchor_head.cpp                                                */
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
 * @file compile_anchor_head.cpp
 * @brief Checks at compile time that an anchor head copies and moves with its output name.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/vision/training/anchor_head.hpp>


static_assert(std::is_copy_constructible<SushiAI::Vision::Training::AnchorHead>::value,
              "an anchor objective keeps a copy of every head it was made for");
static_assert(std::is_nothrow_move_constructible<SushiAI::Vision::Training::AnchorHead>::value,
              "a list of heads grows without copying the output names");
