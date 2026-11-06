/**************************************************************************/
/* compile_sushiai_log_project.cpp                                        */
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
 * @file compile_sushiai_log_project.cpp
 * @brief Checks at compile time sushiai_log_project.hpp: the name and the level variable of SushiAI's log project.
 * @author Mustafa Garip
 */

#include <SushiAI/log/sushiai_log_project.hpp>

static_assert(SushiAI::K_SUSHIAI_LOG_PROJECT.name() == "sushiai", "project name");
static_assert(SushiAI::K_SUSHIAI_LOG_PROJECT.level_variable() == "SA_LOG_LEVEL",
              "level variable");
