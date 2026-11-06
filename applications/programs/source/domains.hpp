/**************************************************************************/
/* domains.hpp                                                            */
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
 * @file domains.hpp
 * @brief Declares the functions that add the rows of each linked domain to the app registries.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>

#include "registries.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @brief Adds the node, dataset, objective, run, report and importer rows of vision,
         *        or nothing in a build that does not link it.
         * @param registries The registries the rows are added to.
         * @return Ok, or the first failure a register function returned.
         */
        [[nodiscard]] Status register_vision_domain(Registries& registries);

        /**
         * @brief Adds the node, dataset, objective and report rows of language, or nothing
         *        in a build that does not link it.
         * @param registries The registries the rows are added to.
         * @return Ok, or the first failure a register function returned.
         */
        [[nodiscard]] Status register_language_domain(Registries& registries);

    } // namespace Apps
} // namespace SushiAI
