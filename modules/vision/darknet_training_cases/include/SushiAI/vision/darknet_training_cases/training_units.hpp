/**************************************************************************/
/* training_units.hpp                                                     */
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
 * @file training_units.hpp
 * @brief Declares the roundoff helper and the overfit share the Darknet training suites share.
 * @author Mustafa Garip
 */

#pragma once

#include <support/tolerance.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Returns @p count times the float unit roundoff. */
        constexpr double units(double count)
        {
            return count * TestSupport::unit_roundoff<float>();
        }

        /**
         * @brief Holds the share of its first cost that the cost of 200 steps on one batch
         *        must fall under: a tenth, the learning check the training suites name.
         */
        inline constexpr double K_OVERFIT_COST_SHARE = 0.1;

    } // namespace Test
} // namespace SushiAI
