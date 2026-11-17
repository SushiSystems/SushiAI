/**************************************************************************/
/* rule_gradient_support.hpp                                              */
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
 * @file rule_gradient_support.hpp
 * @brief Declares the fixture and the sample points the gradient tests of the backward
 *        rules share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiAI/engine_fixture/engine_test.hpp>

/** @brief Provides the engine every gradient case of the backward rules records on. */
class Medium_Autograd : public SushiAI::Test::EngineTest
{

};

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Returns the @p count positive values of a probe point: 0.5 to 1.7 in steps of 0.1.
         * @param seed Shifts the sequence so that two points of one test differ.
         *
         * Each value sits above 0.1, so the square root, the logarithm and the powers are smooth
         * over the whole step of a central difference.
         */
        inline std::vector<double> positive_point(std::size_t count, std::size_t seed)
        {
            constexpr std::size_t K_STEPS = 13;
            constexpr double K_FIRST = 0.5;
            constexpr double K_STEP = 0.1;
            constexpr std::size_t K_STRIDE = 7;

            std::vector<double> values(count);
            for (std::size_t i = 0; i < count; ++i)
                values[i] = K_FIRST + K_STEP * static_cast<double>((i * K_STRIDE + seed) % K_STEPS);
            return values;
        }
    } // namespace Test
} // namespace SushiAI
