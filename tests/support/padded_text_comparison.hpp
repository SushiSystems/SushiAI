/**************************************************************************/
/* padded_text_comparison.hpp                                             */
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
 * @file padded_text_comparison.hpp
 * @brief Declares the helpers that compare a batch of padded texts with the same texts run alone.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <support/tolerance.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Returns @p count elements of @p values, starting at element @p from. */
        inline std::vector<double> leading(const std::vector<double>& values, std::size_t from,
                                           std::size_t count)
        {
            return std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(from),
                                       values.begin() + static_cast<std::ptrdiff_t>(from + count));
        }

        /** @brief Returns @p short_text, then @p padding, then @p full_text, in one batch. */
        inline std::vector<double> padded_batch(const std::vector<double>& short_text,
                                                const std::vector<double>& padding,
                                                const std::vector<double>& full_text)
        {
            std::vector<double> batch = short_text;
            batch.insert(batch.end(), padding.begin(), padding.end());
            batch.insert(batch.end(), full_text.begin(), full_text.end());
            return batch;
        }

        /**
         * @brief Expects every element of @p computed within the bound of @p reference.
         * @tparam Real The precision of the computation: float or double.
         * @param operations The rounded operations on the longest path to one element.
         * @param scale The sum of the absolute products the bound is taken against.
         */
        template<typename Real>
        void expect_close(const std::vector<double>& computed, const std::vector<double>& reference,
                          std::size_t operations, double scale)
        {
            ASSERT_EQ(computed.size(), reference.size());
            for (std::size_t i = 0; i < computed.size(); ++i)
            {
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<Real>(
                    computed[i], reference[i], scale, operations)))
                    << "at element " << i;
            }
        }

        /**
         * @brief Expects every element of @p computed within the normwise bound of @p reference:
         *        @p operations times the unit roundoff times the largest element of it.
         */
        template<typename Real>
        void expect_close_normwise(const std::vector<double>& computed,
                                   const std::vector<double>& reference, std::size_t operations)
        {
            double largest = 0.0;
            for (const double element : reference)
                largest = std::max(largest, std::fabs(element));

            expect_close<Real>(computed, reference, operations, largest);
        }
    } // namespace Test
} // namespace SushiAI
