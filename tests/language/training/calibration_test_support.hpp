/**************************************************************************/
/* calibration_test_support.hpp                                           */
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
 * @file calibration_test_support.hpp
 * @brief Declares the sets of logits the calibration tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

/** @brief Holds what the calibration tests of the language training module share. */
namespace SushiAI::Language::Training::Support
{
    /** @brief Holds the classes of every set below. */
    inline constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the rows of one group: the label of the first three is 0 and the last is 1. */
    inline constexpr std::size_t K_GROUP_ROWS = 4;

    /** @brief Holds a set of rows of logits and their labels. */
    struct LogitSet
    {
        /** @brief Holds [rows, classes] logits, row-major. */
        std::vector<float> logits;

        /** @brief Holds one class index per row. */
        std::vector<std::int32_t> labels;

        /** @brief Returns the logits as the span the calibration functions read. */
        SushiRuntime::span<const float> logits_span() const
        {
            return SushiRuntime::span<const float>(logits.data(), logits.size());
        }

        /** @brief Returns the labels as the span the calibration functions read. */
        SushiRuntime::span<const std::int32_t> labels_span() const
        {
            return SushiRuntime::span<const std::int32_t>(labels.data(), labels.size());
        }
    };

    /** @brief Returns ln 3, the margin at which a two-class row is right with probability 3/4. */
    inline double three_to_one_margin()
    {
        return std::log(3.0);
    }

    /**
     * @brief Returns groups of four rows of the logits [factor * ln 3, 0], labelled 0, 0, 0, 1.
     *
     * At factor 1 a group is calibrated exactly: three rows of four are right and the logits
     * say 3/4. A factor scales the logits and nothing else.
     */
    inline LogitSet overconfident_groups(double factor, std::size_t groups)
    {
        const float margin = static_cast<float>(factor * three_to_one_margin());
        LogitSet set;

        for (std::size_t group = 0; group < groups; ++group)
        {
            for (std::size_t row = 0; row < K_GROUP_ROWS; ++row)
            {
                set.logits.push_back(margin);
                set.logits.push_back(0.0f);
                set.labels.push_back(row == K_GROUP_ROWS - 1 ? 1 : 0);
            }
        }

        return set;
    }
} // namespace SushiAI::Language::Training::Support
