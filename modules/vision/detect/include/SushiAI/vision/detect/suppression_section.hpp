/**************************************************************************/
/* suppression_section.hpp                                                */
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
 * @file suppression_section.hpp
 * @brief Declares the "suppression" object of a task section, its reader and its writer.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Selects which members a task section's "suppression" object carries. */
        enum class SuppressionKeys
        {
            WithThreshold,    /**< The object carries "threshold"; the reader requires it. */
            WithoutThreshold, /**< The object carries no "threshold"; the reader refuses it. */
        };

        /** @brief Holds what a task section's "suppression" object states. */
        struct SuppressionSection
        {
            /** @brief Selects the measure the suppression compares with its threshold. */
            SuppressionCriterion criterion = SuppressionCriterion::IoU;

            /** @brief Holds the measure above which a box suppresses a lower-scoring one. */
            float threshold = 0.7f;

            /** @brief Holds the exponent of the distance term, read by DistanceIoU only. */
            float distance_exponent = K_DEFAULT_DISTANCE_EXPONENT;
        };

        /**
         * @brief Reads the "suppression" member of a task section.
         * @param task The "task" object.
         * @param where The name a refusal gives the task, "task".
         * @param keys Whether the object carries "threshold".
         * @return The section, or InvalidDescription naming the failing key: an unknown key or
         *         criterion, a threshold outside [0, 1], or an exponent the criterion ignores.
         */
        [[nodiscard]] Result<SuppressionSection> read_suppression_section(
            const Config::Value& task, std::string_view where,
            SuppressionKeys keys = SuppressionKeys::WithThreshold) noexcept;

        /**
         * @brief Writes @p section as the "suppression" object of a task section.
         * @param keys Whether the object carries "threshold"; without it the threshold is not
         *        written or checked.
         * @return The object, with the exponent only under a criterion that reads it, or
         *         InvalidDescription naming the member that fails.
         */
        [[nodiscard]] Result<Config::Value> write_suppression_section(
            const SuppressionSection& section, std::string_view where,
            SuppressionKeys keys = SuppressionKeys::WithThreshold) noexcept;

    } // namespace Vision::Detect
} // namespace SushiAI
