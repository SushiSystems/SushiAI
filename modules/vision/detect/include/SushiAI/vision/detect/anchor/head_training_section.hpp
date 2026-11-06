/**************************************************************************/
/* head_training_section.hpp                                              */
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
 * @file head_training_section.hpp
 * @brief Declares the form of an anchor head's training options as an objective section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the "type" an objective section states when it trains anchor heads. */
        inline constexpr std::string_view K_ANCHOR_OBJECTIVE_TYPE = "anchor";

        /**
         * @brief Reads the objective section of a training file as HeadTrainingOptions.
         * @param section The "objective" object, with its "type" or with the type taken out.
         * @return The options, a key left out keeping the default of its field, or
         *         InvalidDescription naming the key: a key the section does not have, a
         *         value of another kind, a string no enumerator has, another "type", or
         *         what validate_head_training refuses.
         */
        [[nodiscard]] Result<HeadTrainingOptions> read_head_training_section(
            const Config::Value& section) noexcept;

        /**
         * @brief Writes HeadTrainingOptions as the objective section of a training file.
         * @return The "objective" object with its "type" and every key, each float as the
         *         double of equal value, or InvalidDescription for what
         *         validate_head_training refuses.
         */
        [[nodiscard]] Result<Config::Value> write_head_training_section(
            const HeadTrainingOptions& options) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
