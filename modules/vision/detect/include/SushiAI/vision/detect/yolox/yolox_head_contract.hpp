/**************************************************************************/
/* yolox_head_contract.hpp                                                */
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
 * @file yolox_head_contract.hpp
 * @brief Declares the port names and the objectness prior a YOLOX head and its decoding share.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace SushiAI
{
    namespace Vision::Detect::Yolox
    {
        /** @brief Names the role of a class-logit port. */
        inline constexpr std::string_view K_YOLOX_CLASSES = "classes";

        /** @brief Names the role of a box port: centre offset x and y, then log width and height. */
        inline constexpr std::string_view K_YOLOX_BOXES = "boxes";

        /** @brief Names the role of an objectness-logit port. */
        inline constexpr std::string_view K_YOLOX_OBJECTNESS = "objectness";

        /** @brief Holds the number of channels of a box port. */
        inline constexpr std::size_t K_YOLOX_BOX_CHANNELS = 4;

        /** @brief Holds the number of channels of an objectness port. */
        inline constexpr std::size_t K_YOLOX_OBJECTNESS_CHANNELS = 1;

        /** @brief Holds the probability the class and objectness biases start from. */
        inline constexpr double K_YOLOX_PRIOR_PROBABILITY = 0.01;

        /**
         * @brief Builds the name of a head port, "level<level>.<role>".
         * @param role K_YOLOX_CLASSES, K_YOLOX_BOXES or K_YOLOX_OBJECTNESS.
         */
        [[nodiscard]] std::string yolox_port_name(std::size_t level,
                                                  std::string_view role) noexcept;

        /** @brief Returns the logit of K_YOLOX_PRIOR_PROBABILITY, log(p / (1 - p)). */
        [[nodiscard]] double yolox_prior_bias() noexcept;

    } // namespace Vision::Detect::Yolox
} // namespace SushiAI
