/**************************************************************************/
/* yolox_head_contract.cpp                                                */
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
 * @file yolox_head_contract.cpp
 * @brief Implements the port names and the objectness prior a YOLOX head and its decoding share.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/yolox/yolox_head_contract.hpp>

#include <cmath>

namespace SushiAI
{
    namespace Vision::Detect::Yolox
    {
        std::string yolox_port_name(std::size_t level, std::string_view role) noexcept
        {
            std::string name("level");
            name.append(std::to_string(level)).append(".").append(role);

            return name;
        }

        double yolox_prior_bias() noexcept
        {
            return std::log(K_YOLOX_PRIOR_PROBABILITY / (1.0 - K_YOLOX_PRIOR_PROBABILITY));
        }
    } // namespace Vision::Detect::Yolox
} // namespace SushiAI
