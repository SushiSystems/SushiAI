/**************************************************************************/
/* ignore_index.hpp                                                       */
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
 * @file ignore_index.hpp
 * @brief Defines the index a mask holds, by default, where no class is known.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /** @brief Holds the value a pixel of no known class has unless the caller states another. */
    inline constexpr std::int32_t K_DEFAULT_IGNORE_INDEX = 255;

} // namespace SushiAI::Vision::Segment
