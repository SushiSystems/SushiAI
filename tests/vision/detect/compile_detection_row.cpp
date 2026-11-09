/**************************************************************************/
/* compile_detection_row.cpp                                              */
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
 * @file compile_detection_row.cpp
 * @brief Checks at compile time the column layout of a detection row.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/detect/detection_row.hpp>

using namespace SushiAI::Vision::Detect;

namespace
{
    static_assert(K_DETECTION_SCORE + 1 == K_DETECTION_CLASS,
                  "the class column follows the score column");
    static_assert(K_DETECTION_CLASS + 1 == K_DETECTION_VALUES,
                  "the class column is the last of a row");
    static_assert(K_DETECTION_SCORE > 3,
                  "the four corner columns come before the score");
} // namespace
