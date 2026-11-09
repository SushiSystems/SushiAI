/**************************************************************************/
/* compile_vision.cpp                                                     */
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
 * @file compile_vision.cpp
 * @brief Checks at compile time that the vision umbrella header alone names the detect, segment
 *        and imaging units.
 * @author Mustafa Garip
 */

#include <type_traits>
#include <SushiAI/vision.hpp>

namespace
{
    static_assert(std::is_class_v<SushiAI::Vision::Detect::Anchor::AnchorSize>,
                  "the umbrella names the anchor size of detect");
    static_assert(std::is_class_v<SushiAI::Vision::Segment::ConfusionMatrix>,
                  "the umbrella names the confusion matrix of segment");
    static_assert(std::is_class_v<SushiAI::Vision::Imaging::Image>,
                  "the umbrella names the image of imaging");
} // namespace
