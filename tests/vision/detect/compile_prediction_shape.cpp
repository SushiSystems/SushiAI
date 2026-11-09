/**************************************************************************/
/* compile_prediction_shape.cpp                                           */
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
 * @file compile_prediction_shape.cpp
 * @brief Checks at compile time the type and order of the prediction extents.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <type_traits>
#include <SushiAI/vision/detect/prediction_shape.hpp>

using SushiAI::Vision::Detect::PredictionShape;

namespace
{
    static_assert(std::is_aggregate_v<PredictionShape>,
                  "a head fills a prediction shape by braces");
    static_assert(std::is_same_v<decltype(PredictionShape::batch), std::int64_t>,
                  "the image count is a 64-bit extent");
    static_assert(std::is_same_v<decltype(PredictionShape::anchors), std::int64_t>,
                  "the anchor count is a 64-bit extent");
    static_assert(std::is_same_v<decltype(PredictionShape::num_classes), std::int64_t>,
                  "the class count is a 64-bit extent");
} // namespace
