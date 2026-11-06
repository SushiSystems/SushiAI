/**************************************************************************/
/* anchor_head.hpp                                                        */
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
 * @file anchor_head.hpp
 * @brief Declares AnchorHead, what an anchor objective knows of one head of the model.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <string>

#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds what the objective needs of one anchor head of the model. */
    struct AnchorHead
    {
        std::string output;                           /**< Name the model publishes it under. */
        Detect::Anchor::HeadShape shape;              /**< Extents of the published output. */
        Detect::Anchor::HeadOptions options;          /**< Classes, coding and offset scale. */
        Detect::Anchor::HeadTrainingOptions training; /**< Weights and thresholds of the delta. */
    };

} // namespace SushiAI::Vision::Training
