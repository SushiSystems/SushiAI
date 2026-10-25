/**************************************************************************/
/* head_delta_rule_choice.hpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file head_delta_rule_choice.hpp
 * @brief Declares the function that builds the head delta rule an anchor task selects.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <memory>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Builds the rule @p choice names: Darknet's for BestShape, the shape-ratio rule
     *        with the choice's settings for ShapeRatio.
     * @return The rule, never null; or InvalidDescription for shape-ratio settings that
     *         validate_shape_ratio_options refuses.
     */
    [[nodiscard]] Result<std::shared_ptr<const HeadDeltaRule>> choose_head_delta_rule(
        const Detect::Anchor::AssignmentChoice& choice);

} // namespace SushiAI::Vision::Training
