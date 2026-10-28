/**************************************************************************/
/* head_delta_rule_choice.cpp                                             */
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
 * @file head_delta_rule_choice.cpp
 * @brief Implements choose_head_delta_rule, the one place an assignment kind picks a rule.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/head_delta_rule_choice.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

namespace SushiAI::Vision::Training
{
    Result<std::shared_ptr<const HeadDeltaRule>> choose_head_delta_rule(
        const Detect::Anchor::AssignmentChoice& choice) noexcept
    {
        switch (choice.rule)
        {
            case Detect::Anchor::AssignmentRuleKind::BestShape:
                return ::SushiAI::Detail::guard_value(
                    ::SushiAI::Detail::SeamPhase::Allocate, []()
                {
                    return std::shared_ptr<const HeadDeltaRule>(
                        std::make_shared<BestShapeHeadDeltaRule>());
                });
            case Detect::Anchor::AssignmentRuleKind::ShapeRatio:
                SA_TRY(Detect::Anchor::validate_shape_ratio_options(choice.shape_ratio));
                return ::SushiAI::Detail::guard_value(
                    ::SushiAI::Detail::SeamPhase::Allocate, [&choice]()
                {
                    return std::shared_ptr<const HeadDeltaRule>(
                        std::make_shared<ShapeRatioHeadDeltaRule>(choice.shape_ratio));
                });
        }

        SA_FATAL("no head delta rule is defined for assignment {}",
                 static_cast<int>(choice.rule));
    }

} // namespace SushiAI::Vision::Training
