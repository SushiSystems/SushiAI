/**************************************************************************/
/* assignment_section.hpp                                                 */
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
 * @file assignment_section.hpp
 * @brief Declares the assignment rule an anchor task selects and its form in a task section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Names the rule that assigns truths to the slots of an anchor head. */
        enum class AssignmentRuleKind : std::uint8_t
        {
            BestShape,  /**< The centre cell; best anchor by shape_iou, more above a threshold. */
            ShapeRatio, /**< Every anchor within a side ratio, in the centre and near cells. */
        };

        /** @brief Holds the rule an anchor task selects and the settings that rule reads. */
        struct AssignmentChoice
        {
            AssignmentRuleKind rule = AssignmentRuleKind::BestShape; /**< The selected rule. */
            ShapeRatioOptions shape_ratio; /**< Read under ShapeRatio only. */
        };

        /**
         * @brief Reads the "assignment" member of a task section.
         * @param section The member, or null when the section leaves it out.
         * @param where Names the member in a refusal.
         * @return The choice; BestShape when @p section is null; or InvalidDescription naming
         *         the key that fails.
         */
        [[nodiscard]] Result<AssignmentChoice> read_assignment_section(
            const Config::Value* section, std::string_view where);

        /**
         * @brief Reports whether @p choice is written; BestShape is the default and left out.
         */
        [[nodiscard]] bool writes_assignment_section(const AssignmentChoice& choice) noexcept;

        /**
         * @brief Writes @p choice as the "assignment" member of a task section.
         * @return The member, each float as the double of equal value, or InvalidDescription
         *         naming the key that fails.
         */
        [[nodiscard]] Result<Config::Value> write_assignment_section(
            const AssignmentChoice& choice, std::string_view where);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
