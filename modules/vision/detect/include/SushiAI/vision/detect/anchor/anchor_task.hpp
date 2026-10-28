/**************************************************************************/
/* anchor_task.hpp                                                        */
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
 * @file anchor_task.hpp
 * @brief Declares what decoding an anchor-box model needs and its form as a task section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the "type" a task section states when it describes an AnchorTask. */
        inline constexpr std::string_view K_ANCHOR_TASK_TYPE = "anchor_detection";

        /** @brief Holds which model output one head reads and how that output is decoded. */
        struct AnchorTaskHead
        {
            std::string output;  /**< Published name of the model output the head reads. */
            HeadOptions options; /**< Class count, coding and offset stretch of the head. */
        };

        /** @brief Holds what decoding the heads of an anchor-box model needs. */
        struct AnchorTask
        {
            /** @brief Holds the class count of the model; every head carries the same count. */
            std::int64_t num_classes = 0;

            /** @brief Holds the anchors as fractions of the input and each head's mask. */
            AnchorLayout layout;

            /** @brief Holds the heads; entry k belongs to head k of layout. */
            std::vector<AnchorTaskHead> heads;

            /** @brief Selects the measure the suppression compares with its threshold. */
            SuppressionCriterion criterion = SuppressionCriterion::IoU;

            /** @brief Holds the exponent of the distance term, read by DistanceIoU only. */
            float distance_exponent = K_DEFAULT_DISTANCE_EXPONENT;

            /** @brief Selects the rule that assigns truths to slots in training. */
            AssignmentChoice assignment;
        };

        /**
         * @brief Reads the task section of a model file as an AnchorTask.
         * @param task The "task" object, as Config::ModelSpec::task holds it.
         * @return The task, or InvalidDescription naming the key and the entry that fails.
         *         Under a criterion that does not read it, distance_exponent is the default.
         */
        [[nodiscard]] Result<AnchorTask> read_anchor_task(const Config::Value& task) noexcept;

        /**
         * @brief Writes an AnchorTask as the task section of a model file.
         * @return The "task" object, each float as the double of equal value, or
         *         InvalidDescription naming the key and the entry that fails. The exponent is
         *         written for a criterion that reads it and left out otherwise.
         */
        [[nodiscard]] Result<Config::Value> write_anchor_task(const AnchorTask& task) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
