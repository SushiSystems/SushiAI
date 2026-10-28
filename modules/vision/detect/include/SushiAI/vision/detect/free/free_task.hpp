/**************************************************************************/
/* free_task.hpp                                                          */
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
 * @file free_task.hpp
 * @brief Declares what an anchor-free model's graph cannot say and its form as a task section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <string_view>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /** @brief Holds the "type" a task section states when it describes a FreeTask. */
        inline constexpr std::string_view K_FREE_TASK_TYPE = "anchor_free_detection";

        /**
         * @brief Holds what decoding and training an anchor-free model needs beyond its graph.
         *
         * Strides, ports, reg_max and the class count come from the built model, never from
         * here. Of each AssignerConfig the section holds topk, topk2, alpha and beta only.
         */
        struct FreeTask
        {
            /** @brief Holds the class score a detection must exceed to be kept. */
            float score_threshold = 0.25f;

            /** @brief Holds the measure above which a box suppresses a lower-scoring one. */
            float suppression_threshold = 0.7f;

            /** @brief Selects the measure the suppression compares with its threshold. */
            SuppressionCriterion criterion = SuppressionCriterion::IoU;

            /** @brief Holds the exponent of the distance term, read by DistanceIoU only. */
            float distance_exponent = K_DEFAULT_DISTANCE_EXPONENT;

            /** @brief Holds the one-to-many assigner settings; the preset for one class. */
            AssignerConfig one_to_many = one_to_many_assigner_config(1, {});

            /** @brief Holds the one-to-one assigner settings; the preset for one class. */
            AssignerConfig one_to_one = one_to_one_assigner_config(1, {});
        };

        /**
         * @brief Reads the task section of a model file as a FreeTask.
         * @param task The "task" object, as Config::ModelSpec::task holds it.
         * @return The task, or InvalidDescription naming the key that fails. An absent
         *         "assignment" or branch keeps the FreeTask default.
         */
        [[nodiscard]] Result<FreeTask> read_free_task(const Config::Value& task) noexcept;

        /**
         * @brief Writes a FreeTask as the task section of a model file.
         * @return The "task" object with both branches written, each float as the double of
         *         equal value, or InvalidDescription naming the key that fails.
         */
        [[nodiscard]] Result<Config::Value> write_free_task(const FreeTask& task) noexcept;

    } // namespace Vision::Detect::Free
} // namespace SushiAI
