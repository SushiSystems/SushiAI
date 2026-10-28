/**************************************************************************/
/* shape_ratio_delta.hpp                                                  */
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
 * @file shape_ratio_delta.hpp
 * @brief Declares the loss and delta of one anchor head under the shape-ratio assignment.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the most levels the objectness balance has a weight for. */
        inline constexpr std::int64_t K_MAX_BALANCED_LEVELS = 5;

        /** @brief Holds what the loss of one head level reads. */
        struct ShapeRatioDeltaInput
        {
            SushiRuntime::span<const float> head; /**< `[B, H, W, A * (5 + C)]` logits. */
            HeadShape shape;                      /**< Images and grid of the head. */
            HeadOptions options;                  /**< Class count and offset stretch. */
            const AnchorLayout* layout = nullptr; /**< Anchors and every head's mask. */
            std::int64_t head_index = 0;          /**< Level of the layout the head is. */
            const HeadTrainingOptions* training = nullptr; /**< The three base gains. */
            NetworkSize network;                           /**< Input size in pixels. */
            /// @brief Holds the level's assignments, as AssignmentRule::assign orders them.
            SushiRuntime::span<const TargetAssignment> assignments;
        };

        /** @brief Holds the factors that turn a level's three means into its share of the loss. */
        struct ShapeRatioFactors
        {
            double box = 0.0;        /**< Box gain times 3 over the level count. */
            double objectness = 0.0; /**< Objectness gain, input area and level balance. */
            double classes = 0.0;    /**< Class gain times C over 80 and 3 over the levels. */
        };

        /** @brief Holds the three means of one level and the share of the loss they give. */
        struct ShapeRatioLoss
        {
            double box = 0.0;        /**< Mean of 1 - complete IoU over the assignments. */
            double objectness = 0.0; /**< Mean objectness cross-entropy over every slot. */
            double classes = 0.0;    /**< Mean class cross-entropy; 0 for one class. */
            double total = 0.0;      /**< Factored sum of the three, times the batch. */
        };

        /**
         * @brief Returns the objectness balance of level @p level among @p levels.
         * @return 4, 1, 0.4 for three levels; 4, 1, 0.25, 0.06, 0.02 by level otherwise.
         * @pre 0 <= @p level < @p levels <= K_MAX_BALANCED_LEVELS; breaking it is fatal.
         */
        [[nodiscard]] double objectness_balance(std::int64_t levels, std::int64_t level) noexcept;

        /**
         * @brief Returns the factors of level @p input.head_index under @p input.training.
         * @pre validate_shape_ratio_delta(@p input) succeeds.
         */
        [[nodiscard]] ShapeRatioFactors derive_shape_ratio_factors(
            const ShapeRatioDeltaInput& input) noexcept;

        /**
         * @brief Checks that @p input describes a head and assignments the loss can read.
         * @return Success, the failure of validate_head or validate_head_training, or
         *         InvalidDescription for a null pointer, a coding other than Logit, a
         *         size form other than SquaredSigmoid, a head span of the wrong size or
         *         holding a NaN or infinity, a level count above K_MAX_BALANCED_LEVELS, or
         *         an assignment outside the head or sharing a slot with another.
         */
        [[nodiscard]] Status validate_shape_ratio_delta(const ShapeRatioDeltaInput& input) noexcept;

        /**
         * @brief Writes the objectness target of every slot: the assigned slot's complete IoU
         *        with its truth, clamped at 0, and 0 elsewhere.
         * @param targets Replaced by `[B, H, W, A]`; a later record for the same slot wins.
         * @return Success, or the failure of validate_shape_ratio_delta.
         */
        [[nodiscard]] Status build_objectness_targets(const ShapeRatioDeltaInput& input,
                                                      std::vector<float>& targets) noexcept;

        /**
         * @brief Measures the level's loss against fixed objectness targets.
         * @param targets `[B, H, W, A]`, as build_objectness_targets writes them.
         * @return The loss, or the failure of validate_shape_ratio_delta, or
         *         InvalidDescription for targets of the wrong size.
         */
        [[nodiscard]] Result<ShapeRatioLoss> measure_shape_ratio_loss(
            const ShapeRatioDeltaInput& input, SushiRuntime::span<const float> targets) noexcept;

        /**
         * @brief Writes the delta of the level, `-B` times the derivative of its loss by each
         *        logit, with the objectness targets held fixed.
         * @param result Receives the delta, the assigned rows and classes, the assignment
         *        count and the sum of each assigned slot's IoU; a failure leaves it untouched.
         * @return Success, or the failure of validate_shape_ratio_delta.
         */
        [[nodiscard]] Status compute_shape_ratio_delta(const ShapeRatioDeltaInput& input,
                                                       HeadDelta& result) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
