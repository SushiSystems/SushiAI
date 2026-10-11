/**************************************************************************/
/* head_delta.hpp                                                         */
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
 * @file head_delta.hpp
 * @brief Declares the training delta of an anchor head and the assignment it records.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the delta of one head and the truths its slots were assigned. */
        struct HeadDelta
        {
            std::vector<float> delta; /**< `[B, H, W, A * (5 + C)]`; training descends -delta. */
            /// @brief Holds `[B, H, W, A]`: the truth row a slot's best anchor took, else -1.
            std::vector<std::int64_t> assigned_row;
            /// @brief Holds `[B, H, W, A]`: the class of that truth, else -1.
            std::vector<std::int64_t> assigned_class;
            std::int64_t assignments = 0; /**< Slots a truth was assigned to, repeats counted. */
            double iou_sum = 0.0;         /**< Sum over assignments of the slot's IoU with it. */
        };

        /**
         * @brief Computes the delta of a head's `[B, H, W, A * (5 + C)]` output against truths.
         * @param network Size the truth's pixel corners are divided by; both sides positive.
         * @param result Receives the delta and the assignment; a failure leaves it untouched.
         * @return Success, or the failure of validate_head_truth on @p truth.
         * @pre validate_head and validate_head_training succeed; @p head holds its shape; the
         *      box loss has a delta under the head's coding. Breaking one of these is fatal.
         */
        [[nodiscard]] Status compute_head_delta(
            SushiRuntime::span<const float> head, const HeadShape& shape,
            const HeadOptions& options, const AnchorLayout& layout, std::int64_t head_index,
            const HeadTrainingOptions& training, const Detect::GroundTruthBatch& truth,
            NetworkSize network, HeadDelta& result);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
