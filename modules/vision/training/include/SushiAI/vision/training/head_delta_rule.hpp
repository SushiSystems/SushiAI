/**************************************************************************/
/* head_delta_rule.hpp                                                    */
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
 * @file head_delta_rule.hpp
 * @brief Declares HeadDeltaRule, the rule that turns a batch's heads into deltas and a cost.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#pragma once

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds what a rule reads to compute the deltas of every head of one batch. */
    struct HeadDeltaBatch
    {
        Detect::Anchor::NetworkSize network;                 /**< Input size in pixels. */
        const Detect::Anchor::AnchorLayout* layout = nullptr; /**< Anchors and head masks. */
        SushiRuntime::span<const AnchorHead> heads;           /**< Heads, in layout order. */
        /// @brief Holds each head's `[B, H, W, A * (5 + C)]` output, in head order.
        SushiRuntime::span<const SushiRuntime::span<const float>> outputs;
        const Detect::GroundTruthBatch* truth = nullptr; /**< Truth rows of the batch. */
    };

    /** @brief Computes the delta of every anchor head of a batch and the cost it reports. */
    class HeadDeltaRule
    {
        public:
            /** @brief Destroys the rule. */
            virtual ~HeadDeltaRule() = default;

            /**
             * @brief Checks that the rule can train @p heads under @p layout.
             * @return Success, or InvalidDescription naming the head and what it breaks.
             */
            [[nodiscard]] virtual Status check(
                const Detect::Anchor::AnchorLayout& layout,
                SushiRuntime::span<const AnchorHead> heads) const = 0;

            /**
             * @brief Writes the delta of every head of @p batch and returns the batch's cost.
             * @param deltas One per head, in head order; each is replaced.
             * @return The cost summed over the heads, or the refusal of the truth.
             * @pre check() succeeded for the batch's layout and heads; outputs and deltas hold
             *      one entry per head; layout and truth are not null.
             */
            [[nodiscard]] virtual Result<Detect::Anchor::HeadCost> compute(
                const HeadDeltaBatch& batch,
                SushiRuntime::span<Detect::Anchor::HeadDelta> deltas) const = 0;

        protected:
            /** @brief Constructs the base of a rule. */
            HeadDeltaRule() = default;

            /** @brief Copies the base of a rule. */
            HeadDeltaRule(const HeadDeltaRule&) = default;

            /** @brief Moves the base of a rule. */
            HeadDeltaRule(HeadDeltaRule&&) = default;

            /** @brief Copy-assigns the base of a rule. */
            HeadDeltaRule& operator=(const HeadDeltaRule&) = default;

            /** @brief Move-assigns the base of a rule. */
            HeadDeltaRule& operator=(HeadDeltaRule&&) = default;
    };

} // namespace SushiAI::Vision::Training
