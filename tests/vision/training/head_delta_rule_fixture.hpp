/**************************************************************************/
/* head_delta_rule_fixture.hpp                                            */
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
 * @file head_delta_rule_fixture.hpp
 * @brief Declares the two-head batch the head delta rule tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>

/** @brief Holds the two-head batch the tests of the head delta rules share. */
namespace SushiAI::HeadDeltaRuleTest
{
    /** @brief Holds the class count of both heads. */
    inline constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the channels of a head of two anchors: 2 * (5 + K_CLASSES). */
    inline constexpr std::int64_t K_CHANNELS = 16;

    /** @brief Holds the size of the network input in pixels, both sides. */
    inline constexpr std::int64_t K_NETWORK_SIDE = 64;

    /** @brief Holds the grid sides of the two heads: strides 16 and 32 over the input. */
    inline constexpr std::int64_t K_SIDES[2] = {4, 2};

    /**
     * @brief Holds a layout, two heads, their outputs and a truth, and views them as a batch.
     *
     * The layout has four anchors and two heads of two anchors each. The truth is one image of two
     * boxes, one of 16 pixels centred at (20, 20) and one of 24 by 16 pixels centred at (44, 40).
     * An output holds the sequence 0.8 sin(0.37 i + 0.1), a logit in (-0.8, 0.8). The object
     * holds views of itself, so it is neither copied nor moved.
     */
    class RuleBatch final
    {
        public:
            /**
             * @brief Builds the batch with every head of the size form @p size and the coding
             *        @p coding.
             */
            RuleBatch(Vision::Detect::Anchor::SizeForm size,
                      Vision::Detect::Anchor::HeadCoding coding =
                          Vision::Detect::Anchor::HeadCoding::Logit)
            {
                layout_.anchors = {{0.25f, 0.25f}, {0.5f, 0.375f}, {0.75f, 0.75f}, {1.0f, 0.5f}};
                layout_.head_anchors = {{0, 1}, {2, 3}};

                for (std::size_t level = 0; level < 2; ++level)
                {
                    Vision::Training::AnchorHead head;
                    head.output = "head." + std::to_string(level);
                    head.shape = Vision::Detect::Anchor::HeadShape{1, K_SIDES[level],
                                                                   K_SIDES[level]};
                    head.options.num_classes = K_CLASSES;
                    head.options.coding = coding;
                    head.options.size = size;
                    heads_.push_back(head);

                    const std::size_t count = static_cast<std::size_t>(
                        K_SIDES[level] * K_SIDES[level] * K_CHANNELS);
                    std::vector<float> values(count);
                    for (std::size_t index = 0; index < count; ++index)
                        values[index] = 0.8f * std::sin(0.37f * static_cast<float>(index) + 0.1f);
                    outputs_.push_back(std::move(values));
                }
                for (const std::vector<float>& values : outputs_)
                    views_.emplace_back(values.data(), values.size());

                truth_.batch = 1;
                truth_.max_boxes = 2;
                truth_.boxes = {12.0f, 12.0f, 28.0f, 28.0f, 32.0f, 32.0f, 56.0f, 48.0f};
                truth_.labels = {1, 2};
                truth_.valid = {1, 1};
            }

            RuleBatch(const RuleBatch&) = delete;
            RuleBatch& operator=(const RuleBatch&) = delete;
            RuleBatch(RuleBatch&&) = delete;
            RuleBatch& operator=(RuleBatch&&) = delete;

            /** @brief Returns the layout. */
            const Vision::Detect::Anchor::AnchorLayout& layout() const noexcept
            {
                return layout_;
            }

            /** @brief Returns the heads, which a case may change before it asks for a batch. */
            std::vector<Vision::Training::AnchorHead>& heads() noexcept
            {
                return heads_;
            }

            /** @brief Returns the truth, which a case may change before it asks for a batch. */
            Vision::Detect::GroundTruthBatch& truth() noexcept
            {
                return truth_;
            }

            /** @brief Returns the heads as the span a rule's check reads. */
            SushiRuntime::span<const Vision::Training::AnchorHead> head_span() const noexcept
            {
                return SushiRuntime::span<const Vision::Training::AnchorHead>(heads_.data(),
                                                                              heads_.size());
            }

            /** @brief Returns the output of head @p level. */
            SushiRuntime::span<const float> output(std::size_t level) const noexcept
            {
                return views_[level];
            }

            /** @brief Returns the batch a rule's compute reads, over the members of this object. */
            Vision::Training::HeadDeltaBatch batch() const noexcept
            {
                Vision::Training::HeadDeltaBatch described;
                described.network = Vision::Detect::Anchor::NetworkSize{K_NETWORK_SIDE,
                                                                         K_NETWORK_SIDE};
                described.layout = &layout_;
                described.heads = head_span();
                described.outputs = SushiRuntime::span<const SushiRuntime::span<const float>>(
                    views_.data(), views_.size());
                described.truth = &truth_;
                return described;
            }

        private:
            /** @brief Holds the anchors and the head masks. */
            Vision::Detect::Anchor::AnchorLayout layout_;

            /** @brief Holds the two heads. */
            std::vector<Vision::Training::AnchorHead> heads_;

            /** @brief Holds the output values of each head. */
            std::vector<std::vector<float>> outputs_;

            /** @brief Holds a view of each output. */
            std::vector<SushiRuntime::span<const float>> views_;

            /** @brief Holds the ground truth. */
            Vision::Detect::GroundTruthBatch truth_;
    };

    /** @brief Returns two empty deltas, one per head of a RuleBatch. */
    inline std::vector<Vision::Detect::Anchor::HeadDelta> make_deltas()
    {
        return std::vector<Vision::Detect::Anchor::HeadDelta>(2);
    }

    /** @brief Returns @p deltas as the span a rule's compute writes. */
    inline SushiRuntime::span<Vision::Detect::Anchor::HeadDelta> view_deltas(
        std::vector<Vision::Detect::Anchor::HeadDelta>& deltas) noexcept
    {
        return SushiRuntime::span<Vision::Detect::Anchor::HeadDelta>(deltas.data(),
                                                                     deltas.size());
    }
} // namespace SushiAI::HeadDeltaRuleTest
