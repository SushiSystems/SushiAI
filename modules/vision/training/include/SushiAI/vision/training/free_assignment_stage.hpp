/**************************************************************************/
/* free_assignment_stage.hpp                                              */
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
 * @file free_assignment_stage.hpp
 * @brief Declares the host stage that assigns an anchor-free step's targets and fills its leaves.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/assigner.hpp>
#include <SushiAI/vision/detect/free/assignment.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/target_leaves.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Names the array a leaf of the anchor-free loss is cut from. */
    enum class FreeLeafKind : std::uint8_t
    {
        TargetScores,        /**< `[B, H, W, C]` soft class targets. */
        BoxMask,             /**< `[B, H, W, 4]` 1 on a foreground row. */
        NormalizedWeight,    /**< `[B, H, W, 1]` a row's score sum over the normaliser. */
        TargetBoxes,         /**< `[B, H, W, 4]` x1 y1 x2 y2 in grid units. */
        TargetDistances,     /**< `[B, H, W, 4]` left, top, right, bottom in grid units. */
        L1Scale,             /**< `[B, H, W, 4]` the factor of each L1 error. */
        SafeBoxes,           /**< `[B, H, W, 4]` K_SAFE_BOX in every row. */
        DistributionWeights, /**< `[B, H, W, 4 * reg_max]` DFL bin weights; reg_max above 1. */
        AnchorPoints,        /**< `[B, H, W, 2]` cell centres in grid units. */
        AnchorStrides,       /**< `[B, H, W, 4]` the level's stride. */
        TargetScoresSum,     /**< `[1]` the sum of all target scores, at least 1. */
    };

    /** @brief Names the copy of the head a stage assigns: one-to-many or one-to-one. */
    enum class FreeBranch : std::uint8_t
    {
        OneToMany, /**< The copy every model has; roles "free.<kind>...". */
        OneToOne,  /**< The copy of a model with one; roles "free.one_to_one.<kind>...". */
    };

    /** @brief Holds one leaf the anchor-free loss reads, as the step declares it. */
    struct FreeLeaf
    {
        /** @brief Holds the role and debug name, "free.<kind>.level<l>" or "free.<kind>". */
        std::string role;
        /** @brief Holds the float32 extents of the leaf. */
        Shape shape;
        /** @brief Holds the array the leaf is cut from. */
        FreeLeafKind kind = FreeLeafKind::TargetScores;
        /** @brief Holds the level the leaf covers; 0 for TargetScoresSum. */
        std::int64_t level = 0;

        /** @brief Holds the copy of the head the leaf serves. */
        FreeBranch branch = FreeBranch::OneToMany;
    };

    /** @brief Holds host views of one level's ports of one copy, NHWC as the head emits them. */
    struct FreeLevelViews
    {
        SushiRuntime::span<const float> boxes;  /**< `[B, H, W, 4 * reg_max]` box outputs. */
        SushiRuntime::span<const float> scores; /**< `[B, H, W, C]` class logits. */
    };

    /**
     * @brief Decodes the ports of one copy of the head, assigns the truth and fills the loss
     *        leaves.
     * @see docs/agent/2026_10_25_ULTRALYTICS_DETECTORS/PLAN.md, section 4.
     */
    class FreeAssignmentStage final
    {
        public:
            /**
             * @brief Builds the stage for @p facts and the settings of @p branch in @p task.
             * @param batch Images per step, B.
             * @return The stage, or InvalidDescription for a batch below 1, no level, a
             *         reg_max or class count below 1, fewer anchors than topk or topk2, or a
             *         one-to-one branch of a model without that copy.
             */
            [[nodiscard]] static Result<FreeAssignmentStage> make(
                const FreeModelFacts& facts, const Detect::Free::FreeTask& task,
                std::int64_t batch, FreeBranch branch = FreeBranch::OneToMany);

            /** @brief Returns every leaf, each kind over the levels in order, the sum last. */
            [[nodiscard]] const std::vector<FreeLeaf>& leaves() const noexcept;

            /**
             * @brief Joins the levels, decodes them, assigns @p truth and fills every leaf.
             * @param levels One entry per level of the facts, finest first.
             * @param truth Pixel corners of @p batch images.
             * @return InvalidDescription if a view or the truth does not match the facts and the
             *         batch, or if a box output, a score or a valid truth box holds a NaN or an
             *         infinity; the message names the level and the first bad index.
             */
            [[nodiscard]] Status assign(SushiRuntime::span<const FreeLevelViews> levels,
                                        const Detect::GroundTruthBatch& truth);

            /**
             * @brief Copies leaf @p leaf of the last assign() into @p values.
             * @param values Receives as many values as the leaf's shape holds.
             * @return InvalidDescription if @p values has another size.
             *         An out-of-range @p leaf stops the process.
             */
            [[nodiscard]] Status write_leaf(std::size_t leaf,
                                            SushiRuntime::span<float> values) const;

            /** @brief Returns the assignment of the last assign(), over the joined anchors. */
            [[nodiscard]] const Detect::Free::Assignment& assignment() const noexcept;

            /** @brief Returns the target leaves of the last assign(), over the joined anchors. */
            [[nodiscard]] const Detect::Free::TargetLeaves& targets() const noexcept;

            /** @brief Returns the `[B, A, 4 * reg_max]` DFL bin weights; empty at reg_max 1. */
            [[nodiscard]] const std::vector<float>& distribution_weights() const noexcept;

        private:
            /** @brief Stores checked extents and builds the anchors and the leaf list. */
            FreeAssignmentStage(const FreeModelFacts& facts,
                                const Detect::Free::AssignerConfig& config, std::int64_t batch,
                                FreeBranch branch);

            /** @brief Writes the joined logits and scores of @p levels after checking them. */
            [[nodiscard]] Status join_levels(SushiRuntime::span<const FreeLevelViews> levels);

            /** @brief Refuses @p truth unless every box value of a valid slot is finite. */
            [[nodiscard]] Status refuse_non_finite_truth(
                const Detect::GroundTruthBatch& truth) const;

            /** @brief Writes the predicted pixel boxes and the score probabilities. */
            [[nodiscard]] Status decode_predictions();

            /** @brief Returns the joined array and its channels for a per-level @p kind. */
            [[nodiscard]] SushiRuntime::span<const float> joined_source(
                FreeLeafKind kind, std::int64_t& channels) const;

            /// @brief Holds the distribution bins per box side.
            std::int64_t reg_max_ = 0;
            /// @brief Holds the classes, C.
            std::int64_t classes_ = 0;
            /// @brief Holds the images per step, B.
            std::int64_t batch_ = 0;
            /// @brief Holds the anchors over all levels, A.
            std::int64_t anchors_ = 0;
            /// @brief Holds the input height in pixels, the finest level's height times stride.
            float image_height_ = 0.0f;
            /// @brief Holds the input width in pixels, the finest level's width times stride.
            float image_width_ = 0.0f;
            /// @brief Holds the extents and stride of each level, finest first.
            std::vector<Detect::Free::FeatureLevel> levels_;
            /// @brief Holds the index of each level's first anchor.
            std::vector<std::int64_t> offsets_;
            /// @brief Holds the assigner of the branch.
            Detect::Free::TaskAlignedAssigner assigner_;
            /// @brief Holds `[A, 2]` anchor points in grid units.
            std::vector<float> points_;
            /// @brief Holds `[A, 2]` anchor points in pixels.
            std::vector<float> pixel_points_;
            /// @brief Holds `[A]` anchor strides.
            std::vector<float> strides_;
            /// @brief Holds the anchor points and strides as batch leaves.
            Detect::Free::AnchorLeaves anchor_leaves_;
            /// @brief Holds `[B, A, 4 * reg_max]` joined box outputs.
            std::vector<float> box_outputs_;
            /// @brief Holds `[B, A, 4]` predicted x1 y1 x2 y2 in pixels.
            std::vector<float> predicted_boxes_;
            /// @brief Holds `[B, A, C]` joined logits, then their sigmoid.
            std::vector<float> probabilities_;
            /// @brief Holds the assignment of the last assign().
            Detect::Free::Assignment assignment_;
            /// @brief Holds the target leaves of the last assign().
            Detect::Free::TargetLeaves targets_;
            /// @brief Holds the DFL bin weights of the last assign(); empty at reg_max 1.
            std::vector<float> distribution_weights_;
            /// @brief Holds every leaf the step declares.
            std::vector<FreeLeaf> leaves_;
    };
} // namespace SushiAI::Vision::Training
