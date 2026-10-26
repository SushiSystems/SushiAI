/**************************************************************************/
/* free_assignment_stage.cpp                                              */
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
 * @file free_assignment_stage.cpp
 * @brief Implements the host stage that assigns an anchor-free step's targets.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_assignment_stage.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>
#include <utility>

#include <SushiAI/failure/error.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/float_order.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        using Detect::Free::FreeTask;
        using Detect::Free::FeatureLevel;

        /** @brief Holds the per-level kinds in the order the leaf list gives them. */
        constexpr std::array<FreeLeafKind, 10> K_LEVEL_KINDS = {
            FreeLeafKind::TargetScores,     FreeLeafKind::BoxMask,
            FreeLeafKind::NormalizedWeight, FreeLeafKind::TargetBoxes,
            FreeLeafKind::TargetDistances,  FreeLeafKind::L1Scale,
            FreeLeafKind::SafeBoxes,        FreeLeafKind::DistributionWeights,
            FreeLeafKind::AnchorPoints,     FreeLeafKind::AnchorStrides};

        /** @brief Returns the word a leaf role names @p kind by. */
        std::string_view kind_name(FreeLeafKind kind) noexcept
        {
            switch (kind)
            {
                case FreeLeafKind::TargetScores:
                    return "target_scores";
                case FreeLeafKind::BoxMask:
                    return "box_mask";
                case FreeLeafKind::NormalizedWeight:
                    return "normalized_weight";
                case FreeLeafKind::TargetBoxes:
                    return "target_boxes";
                case FreeLeafKind::TargetDistances:
                    return "target_distances";
                case FreeLeafKind::L1Scale:
                    return "l1_scale";
                case FreeLeafKind::SafeBoxes:
                    return "safe_boxes";
                case FreeLeafKind::DistributionWeights:
                    return "distribution_weights";
                case FreeLeafKind::AnchorPoints:
                    return "anchor_points";
                case FreeLeafKind::AnchorStrides:
                    return "anchor_strides";
                case FreeLeafKind::TargetScoresSum:
                    return "target_scores_sum";
            }
            return "unknown";
        }

        /** @brief Returns the channels of a per-level leaf of @p kind. */
        std::int64_t kind_channels(FreeLeafKind kind, std::int64_t classes,
                                   std::int64_t reg_max) noexcept
        {
            switch (kind)
            {
                case FreeLeafKind::TargetScores:
                    return classes;
                case FreeLeafKind::NormalizedWeight:
                    return 1;
                case FreeLeafKind::DistributionWeights:
                    return static_cast<std::int64_t>(Detect::K_BOX_VALUES) * reg_max;
                case FreeLeafKind::AnchorPoints:
                    return static_cast<std::int64_t>(Detect::Free::K_POINT_VALUES);
                default:
                    return static_cast<std::int64_t>(Detect::K_BOX_VALUES);
            }
        }

        /** @brief Returns "free." or "free.one_to_one.", the start of every role of @p branch. */
        std::string role_prefix(FreeBranch branch)
        {
            return branch == FreeBranch::OneToOne ? "free.one_to_one." : "free.";
        }

        /** @brief Returns "<prefix><kind>.level<l>". */
        std::string level_role(FreeBranch branch, FreeLeafKind kind, std::size_t level)
        {
            std::string role = role_prefix(branch);
            role.append(kind_name(kind)).append(".level").append(std::to_string(level));
            return role;
        }

        /** @brief Returns the logistic sigmoid of @p logit. */
        float sigmoid(float logit) noexcept
        {
            return 1.0f / (1.0f + std::exp(-logit));
        }

        /** @brief Returns @p data as a span of its values. */
        SushiRuntime::span<const float> view(const std::vector<float>& data) noexcept
        {
            return SushiRuntime::span<const float>(data.data(), data.size());
        }

        /** @brief Returns @p data as a writable span of its values. */
        SushiRuntime::span<float> view(std::vector<float>& data) noexcept
        {
            return SushiRuntime::span<float>(data.data(), data.size());
        }
    } // namespace

    Result<FreeAssignmentStage> FreeAssignmentStage::make(const FreeModelFacts& facts,
                                                          const FreeTask& task,
                                                          std::int64_t batch, FreeBranch branch)
    {
        SA_FAIL_IF(branch == FreeBranch::OneToOne && !facts.has_one_to_one,
                   StatusCode::InvalidDescription,
                   "a one-to-one stage needs a model that publishes a one-to-one copy");
        SA_FAIL_IF(batch < 1, StatusCode::InvalidDescription,
                   "an anchor-free stage needs a batch of at least 1; got {}", batch);
        SA_FAIL_IF(facts.levels.empty(), StatusCode::InvalidDescription,
                   "an anchor-free stage needs at least one level");
        SA_FAIL_IF(facts.reg_max < 1 || facts.classes < 1, StatusCode::InvalidDescription,
                   "an anchor-free stage needs a reg_max and a class count of at least 1; got "
                   "{} and {}", facts.reg_max, facts.classes);

        std::vector<float> strides;
        std::int64_t anchors = 0;
        for (const FreeLevelFacts& level : facts.levels)
        {
            SA_FAIL_IF(level.height < 1 || level.width < 1 || level.stride < 1,
                       StatusCode::InvalidDescription,
                       "an anchor-free level needs positive extents and stride; got {} by {} "
                       "at stride {}", level.height, level.width, level.stride);
            strides.push_back(static_cast<float>(level.stride));
            anchors += level.height * level.width;
        }

        const bool one_to_one = branch == FreeBranch::OneToOne;
        const SushiRuntime::span<const float> level_strides(strides.data(), strides.size());
        const Detect::Free::AssignerConfig& settings =
            one_to_one ? task.one_to_one : task.one_to_many;
        Detect::Free::AssignerConfig config =
            one_to_one ? Detect::Free::one_to_one_assigner_config(facts.classes, level_strides)
                       : Detect::Free::one_to_many_assigner_config(facts.classes, level_strides);
        config.topk = settings.topk;
        config.topk2 = settings.topk2;
        config.alpha = settings.alpha;
        config.beta = settings.beta;
        SA_FAIL_IF(config.topk < 1 || config.topk2 < 1 || anchors < config.topk ||
                       anchors < config.topk2,
                   StatusCode::InvalidDescription,
                   "the {} topk {} and topk2 {} must lie in 1..{}, the anchor count",
                   one_to_one ? "one-to-one" : "one-to-many", config.topk, config.topk2,
                   anchors);

        return Result<FreeAssignmentStage>(FreeAssignmentStage(facts, config, batch, branch));
    }

    FreeAssignmentStage::FreeAssignmentStage(const FreeModelFacts& facts,
                                             const Detect::Free::AssignerConfig& config,
                                             std::int64_t batch, FreeBranch branch)
        : reg_max_(facts.reg_max), classes_(facts.classes), batch_(batch), assigner_(config)
    {
        const FreeLevelFacts& finest = facts.levels.front();
        image_height_ = static_cast<float>(finest.height * finest.stride);
        image_width_ = static_cast<float>(finest.width * finest.stride);

        for (const FreeLevelFacts& level : facts.levels)
        {
            offsets_.push_back(anchors_);
            levels_.push_back(
                FeatureLevel{level.height, level.width, static_cast<float>(level.stride)});
            anchors_ += level.height * level.width;
        }

        const SushiRuntime::span<const FeatureLevel> levels(levels_.data(), levels_.size());
        const std::size_t anchors = static_cast<std::size_t>(anchors_);
        points_.resize(anchors * Detect::Free::K_POINT_VALUES);
        pixel_points_.resize(points_.size());
        strides_.resize(anchors);
        Detect::Free::make_anchors(levels, view(points_), view(strides_));
        Detect::Free::scale_points(view(points_), view(strides_), view(pixel_points_));
        Detect::Free::build_anchor_leaves(levels, batch_, anchor_leaves_);

        for (const FreeLeafKind kind : K_LEVEL_KINDS)
        {
            if (kind == FreeLeafKind::DistributionWeights && reg_max_ == 1)
                continue;
            const std::int64_t channels = kind_channels(kind, classes_, reg_max_);
            for (std::size_t level = 0; level < levels_.size(); ++level)
            {
                const FeatureLevel& extents = levels_[level];
                leaves_.push_back(FreeLeaf{level_role(branch, kind, level),
                                           Shape{batch_, extents.height, extents.width, channels},
                                           kind, static_cast<std::int64_t>(level), branch});
            }
        }
        leaves_.push_back(FreeLeaf{role_prefix(branch).append(
                                       kind_name(FreeLeafKind::TargetScoresSum)),
                                   Shape{1}, FreeLeafKind::TargetScoresSum, 0, branch});
    }

    const std::vector<FreeLeaf>& FreeAssignmentStage::leaves() const noexcept
    {
        return leaves_;
    }

    void FreeAssignmentStage::assign(SushiRuntime::span<const FreeLevelViews> levels,
                                     const Detect::GroundTruthBatch& truth)
    {
        SA_ZONE("host", "assign");
        SA_THROW_IF(truth.batch != batch_, "the truth holds {} images for a stage of {}",
                    truth.batch, batch_);
        join_levels(levels);
        refuse_non_finite_truth(truth);
        decode_predictions();

        Detect::Free::AssignerInputs inputs;
        inputs.batch = batch_;
        inputs.anchors = anchors_;
        inputs.max_boxes = truth.max_boxes;
        inputs.scores = view(probabilities_);
        inputs.boxes = view(predicted_boxes_);
        inputs.anchor_points = view(pixel_points_);
        inputs.gt_labels =
            SushiRuntime::span<const std::int64_t>(truth.labels.data(), truth.labels.size());
        inputs.gt_boxes = view(truth.boxes);
        inputs.gt_valid =
            SushiRuntime::span<const std::uint8_t>(truth.valid.data(), truth.valid.size());
        assigner_.assign(inputs, assignment_);

        Detect::Free::TargetLeafInputs geometry;
        geometry.anchor_points = view(points_);
        geometry.strides = view(strides_);
        geometry.image_height = image_height_;
        geometry.image_width = image_width_;
        Detect::Free::build_target_leaves(assignment_, geometry, targets_);

        if (reg_max_ == 1)
            return;
        distribution_weights_.resize(targets_.target_distances.size() *
                                     static_cast<std::size_t>(reg_max_));
        Detect::Free::fill_distribution_weights(view(targets_.target_distances),
                                                view(targets_.normalized_weight), reg_max_,
                                                view(distribution_weights_));
    }

    void FreeAssignmentStage::join_levels(SushiRuntime::span<const FreeLevelViews> levels)
    {
        SA_THROW_IF(levels.size() != levels_.size(), "the stage has {} levels; got {} views",
                    levels_.size(), levels.size());
        const std::size_t rows = static_cast<std::size_t>(batch_ * anchors_);
        const std::int64_t box_channels =
            static_cast<std::int64_t>(Detect::K_BOX_VALUES) * reg_max_;
        box_outputs_.resize(rows * static_cast<std::size_t>(box_channels));
        probabilities_.resize(rows * static_cast<std::size_t>(classes_));

        for (std::size_t level = 0; level < levels_.size(); ++level)
        {
            const std::int64_t level_anchors = levels_[level].height * levels_[level].width;
            const std::size_t level_rows = static_cast<std::size_t>(batch_ * level_anchors);
            const FreeLevelViews& views = levels[level];
            SA_THROW_IF(views.boxes.size() != level_rows * static_cast<std::size_t>(box_channels) ||
                            views.scores.size() != level_rows * static_cast<std::size_t>(classes_),
                        "level {} needs {} box and {} score values; got {} and {}", level,
                        level_rows * static_cast<std::size_t>(box_channels),
                        level_rows * static_cast<std::size_t>(classes_), views.boxes.size(),
                        views.scores.size());

            const std::int64_t bad_box = Detect::first_non_finite(views.boxes);
            SA_THROW_IF(bad_box >= 0, "level {} box output {} is {}; every output must be finite",
                        level, bad_box, views.boxes[static_cast<std::size_t>(bad_box)]);
            const std::int64_t bad_score = Detect::first_non_finite(views.scores);
            SA_THROW_IF(bad_score >= 0, "level {} score {} is {}; every score must be finite",
                        level, bad_score, views.scores[static_cast<std::size_t>(bad_score)]);

            Detect::Free::LevelPlacement placement{batch_, anchors_, box_channels,
                                                   offsets_[level], level_anchors};
            Detect::Free::join_level(views.boxes, placement, view(box_outputs_));
            placement.channels = classes_;
            Detect::Free::join_level(views.scores, placement, view(probabilities_));
        }
    }

    void FreeAssignmentStage::refuse_non_finite_truth(const Detect::GroundTruthBatch& truth) const
    {
        constexpr std::size_t values = Detect::K_BOX_VALUES;
        for (std::size_t slot = 0; slot < truth.valid.size(); ++slot)
        {
            if (truth.valid[slot] == 0 || slot * values + values > truth.boxes.size())
                continue;

            const SushiRuntime::span<const float> box(truth.boxes.data() + slot * values, values);
            const std::int64_t bad = Detect::first_non_finite(box);
            SA_THROW_IF(bad >= 0, "truth slot {} has the box value {} at {}; it must be finite",
                        slot, box[static_cast<std::size_t>(bad)], bad);
        }
    }

    void FreeAssignmentStage::decode_predictions()
    {
        predicted_boxes_.resize(static_cast<std::size_t>(batch_ * anchors_) *
                                Detect::K_BOX_VALUES);
        Detect::Free::expect_distances(view(box_outputs_), reg_max_, view(predicted_boxes_));
        Detect::Free::decode_xyxy(view(predicted_boxes_), view(points_), view(predicted_boxes_));
        Detect::Free::scale_boxes(view(predicted_boxes_), view(strides_), view(predicted_boxes_));
        for (float& value : probabilities_)
            value = sigmoid(value);
    }

    void FreeAssignmentStage::write_leaf(std::size_t leaf, SushiRuntime::span<float> values) const
    {
        SA_THROW_IF(leaf >= leaves_.size(), "leaf {} is beyond the stage's {}", leaf,
                    leaves_.size());
        const FreeLeaf& described = leaves_[leaf];
        SA_THROW_IF(static_cast<std::int64_t>(values.size()) != described.shape.num_elements(),
                    "the leaf '{}' holds {} values; got {}", described.role,
                    described.shape.num_elements(), values.size());

        if (described.kind == FreeLeafKind::TargetScoresSum)
        {
            values[0] = targets_.target_scores_sum[0];
            return;
        }
        std::int64_t channels = 0;
        const SushiRuntime::span<const float> joined = joined_source(described.kind, channels);
        const std::size_t level = static_cast<std::size_t>(described.level);
        const Detect::Free::LevelPlacement placement{
            batch_, anchors_, channels, offsets_[level],
            levels_[level].height * levels_[level].width};
        Detect::Free::split_level(joined, placement, values);
    }

    SushiRuntime::span<const float> FreeAssignmentStage::joined_source(
        FreeLeafKind kind, std::int64_t& channels) const
    {
        channels = kind_channels(kind, classes_, reg_max_);
        switch (kind)
        {
            case FreeLeafKind::TargetScores:
                return view(assignment_.target_scores);
            case FreeLeafKind::BoxMask:
                return view(targets_.box_mask);
            case FreeLeafKind::NormalizedWeight:
                return view(targets_.normalized_weight);
            case FreeLeafKind::TargetBoxes:
                return view(targets_.target_boxes);
            case FreeLeafKind::TargetDistances:
                return view(targets_.target_distances);
            case FreeLeafKind::L1Scale:
                return view(targets_.l1_scale);
            case FreeLeafKind::SafeBoxes:
                return view(targets_.safe_boxes);
            case FreeLeafKind::DistributionWeights:
                return view(distribution_weights_);
            case FreeLeafKind::AnchorPoints:
                return view(anchor_leaves_.points);
            case FreeLeafKind::AnchorStrides:
                return view(anchor_leaves_.strides);
            case FreeLeafKind::TargetScoresSum:
                break;
        }
        SA_THROW_IF(true, "the leaf kind {} is not cut from a joined array",
                    static_cast<int>(kind));
        return {};
    }

    const Detect::Free::Assignment& FreeAssignmentStage::assignment() const noexcept
    {
        return assignment_;
    }

    const Detect::Free::TargetLeaves& FreeAssignmentStage::targets() const noexcept
    {
        return targets_;
    }

    const std::vector<float>& FreeAssignmentStage::distribution_weights() const noexcept
    {
        return distribution_weights_;
    }
} // namespace SushiAI::Vision::Training
