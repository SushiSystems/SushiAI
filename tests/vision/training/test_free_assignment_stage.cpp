/**************************************************************************/
/* test_free_assignment_stage.cpp                                         */
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

/**
 * @file test_free_assignment_stage.cpp
 * @brief Tests the anchor-free assignment stage on two images over two levels of a 32 by 32 input.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_assignment_stage.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

using namespace SushiAI;
using Vision::Training::FreeAssignmentStage;
using Vision::Training::FreeLeaf;
using Vision::Training::FreeLeafKind;
using Vision::Training::FreeLevelViews;
using Vision::Training::FreeModelFacts;

namespace
{
    /** @brief Holds the images of the fixture. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of the fixture. */
    constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the side of each level: 4 at stride 8, 2 at stride 16. */
    constexpr std::array<std::int64_t, 2> K_SIDES = {4, 2};

    /** @brief Holds the stride of each level. */
    constexpr std::array<std::int64_t, 2> K_STRIDES = {8, 16};

    /** @brief Holds the anchors over both levels: 16 then 4. */
    constexpr std::size_t K_ANCHORS = 20;

    /** @brief Holds the grid distance every side of a matching prediction decodes to. */
    constexpr float K_MATCH_DISTANCE = 0.5f;

    /** @brief Holds the grid distance every side of any other prediction decodes to. */
    constexpr float K_OTHER_DISTANCE = 0.1f;

    /** @brief Holds the logit that leaves a bin out of the softmax in float32. */
    constexpr float K_ABSENT_BIN = -100.0f;

    /** @brief Holds the tolerance of a value that passes through a CIoU or a softmax. */
    constexpr float K_TOLERANCE = 1e-4f;

    /** @brief Holds the foreground anchor of truth A in image 0: level 0, row 1, column 1. */
    constexpr std::size_t K_ANCHOR_A = 5;

    /** @brief Holds the foreground anchor of truth B in image 0: level 1, row 1, column 1. */
    constexpr std::size_t K_ANCHOR_B = 19;

    /** @brief Holds the foreground anchor of truth C in image 1: level 1, row 0, column 0. */
    constexpr std::size_t K_ANCHOR_C = 16;

    /** @brief Holds the level-0 anchors inside truth C that top-k keeps at metric 0. */
    constexpr std::array<std::size_t, 4> K_ZERO_METRIC_C = {0, 1, 4, 5};

    /** @brief Returns the facts of a model of K_CLASSES classes over K_SIDES at @p reg_max. */
    FreeModelFacts make_facts(std::int64_t reg_max)
    {
        FreeModelFacts facts;
        facts.reg_max = reg_max;
        facts.classes = K_CLASSES;
        for (std::size_t level = 0; level < K_SIDES.size(); ++level)
        {
            Vision::Training::FreeLevelFacts entry;
            entry.height = K_SIDES[level];
            entry.width = K_SIDES[level];
            entry.stride = K_STRIDES[level];
            facts.levels.push_back(entry);
        }
        return facts;
    }

    /** @brief Holds the two levels' port values of the fixture. */
    struct Ports
    {
        std::array<std::vector<float>, 2> boxes;  /**< `[B, H, W, 4 * reg_max]` per level. */
        std::array<std::vector<float>, 2> scores; /**< `[B, H, W, C]` per level, all 0. */
    };

    /** @brief Writes the box outputs of one cell so that each side decodes to @p distance. */
    void set_cell(Ports& ports, std::size_t level, std::size_t image, std::size_t row,
                  std::size_t column, float distance, std::int64_t reg_max)
    {
        const std::size_t side = static_cast<std::size_t>(K_SIDES[level]);
        const std::size_t bins = static_cast<std::size_t>(reg_max);
        const std::size_t cell = (image * side + row) * side + column;
        float* out = ports.boxes[level].data() + cell * 4 * bins;
        for (std::size_t box_side = 0; box_side < 4; ++box_side)
        {
            float* logits = out + box_side * bins;
            if (bins == 1)
            {
                logits[0] = distance;
                continue;
            }
            // Bins 0 and 1 share the mass, so the expectation is the weight of bin 1.
            logits[0] = std::log((1.0f - distance) / distance);
            logits[1] = 0.0f;
            for (std::size_t bin = 2; bin < bins; ++bin)
                logits[bin] = K_ABSENT_BIN;
        }
    }

    /** @brief Returns the fixture's ports: three matching cells, every other cell small. */
    Ports make_ports(std::int64_t reg_max)
    {
        Ports ports;
        for (std::size_t level = 0; level < K_SIDES.size(); ++level)
        {
            const std::size_t side = static_cast<std::size_t>(K_SIDES[level]);
            const std::size_t cells = static_cast<std::size_t>(K_BATCH) * side * side;
            ports.boxes[level].assign(cells * 4 * static_cast<std::size_t>(reg_max), 0.0f);
            ports.scores[level].assign(cells * static_cast<std::size_t>(K_CLASSES), 0.0f);
            for (std::size_t image = 0; image < 2; ++image)
                for (std::size_t row = 0; row < side; ++row)
                    for (std::size_t column = 0; column < side; ++column)
                        set_cell(ports, level, image, row, column, K_OTHER_DISTANCE, reg_max);
        }
        set_cell(ports, 0, 0, 1, 1, K_MATCH_DISTANCE, reg_max);
        set_cell(ports, 1, 0, 1, 1, K_MATCH_DISTANCE, reg_max);
        set_cell(ports, 1, 1, 0, 0, K_MATCH_DISTANCE, reg_max);
        return ports;
    }

    /** @brief Returns truths A [8, 8, 16, 16] class 0 and B [16, 16, 32, 32] class 1 in image
     *         0, and C [0, 0, 16, 16] class 1 beside a padding row in image 1. */
    Vision::Detect::GroundTruthBatch make_truth()
    {
        Vision::Detect::GroundTruthBatch truth;
        truth.batch = K_BATCH;
        truth.max_boxes = 2;
        truth.labels = {0, 1, 1, 0};
        truth.boxes = {8.0f, 8.0f,  16.0f, 16.0f, 16.0f, 16.0f, 32.0f, 32.0f,
                       0.0f, 0.0f,  16.0f, 16.0f, 0.0f,  0.0f,  0.0f,  0.0f};
        truth.valid = {1, 1, 1, 0};
        return truth;
    }

    /** @brief Builds the stage at @p reg_max and runs it on the fixture. */
    FreeAssignmentStage run_fixture(std::int64_t reg_max)
    {
        Result<FreeAssignmentStage> made = FreeAssignmentStage::make(
            make_facts(reg_max), Vision::Detect::Free::FreeTask{}, K_BATCH);
        EXPECT_TRUE(made.status().is_ok());
        FreeAssignmentStage stage = std::move(made.value());

        const Ports ports = make_ports(reg_max);
        std::array<FreeLevelViews, 2> views;
        for (std::size_t level = 0; level < views.size(); ++level)
        {
            views[level].boxes = SushiRuntime::span<const float>(ports.boxes[level].data(),
                                                                 ports.boxes[level].size());
            views[level].scores = SushiRuntime::span<const float>(ports.scores[level].data(),
                                                                  ports.scores[level].size());
        }
        stage.assign(SushiRuntime::span<const FreeLevelViews>(views.data(), views.size()),
                     make_truth());
        return stage;
    }

    /** @brief Returns the index of the leaf named @p role, or the leaf count. */
    std::size_t find_leaf(const FreeAssignmentStage& stage, const std::string& role)
    {
        const std::vector<FreeLeaf>& leaves = stage.leaves();
        for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf)
        {
            if (leaves[leaf].role == role)
                return leaf;
        }
        return leaves.size();
    }

    /** @brief Checks the foreground anchors, target scores and normaliser of the fixture. */
    void expect_fixture_targets(const FreeAssignmentStage& stage)
    {
        const auto& assignment = stage.assignment();
        for (std::size_t row = 0; row < K_BATCH * K_ANCHORS; ++row)
        {
            bool foreground = row == K_ANCHOR_A || row == K_ANCHOR_B ||
                              row == K_ANCHORS + K_ANCHOR_C;
            for (const std::size_t anchor : K_ZERO_METRIC_C)
                foreground = foreground || row == K_ANCHORS + anchor;
            EXPECT_EQ(assignment.foreground_mask[row], foreground ? 1 : 0) << "row " << row;
        }
        const std::vector<float>& scores = assignment.target_scores;
        EXPECT_NEAR(scores[K_ANCHOR_A * 2 + 0], 1.0f, K_TOLERANCE);
        EXPECT_NEAR(scores[K_ANCHOR_A * 2 + 1], 0.0f, K_TOLERANCE);
        EXPECT_NEAR(scores[K_ANCHOR_B * 2 + 1], 1.0f, K_TOLERANCE);
        EXPECT_NEAR(scores[(K_ANCHORS + K_ANCHOR_C) * 2 + 1], 1.0f, K_TOLERANCE);
        for (const std::size_t anchor : K_ZERO_METRIC_C)
            EXPECT_EQ(scores[(K_ANCHORS + anchor) * 2 + 1], 0.0f) << "anchor " << anchor;

        const auto& targets = stage.targets();
        EXPECT_NEAR(targets.target_scores_sum[0], 3.0f, 3.0f * K_TOLERANCE);
        EXPECT_NEAR(targets.normalized_weight[K_ANCHOR_A], 1.0f / 3.0f, K_TOLERANCE);
        const std::array<float, 4> box_a = {1.0f, 1.0f, 2.0f, 2.0f};
        const std::array<float, 4> box_c = {0.0f, 0.0f, 1.0f, 1.0f};
        for (std::size_t side = 0; side < 4; ++side)
        {
            EXPECT_NEAR(targets.target_boxes[K_ANCHOR_A * 4 + side], box_a[side], K_TOLERANCE);
            EXPECT_NEAR(targets.target_boxes[K_ANCHOR_B * 4 + side], box_a[side], K_TOLERANCE);
            EXPECT_NEAR(targets.target_boxes[(K_ANCHORS + K_ANCHOR_C) * 4 + side], box_c[side],
                        K_TOLERANCE);
            EXPECT_NEAR(targets.target_distances[K_ANCHOR_A * 4 + side], K_MATCH_DISTANCE,
                        K_TOLERANCE);
        }
        EXPECT_NEAR(targets.l1_scale[K_ANCHOR_A * 4], 1.0f / 48.0f, K_TOLERANCE);
    }
} // namespace

TEST(Small_VisionTrainingFreeAssignmentStage, MakeRefusesABatchBelowOne)
{
    const Result<FreeAssignmentStage> made =
        FreeAssignmentStage::make(make_facts(1), Vision::Detect::Free::FreeTask{}, 0);
    EXPECT_EQ(made.status().code(), StatusCode::InvalidDescription);
}

TEST(Small_VisionTrainingFreeAssignmentStage, LeavesListEachKindPerLevelThenTheSum)
{
    Result<FreeAssignmentStage> made =
        FreeAssignmentStage::make(make_facts(1), Vision::Detect::Free::FreeTask{}, K_BATCH);
    ASSERT_TRUE(made.status().is_ok());
    const std::vector<FreeLeaf>& leaves = made.value().leaves();

    ASSERT_EQ(leaves.size(), 19u);
    EXPECT_EQ(leaves.front().role, "free.target_scores.level0");
    EXPECT_EQ(leaves.front().shape, (Shape{K_BATCH, 4, 4, K_CLASSES}));
    EXPECT_EQ(leaves[3].role, "free.box_mask.level1");
    EXPECT_EQ(leaves[3].shape, (Shape{K_BATCH, 2, 2, 4}));
    EXPECT_EQ(leaves.back().kind, FreeLeafKind::TargetScoresSum);
    EXPECT_EQ(leaves.back().shape, (Shape{1}));
    EXPECT_EQ(find_leaf(made.value(), "free.distribution_weights.level0"), leaves.size());
}

TEST(Small_VisionTrainingFreeAssignmentStage, RawDistancesGiveHandWorkedTargets)
{
    const FreeAssignmentStage stage = run_fixture(1);
    expect_fixture_targets(stage);
    EXPECT_TRUE(stage.distribution_weights().empty());

    const std::size_t mask = find_leaf(stage, "free.box_mask.level1");
    ASSERT_LT(mask, stage.leaves().size());
    std::vector<float> values(static_cast<std::size_t>(K_BATCH) * 2 * 2 * 4, -1.0f);
    stage.write_leaf(mask, SushiRuntime::span<float>(values.data(), values.size()));
    for (std::size_t cell = 0; cell < values.size() / 4; ++cell)
    {
        const float expected = cell == 3 || cell == 4 ? 1.0f : 0.0f;
        for (std::size_t side = 0; side < 4; ++side)
            EXPECT_EQ(values[cell * 4 + side], expected) << "cell " << cell;
    }

    std::array<float, 1> sum = {0.0f};
    stage.write_leaf(stage.leaves().size() - 1, SushiRuntime::span<float>(sum.data(), 1));
    EXPECT_NEAR(sum[0], 3.0f, 3.0f * K_TOLERANCE);
}

TEST(Small_VisionTrainingFreeAssignmentStage, BinnedDistancesGiveTheSameTargetsAndBinWeights)
{
    constexpr std::int64_t reg_max = 4;
    const FreeAssignmentStage stage = run_fixture(reg_max);
    expect_fixture_targets(stage);

    const std::vector<float>& weights = stage.distribution_weights();
    ASSERT_EQ(weights.size(), static_cast<std::size_t>(K_BATCH) * K_ANCHORS * 4 * reg_max);
    const std::array<float, 4> side_weights = {1.0f / 24.0f, 1.0f / 24.0f, 0.0f, 0.0f};
    for (std::size_t side = 0; side < 4; ++side)
        for (std::size_t bin = 0; bin < 4; ++bin)
            EXPECT_NEAR(weights[(K_ANCHOR_A * 4 + side) * 4 + bin], side_weights[bin],
                        K_TOLERANCE);
    for (std::size_t value = 0; value < 4 * reg_max; ++value)
        EXPECT_EQ(weights[4 * reg_max + value], 0.0f);

    const std::size_t level0 = find_leaf(stage, "free.distribution_weights.level0");
    ASSERT_LT(level0, stage.leaves().size());
    EXPECT_EQ(stage.leaves()[level0].shape, (Shape{K_BATCH, 4, 4, 16}));
}
