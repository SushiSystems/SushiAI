/**************************************************************************/
/* test_free_model_facts.cpp                                              */
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
 * @file test_free_model_facts.cpp
 * @brief Tests the facts derived from the ports of a detect head over a 64 by 64 input.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

using namespace SushiAI;
using Vision::Blocks::DetectHead;
using Vision::Blocks::DetectOptions;
using Vision::Training::derive_free_model_facts;
using Vision::Training::FreeModelFacts;

namespace
{
    /** @brief Holds the class count of the fixture. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the batch extent of the published outputs. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the H and the W of each level over a 64 by 64 input. */
    constexpr std::array<std::int64_t, 3> K_SIDES = {8, 4, 2};

    /** @brief Holds the channels of each level's feature map. */
    constexpr std::array<std::int64_t, 3> K_WIDTHS = {16, 32, 64};

    /** @brief Holds the [H, W, C] input sample of the fixture. */
    const Shape K_INPUT{64, 64, 3};

    /** @brief Builds a head over K_SIDES and returns its ports as published outputs. */
    std::vector<Train::OutputShape> published_ports(const DetectOptions& options)
    {
        std::vector<Shape> levels;
        for (std::size_t level = 0; level < K_SIDES.size(); ++level)
            levels.push_back(Shape{K_SIDES[level], K_SIDES[level], K_WIDTHS[level]});

        Result<std::unique_ptr<DetectHead>> made = DetectHead::create(
            NN::ShapeSpan(levels.data(), levels.size()), K_CLASSES, NN::NormOptions{}, options);
        EXPECT_TRUE(made.status().is_ok());
        std::vector<Train::OutputShape> outputs;
        if (!made.status().is_ok())
            return outputs;

        for (const NN::Port& port : made.value()->ports())
        {
            const Shape& shape = port.shape;
            outputs.push_back(Train::OutputShape{
                port.name, Shape{K_BATCH, shape.dim(0), shape.dim(1), shape.dim(2)}});
        }
        return outputs;
    }
} // namespace

/** @brief Tests that both copies give three levels of strides 8, 16 and 32 and their ports. */
TEST(Small_TrainingFreeModelFacts, AThreeLevelHeadGivesItsStridesPortsAndWidths)
{
    const std::vector<Train::OutputShape> outputs = published_ports(DetectOptions());
    SA_ASSERT_OK_AND_ASSIGN(const FreeModelFacts facts, derive_free_model_facts(K_INPUT, outputs));

    EXPECT_EQ(facts.reg_max, 1);
    EXPECT_EQ(facts.classes, K_CLASSES);
    EXPECT_TRUE(facts.has_one_to_one);
    ASSERT_EQ(facts.levels.size(), 3u);

    const std::array<std::int64_t, 3> strides = {8, 16, 32};
    for (std::size_t level = 0; level < 3; ++level)
    {
        const std::string index = std::to_string(level);
        EXPECT_EQ(facts.levels[level].stride, strides[level]);
        EXPECT_EQ(facts.levels[level].height, K_SIDES[level]);
        EXPECT_EQ(facts.levels[level].width, K_SIDES[level]);
        EXPECT_EQ(facts.levels[level].one_to_many.boxes, "one2many.level" + index + ".boxes");
        EXPECT_EQ(facts.levels[level].one_to_many.scores, "one2many.level" + index + ".scores");
        EXPECT_EQ(facts.levels[level].one_to_one.boxes, "one2one.level" + index + ".boxes");
        EXPECT_EQ(facts.levels[level].one_to_one.scores, "one2one.level" + index + ".scores");
    }
}

/** @brief Tests that a single-copy head of reg_max 16 gives reg_max 16 and no one-to-one. */
TEST(Small_TrainingFreeModelFacts, ASingleCopyHeadGivesItsRegMax)
{
    DetectOptions options;
    options.reg_max = 16;
    options.end2end = false;
    const std::vector<Train::OutputShape> outputs = published_ports(options);
    SA_ASSERT_OK_AND_ASSIGN(const FreeModelFacts facts, derive_free_model_facts(K_INPUT, outputs));

    EXPECT_EQ(facts.reg_max, 16);
    EXPECT_FALSE(facts.has_one_to_one);
    ASSERT_EQ(facts.levels.size(), 3u);
    EXPECT_TRUE(facts.levels[0].one_to_one.boxes.empty());
}

/** @brief Tests that a level that does not divide the input is refused, naming its port. */
TEST(Small_TrainingFreeModelFacts, RefusesALevelThatDoesNotDivideTheInput)
{
    const std::vector<Train::OutputShape> outputs = published_ports(DetectOptions());
    SA_EXPECT_FAILURE(derive_free_model_facts(Shape{60, 60, 3}, outputs),
                      StatusCode::InvalidDescription, "'one2many.level0.boxes'", "8 by 8",
                      "60 by 60");
    SA_EXPECT_FAILURE(derive_free_model_facts(Shape{64, 32, 3}, outputs),
                      StatusCode::InvalidDescription, "'one2many.level0.boxes'",
                      "equal along H and W");
}

/** @brief Tests that a model without the detect ports under their own names is refused. */
TEST(Small_TrainingFreeModelFacts, RefusesOutputsWithoutTheDetectPorts)
{
    const std::vector<Train::OutputShape> outputs = {
        Train::OutputShape{"boxes", Shape{K_BATCH, 8, 8, 4}}};
    SA_EXPECT_FAILURE(derive_free_model_facts(K_INPUT, outputs), StatusCode::InvalidDescription,
                      "'one2many.level0.boxes'");
}
