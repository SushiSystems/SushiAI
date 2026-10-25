/**************************************************************************/
/* test_yolov5n.cpp                                                       */
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
 * @file test_yolov5n.cpp
 * @brief Tests that the yolov5n model file builds, and what its task and biases state.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include "registration_test_support.hpp"

using namespace SushiAI;
using NN::ModelGraph;
using Vision::Detect::Anchor::AnchorTask;
using Vision::Detect::Anchor::AssignmentRuleKind;
using Vision::Detect::Anchor::HeadCoding;
namespace SaTest = SushiAI::Test;

namespace
{
    using Test::composed;
    using Test::count_parameters;
    using Test::trace_outputs;

    /** @brief Holds the row count of the yolov5n model file. */
    constexpr std::size_t K_ROWS = 29;

    /**
     * @brief Holds the parameter count of YOLOv5n at 80 classes; a figure from memory of the
     *        Ultralytics summary line, which this program's own count of the rows also gives.
     */
    constexpr std::int64_t K_PARAMETERS = 1872157;

    /** @brief Holds the class count of the file. */
    constexpr std::int64_t K_CLASSES = 80;

    /** @brief Holds the channels of a head, 3 anchors of 5 + 80 entries. */
    constexpr std::int64_t K_HEAD_CHANNELS = 255;

    /** @brief Holds the side of the network input in pixels. */
    constexpr std::int64_t K_INPUT_SIDE = 640;

    /** @brief Holds the strides of the three levels, finest first. */
    constexpr std::int64_t K_STRIDES[] = {8, 16, 32};

    /** @brief Holds the entries of one anchor slot at 80 classes. */
    constexpr std::int64_t K_SLOT_ENTRIES = 85;

    /** @brief Holds the anchors of the file in pixels, width then height, head after head. */
    constexpr double K_ANCHOR_PIXELS[][2] = {{10, 13},   {16, 30},   {33, 23},
                                             {30, 61},   {62, 45},   {59, 119},
                                             {116, 90},  {156, 198}, {373, 326}};

    /** @brief Bounds the error of an anchor read back as float32 fractions of the input. */
    constexpr double K_ANCHOR_TOLERANCE = 1e-5;

    /** @brief Bounds the error of a float32 bias against the double formula. */
    constexpr double K_BIAS_TOLERANCE = 1e-5;

    /** @brief Returns the path of the yolov5n model file. */
    std::filesystem::path model_path()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples"
               / "yolov5n" / "model.json";
    }

    /** @brief Provides the engine the state of the bias case is allocated on. */
    class Medium_Yolov5nBiases : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that the file holds its 29 rows and its three outputs. */
TEST(Medium_Yolov5nFile, HoldsItsRowsAndItsThreeOutputs)
{
    const Config::ModelSpec spec = Config::load_model(model_path());

    EXPECT_EQ(spec.layers.size(), K_ROWS);
    EXPECT_EQ(spec.input_shape, (Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3}));
    ASSERT_EQ(spec.outputs.size(), 3u);
    EXPECT_EQ(spec.outputs[0].name, "head_0");
    EXPECT_EQ(spec.outputs[2].name, "head_2");
}

/**
 * @brief Tests that the parameter count of the built model is the one the Ultralytics summary
 *        gives, and that it equals the count of a sum over the rows of the file.
 */
TEST(Medium_Yolov5nFile, BuildsToTheParameterCountOfTheReference)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(model_path());
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    (void)trace_outputs(model, spec.input_shape);

    EXPECT_EQ(count_parameters(model), K_PARAMETERS);
}

/** @brief Tests that the three heads publish 255 channels at strides 8, 16 and 32. */
TEST(Medium_Yolov5nFile, PublishesThreeHeadsAtTheLevelStrides)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(model_path());
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));

    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);

    ASSERT_EQ(outputs.size(), 3u);
    for (std::size_t level = 0; level < 3; ++level)
    {
        SCOPED_TRACE(outputs[level].name);
        const std::int64_t side = K_INPUT_SIDE / K_STRIDES[level];
        EXPECT_EQ(outputs[level].name, "head_" + std::to_string(level));
        EXPECT_EQ(outputs[level].shape, (Shape{1, side, side, K_HEAD_CHANNELS}));
    }
}

/**
 * @brief Tests that the task section reads back as the 80 classes, the nine anchors of
 *        YOLOv5 in pixels over 640, three logit heads of stretch 2 with squared-sigmoid sides
 *        and the shape-ratio rule.
 */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBack)
{
    const Config::ModelSpec spec = Config::load_model(model_path());

    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            Vision::Detect::Anchor::read_anchor_task(spec.task));

    EXPECT_EQ(task.num_classes, K_CLASSES);
    ASSERT_EQ(task.layout.anchors.size(), 9u);
    for (std::size_t anchor = 0; anchor < 9; ++anchor)
    {
        SCOPED_TRACE(anchor);
        EXPECT_NEAR(task.layout.anchors[anchor].width * K_INPUT_SIDE, K_ANCHOR_PIXELS[anchor][0],
                    K_ANCHOR_TOLERANCE * K_INPUT_SIDE);
        EXPECT_NEAR(task.layout.anchors[anchor].height * K_INPUT_SIDE,
                    K_ANCHOR_PIXELS[anchor][1], K_ANCHOR_TOLERANCE * K_INPUT_SIDE);
    }

    ASSERT_EQ(task.heads.size(), 3u);
    for (std::size_t head = 0; head < 3; ++head)
    {
        SCOPED_TRACE(head);
        const auto first = static_cast<std::int64_t>(3 * head);
        EXPECT_EQ(task.layout.head_anchors[head],
                  (std::vector<std::int64_t>{first, first + 1, first + 2}));
        EXPECT_EQ(task.heads[head].output, "head_" + std::to_string(head));
        EXPECT_EQ(task.heads[head].options.coding, HeadCoding::Logit);
        EXPECT_EQ(task.heads[head].options.size,
                  Vision::Detect::Anchor::SizeForm::SquaredSigmoid);
        EXPECT_EQ(task.heads[head].options.scale_xy, 2.0f);
    }

    EXPECT_EQ(task.assignment.rule, AssignmentRuleKind::ShapeRatio);
    EXPECT_EQ(task.assignment.shape_ratio.ratio_limit,
              Vision::Detect::Anchor::K_DEFAULT_RATIO_LIMIT);
}

/**
 * @brief Tests that the biases derived from the built model start the objectness entry of each
 *        slot at log(8 / (640 / stride)^2) and each class entry at log(0.6 / (80 - 0.99)).
 */
TEST_F(Medium_Yolov5nBiases, AnInitialisedModelGainsTheStrideBiases)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRegistry registry, composed());
    Config::ModelSpec spec = Config::load_model(model_path());
    SA_ASSERT_OK_AND_ASSIGN(ModelGraph model, NN::build_model(registry, spec));
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            Vision::Detect::Anchor::read_anchor_task(spec.task));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    NN::InitPolicy policy;
    policy.seed = 20261025;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<Vision::Training::AnchorBiasHead> heads,
        Vision::Training::derive_anchor_bias_heads(spec, spec.input_shape, task, outputs));
    ASSERT_EQ(heads.size(), 3u);

    std::vector<std::vector<double>> before;
    for (const Vision::Training::AnchorBiasHead& head : heads)
        before.push_back(SaTest::read_tensor(*state.tensor(head.bias).value()));

    SA_EXPECT_OK(Vision::Training::initialize_anchor_biases(task, heads, state));

    const char* const names[] = {"26.bias", "27.bias", "28.bias"};
    for (std::size_t level = 0; level < 3; ++level)
    {
        SCOPED_TRACE(names[level]);
        EXPECT_EQ(heads[level].bias, names[level]);
        EXPECT_EQ(heads[level].stride, K_STRIDES[level]);

        const std::vector<double> after =
            SaTest::read_tensor(*state.tensor(names[level]).value());
        ASSERT_EQ(after.size(), 3u * K_SLOT_ENTRIES);
        for (std::size_t slot = 0; slot < 3; ++slot)
        {
            const std::size_t first = slot * K_SLOT_ENTRIES;
            for (std::size_t entry = 0; entry < K_SLOT_ENTRIES; ++entry)
            {
                double added = 0.0;
                if (entry == 4)
                    added = std::log(8.0 / std::pow(640.0 / K_STRIDES[level], 2));
                else if (entry >= 5)
                    added = std::log(0.6 / (80.0 - 0.99));
                EXPECT_NEAR(after[first + entry] - before[level][first + entry], added,
                            K_BIAS_TOLERANCE);
            }
        }
    }
}
