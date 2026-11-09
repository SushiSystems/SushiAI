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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_yolov5n.cpp
 * @brief Tests that the yolov5n model file builds, and what its task and biases state.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
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
#include <support/tolerance.hpp>

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
     * @brief Holds the parameter count of YOLOv5n at 80 classes: the 1872157 parameters of the
     *        summary line Ultralytics prints when it loads yolov5n.yaml. The figure is the
     *        published one, not a count made here.
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

    /**
     * @brief Holds the absolute Bounds the error of an anchor read back as float32 fractions of the
     *        input: 256 times the float unit roundoff, the smallest power of two not tighter than
     *        the earlier 1e-5.
     */
    constexpr double K_ANCHOR_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Holds the absolute Bounds the error of a float32 bias against the double formula: 256
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-5.
     */
    constexpr double K_BIAS_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Names the bias of the head of each level, finest first. */
    const char* const K_BIAS_NAMES[] = {"26.bias", "27.bias", "28.bias"};

    /** @brief Returns the path of the yolov5n model file. */
    std::filesystem::path model_path()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples"
               / "yolov5n" / "model.json";
    }
} // namespace

namespace
{
    /** @brief Loads the yolov5n model file. */
    Config::ModelSpec load_spec()
    {
        return SaTest::require_value(Config::load_model(model_path()));
    }

    /** @brief Reads the anchor task of the yolov5n model file. */
    AnchorTask read_task(const Config::ModelSpec& spec)
    {
        return SaTest::require_value(Vision::Detect::Anchor::read_anchor_task(spec.task));
    }

    /** @brief Builds the model of @p spec over the composed registry. */
    ModelGraph build(const Config::ModelSpec& spec)
    {
        const NN::NodeRegistry registry = SaTest::require_value(composed());

        return SaTest::require_value(NN::build_model(registry, spec));
    }

    /** @brief Gives each level of the head, finest first, to a test. */
    class Medium_Yolov5nLevel : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Gives each anchor of the file, in pixels, to a test. */
    class Medium_Yolov5nAnchor : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Gives each head of the task section to a test. */
    class Medium_Yolov5nHead : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Gives each level of the head to a test that derives its bias on the graph alone. */
    class Small_Yolov5nBiasHead : public ::testing::TestWithParam<std::size_t> {};

    /** @brief Gives each level of the head to a test that starts its bias on a host engine. */
    class Small_Yolov5nBiasInit : public ::testing::TestWithParam<std::size_t>
    {
        protected:
            /**
             * @brief Zeroes the bias of the head at @p level, runs initialize_anchor_biases on
             *        the host state and returns the bias it left.
             */
            std::vector<double> biases_after_initialising(std::size_t level)
            {
                const Config::ModelSpec spec = load_spec();
                ModelGraph model = build(spec);
                const std::vector<Train::OutputShape> outputs =
                    trace_outputs(model, spec.input_shape);
                const AnchorTask task = read_task(spec);
                NN::ModelState state =
                    SaTest::require_value(NN::ModelState::create(backend.engine(), model));
                const std::vector<Vision::Training::AnchorBiasHead> heads =
                    SaTest::require_value(Vision::Training::derive_anchor_bias_heads(
                        spec, spec.input_shape, task, outputs));

                SushiBLAS::Tensor& bias = *SaTest::require_value(state.tensor(heads[level].bias));
                const std::vector<double> zeros(static_cast<std::size_t>(bias.num_elements), 0.0);
                SaTest::write_tensor(bias, zeros);
                SA_EXPECT_OK(Vision::Training::initialize_anchor_biases(task, heads, state));

                return SaTest::read_tensor(bias);
            }

            /** @brief Holds the host engine the state is allocated on; no device is used. */
            HostExecutionBackend backend;
    };

    /** @brief Names a level in the name of its test. */
    std::string level_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "level" + std::to_string(info.param);
    }

    /** @brief Names an anchor in the name of its test. */
    std::string anchor_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "anchor" + std::to_string(info.param);
    }

    /** @brief Names a head in the name of its test. */
    std::string head_name(const ::testing::TestParamInfo<std::size_t>& info)
    {
        return "head" + std::to_string(info.param);
    }

    /**
     * @brief Collects entries @p first to @p last of every anchor slot of a head's bias.
     * @param bias The bias of one head: 3 slots of K_SLOT_ENTRIES.
     */
    std::vector<double> slot_entries(const std::vector<double>& bias, std::size_t first,
                                     std::size_t last)
    {
        std::vector<double> entries;
        for (std::size_t slot = 0; slot < 3; ++slot)
        {
            for (std::size_t entry = first; entry <= last; ++entry)
                entries.push_back(bias[slot * K_SLOT_ENTRIES + entry]);
        }

        return entries;
    }

    /** @brief Checks that every element of @p actual is within @p tolerance of @p expected. */
    ::testing::AssertionResult all_near(const std::vector<double>& actual, double expected,
                                        double tolerance)
    {
        for (std::size_t at = 0; at < actual.size(); ++at)
        {
            if (!(std::fabs(actual[at] - expected) <= tolerance))
                return ::testing::AssertionFailure()
                       << "element " << at << " is " << actual[at] << ", expected " << expected
                       << " within " << tolerance;
        }

        return ::testing::AssertionSuccess();
    }
} // namespace

/** @brief Tests that the file holds its 29 rows. */
TEST(Medium_Yolov5nFile, HoldsItsRows)
{
    EXPECT_EQ(load_spec().layers.size(), K_ROWS);
}

/** @brief Tests that the file takes a 640 by 640 by 3 input. */
TEST(Medium_Yolov5nFile, TakesA640By640By3Input)
{
    EXPECT_EQ(load_spec().input_shape, (Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3}));
}

/** @brief Tests that the file publishes three outputs, head_0 to head_2. */
TEST(Medium_Yolov5nFile, PublishesThreeOutputsNamedHead0ToHead2)
{
    const Config::ModelSpec spec = load_spec();

    ASSERT_EQ(spec.outputs.size(), 3u);
    EXPECT_EQ(spec.outputs[0].name, "head_0");
    EXPECT_EQ(spec.outputs[2].name, "head_2");
}

/** @brief Tests that the parameter count of the built model is the one the Ultralytics summary gives. */
TEST(Medium_Yolov5nFile, BuildsToTheParameterCountOfTheReference)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    (void)trace_outputs(model, spec.input_shape);

    EXPECT_EQ(count_parameters(model), K_PARAMETERS);
}

/** @brief Tests that the parameter count of the built model equals the sum over the rows of the file. */
TEST(Medium_Yolov5nFile, BuildsToTheSumOfTheParameterCountsOfItsRows)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    (void)trace_outputs(model, spec.input_shape);
    const std::vector<std::int64_t> rows = SaTest::count_row_parameters(model);

    EXPECT_EQ(rows.size(), K_ROWS);
    EXPECT_EQ(count_parameters(model), std::accumulate(rows.begin(), rows.end(), std::int64_t{0}));
}

/** @brief Tests that a head is named head_<level>. */
TEST_P(Medium_Yolov5nLevel, PublishesTheHeadOfItsName)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(outputs[GetParam()].name, "head_" + std::to_string(GetParam()));
}

/** @brief Tests that a head publishes 255 channels at the stride of its level. */
TEST_P(Medium_Yolov5nLevel, PublishesTheHeadAt255ChannelsAtTheLevelStride)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);
    const std::int64_t side = K_INPUT_SIDE / K_STRIDES[GetParam()];

    ASSERT_EQ(outputs.size(), 3u);
    EXPECT_EQ(outputs[GetParam()].shape, (Shape{1, side, side, K_HEAD_CHANNELS}));
}

INSTANTIATE_TEST_SUITE_P(Levels, Medium_Yolov5nLevel, ::testing::Values(0, 1, 2), level_name);

/** @brief Tests that the task section reads back as 80 classes. */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBackEightyClasses)
{
    EXPECT_EQ(read_task(load_spec()).num_classes, K_CLASSES);
}

/** @brief Tests that the task section reads back as nine anchors. */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBackNineAnchors)
{
    EXPECT_EQ(read_task(load_spec()).layout.anchors.size(), 9u);
}

/** @brief Tests that an anchor of the task section reads back as the YOLOv5 pixels over 640. */
TEST_P(Medium_Yolov5nAnchor, ReadsBackAsTheYOLOv5AnchorInPixels)
{
    const AnchorTask task = read_task(load_spec());
    const std::size_t anchor = GetParam();
    ASSERT_GT(task.layout.anchors.size(), anchor);

    EXPECT_NEAR(task.layout.anchors[anchor].width * K_INPUT_SIDE, K_ANCHOR_PIXELS[anchor][0],
                K_ANCHOR_TOLERANCE * K_INPUT_SIDE);
    EXPECT_NEAR(task.layout.anchors[anchor].height * K_INPUT_SIDE, K_ANCHOR_PIXELS[anchor][1],
                K_ANCHOR_TOLERANCE * K_INPUT_SIDE);
}

INSTANTIATE_TEST_SUITE_P(Anchors, Medium_Yolov5nAnchor,
                         ::testing::Values(0, 1, 2, 3, 4, 5, 6, 7, 8), anchor_name);

/** @brief Tests that the task section reads back as three heads. */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBackThreeHeads)
{
    EXPECT_EQ(read_task(load_spec()).heads.size(), 3u);
}

/** @brief Tests that a head reads the three anchors that follow the anchors of the heads before it. */
TEST_P(Medium_Yolov5nHead, ReadsBackItsThreeAnchors)
{
    const AnchorTask task = read_task(load_spec());
    const std::size_t head = GetParam();
    const auto first = static_cast<std::int64_t>(3 * head);

    ASSERT_GT(task.layout.head_anchors.size(), head);
    EXPECT_EQ(task.layout.head_anchors[head],
              (std::vector<std::int64_t>{first, first + 1, first + 2}));
}

/** @brief Tests that a head reads back the output it is named for. */
TEST_P(Medium_Yolov5nHead, ReadsBackItsOutput)
{
    const AnchorTask task = read_task(load_spec());
    ASSERT_GT(task.heads.size(), GetParam());

    EXPECT_EQ(task.heads[GetParam()].output, "head_" + std::to_string(GetParam()));
}

/** @brief Tests that a head reads back as logit coded. */
TEST_P(Medium_Yolov5nHead, ReadsBackAsLogitCoded)
{
    const AnchorTask task = read_task(load_spec());
    ASSERT_GT(task.heads.size(), GetParam());

    EXPECT_EQ(task.heads[GetParam()].options.coding, HeadCoding::Logit);
}

/** @brief Tests that a head reads back with squared-sigmoid sides. */
TEST_P(Medium_Yolov5nHead, ReadsBackWithSquaredSigmoidSides)
{
    const AnchorTask task = read_task(load_spec());
    ASSERT_GT(task.heads.size(), GetParam());

    EXPECT_EQ(task.heads[GetParam()].options.size,
              Vision::Detect::Anchor::SizeForm::SquaredSigmoid);
}

/** @brief Tests that a head reads back with a centre stretch of 2. */
TEST_P(Medium_Yolov5nHead, ReadsBackWithAStretchOfTwo)
{
    const AnchorTask task = read_task(load_spec());
    ASSERT_GT(task.heads.size(), GetParam());

    EXPECT_EQ(task.heads[GetParam()].options.scale_xy, 2.0f);
}

INSTANTIATE_TEST_SUITE_P(Heads, Medium_Yolov5nHead, ::testing::Values(0, 1, 2), head_name);

/** @brief Tests that the task section reads back the shape-ratio assignment rule. */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBackTheShapeRatioRule)
{
    EXPECT_EQ(read_task(load_spec()).assignment.rule, AssignmentRuleKind::ShapeRatio);
}

/** @brief Tests that the task section reads back the default ratio limit. */
TEST(Medium_Yolov5nFile, TheTaskSectionReadsBackTheDefaultRatioLimit)
{
    EXPECT_EQ(read_task(load_spec()).assignment.shape_ratio.ratio_limit,
              Vision::Detect::Anchor::K_DEFAULT_RATIO_LIMIT);
}

/** @brief Tests that the head of a level is named "<row>.bias" for the rows 26, 27 and 28. */
TEST_P(Small_Yolov5nBiasHead, NamesTheBiasOfItsRow)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);
    const std::vector<Vision::Training::AnchorBiasHead> heads = SaTest::require_value(
        Vision::Training::derive_anchor_bias_heads(spec, spec.input_shape, read_task(spec),
                                                   outputs));

    ASSERT_EQ(heads.size(), 3u);
    EXPECT_EQ(heads[GetParam()].bias, K_BIAS_NAMES[GetParam()]);
}

/** @brief Tests that the head of a level carries the stride of its level. */
TEST_P(Small_Yolov5nBiasHead, CarriesTheStrideOfItsLevel)
{
    const Config::ModelSpec spec = load_spec();
    ModelGraph model = build(spec);
    const std::vector<Train::OutputShape> outputs = trace_outputs(model, spec.input_shape);
    const std::vector<Vision::Training::AnchorBiasHead> heads = SaTest::require_value(
        Vision::Training::derive_anchor_bias_heads(spec, spec.input_shape, read_task(spec),
                                                   outputs));

    ASSERT_EQ(heads.size(), 3u);
    EXPECT_EQ(heads[GetParam()].stride, K_STRIDES[GetParam()]);
}

INSTANTIATE_TEST_SUITE_P(Levels, Small_Yolov5nBiasHead, ::testing::Values(0, 1, 2), level_name);

/**
 * @brief Tests that initialize_anchor_biases starts the objectness entry of each slot of a
 *        zeroed bias at log(8 / (640 / stride)^2).
 */
TEST_P(Small_Yolov5nBiasInit, StartsTheObjectnessEntryOfEachSlot)
{
    const std::vector<double> bias = biases_after_initialising(GetParam());
    const double expected = std::log(8.0 / std::pow(640.0 / K_STRIDES[GetParam()], 2));

    EXPECT_TRUE(all_near(slot_entries(bias, 4, 4), expected, K_BIAS_TOLERANCE));
}

/** @brief Tests that initialize_anchor_biases starts each class entry at log(0.6 / (80 - 0.99)). */
TEST_P(Small_Yolov5nBiasInit, StartsEachClassEntry)
{
    const std::vector<double> bias = biases_after_initialising(GetParam());

    EXPECT_TRUE(all_near(slot_entries(bias, 5, K_SLOT_ENTRIES - 1), std::log(0.6 / (80.0 - 0.99)),
                         K_BIAS_TOLERANCE));
}

/** @brief Tests that initialize_anchor_biases leaves the four box entries of each slot at zero. */
TEST_P(Small_Yolov5nBiasInit, LeavesTheBoxEntriesAtZero)
{
    const std::vector<double> bias = biases_after_initialising(GetParam());

    EXPECT_TRUE(all_near(slot_entries(bias, 0, 3), 0.0, K_BIAS_TOLERANCE));
}

INSTANTIATE_TEST_SUITE_P(Levels, Small_Yolov5nBiasInit, ::testing::Values(0, 1, 2), level_name);
