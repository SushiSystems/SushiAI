/**************************************************************************/
/* test_anchor_bias_init.cpp                                              */
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
 * @file test_anchor_bias_init.cpp
 * @brief Tests the stride-derived start of an anchor head's objectness and class biases.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Vision::Training::AnchorBiasHead;
using Vision::Training::anchor_class_bias;
using Vision::Training::anchor_objectness_bias;
using Vision::Training::derive_anchor_bias_heads;
using Vision::Training::initialize_anchor_biases;
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Holds the absolute Bounds the error of a float32 bias against the double formula: 256
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1e-5.
     */
    constexpr double K_BIAS_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Holds the absolute Bounds the error of the double formula against a value computed
     *        offline: 16384 times the double unit roundoff, the smallest power of two not tighter
     *        than the earlier 1e-12.
     */
    constexpr double K_FORMULA_TOLERANCE = 16384.0 * TestSupport::unit_roundoff<double>();

    /** @brief Holds the anchor slots of each head of the cases. */
    constexpr std::int64_t K_SLOTS = 2;

    /** @brief Holds the value every bias starts from, so an addition shows. */
    constexpr double K_START = 1.0;

    /**
     * @brief Holds a model file whose head_0 is a sigmoid over row 0 and whose head_1 is row 2
     *        itself, both of 14 channels.
     */
    constexpr const char* K_MODEL = R"({
        "format_version": 2,
        "input": { "shape": [64, 64, 3] },
        "layers": [
            { "type": "conv2d", "filters": 14, "kernel": 1, "use_bias": true },
            { "type": "sigmoid" },
            { "type": "conv2d", "from": 0, "filters": 14, "kernel": 1 }
        ],
        "outputs": [
            { "node": 1, "port": "output", "name": "head_0" },
            { "node": 2, "port": "output", "name": "head_1" }
        ]
    })";

    /** @brief Holds a model file whose only head reads a convolution without a bias. */
    constexpr const char* K_UNBIASED_MODEL = R"({
        "format_version": 2,
        "input": { "shape": [64, 64, 3] },
        "layers": [ { "type": "conv2d", "filters": 14, "kernel": 1, "use_bias": false } ],
        "outputs": [ { "node": 0, "port": "output", "name": "head_0" } ]
    })";

    /** @brief Holds a model file whose head ends in a row that is no convolution. */
    constexpr const char* K_POOLED_MODEL = R"({
        "format_version": 2,
        "input": { "shape": [64, 64, 3] },
        "layers": [ { "type": "maxpool2d", "kernel": 2, "stride": 2 } ],
        "outputs": [ { "node": 0, "port": "output", "name": "head_0" } ]
    })";

    /** @brief Holds a task of two classes and two heads of two anchors each. */
    constexpr const char* K_TASK = R"({
        "type": "anchor_detection",
        "classes": 2,
        "anchors": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
        "heads": [
            { "output": "head_0", "anchors": [0, 1], "coding": "activated", "scale_xy": 2.0 },
            { "output": "head_1", "anchors": [2, 3], "coding": "logit", "scale_xy": 1.0 }
        ],
        "suppression": { "criterion": "iou" }
    })";

    /** @brief Reads the model file @p text. */
    Config::ModelSpec spec_from(const char* text)
    {
        const Config::Value root = SaTest::require_value(Config::parse_json(text, "<test>"));
        return SaTest::require_value(Config::parse_model(root, "<test>"));
    }

    /** @brief Reads the task of the cases. */
    Vision::Detect::Anchor::AnchorTask task_of_the_cases()
    {
        const Config::Value root = SaTest::require_value(Config::parse_json(K_TASK, "<test>"));
        return SaTest::require_value(Vision::Detect::Anchor::read_anchor_task(root));
    }

    /** @brief Returns the outputs of the cases: strides 4 and 8 over the 64 pixel input. */
    std::vector<Train::OutputShape> outputs_of_the_cases()
    {
        return {Train::OutputShape{"head_0", Shape{2, 16, 16, 14}},
                Train::OutputShape{"head_1", Shape{2, 8, 8, 14}}};
    }

    /** @brief Describes a float32 entry of the extents @p shape for from_entries. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, Shape shape)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = shape;
        return entry;
    }

    /** @brief Sets every element of the entry @p name of @p state to K_START. */
    void start_entry(NN::ModelState& state, const std::string& name)
    {
        SushiBLAS::Tensor& tensor = *state.tensor(name).value();
        const std::vector<double> values(static_cast<std::size_t>(tensor.num_elements), K_START);
        SaTest::write_tensor(tensor, values);
    }

    /** @brief Reads the entry @p name of @p state. */
    std::vector<double> read_entry(NN::ModelState& state, const std::string& name)
    {
        return SaTest::read_tensor(*state.tensor(name).value());
    }

    /** @brief Returns the state of both heads, started at K_START and then initialised. */
    Result<NN::ModelState> initialized_state(SushiBLAS::Engine& engine)
    {
        SA_TRY_ASSIGN(NN::ModelState state,
                      NN::ModelState::from_entries(engine, {make_entry("0.bias", 1, Shape{14}),
                                                            make_entry("2.bias", 2, Shape{14})}));
        start_entry(state, "0.bias");
        start_entry(state, "2.bias");
        const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4},
                                                   AnchorBiasHead{"2.bias", 8}};
        SA_TRY(initialize_anchor_biases(task_of_the_cases(), heads, state));
        return state;
    }

    /**
     * @brief Returns the bias entry of a head of the given stride after the start: the box
     *        entries at K_START, the objectness and the two class entries raised by their bias.
     */
    std::vector<double> expected_bias(std::int64_t stride)
    {
        std::vector<double> bias;
        for (std::int64_t slot = 0; slot < K_SLOTS; ++slot)
        {
            bias.insert(bias.end(), 4, K_START);
            bias.push_back(K_START + anchor_objectness_bias(stride));
            bias.push_back(K_START + anchor_class_bias(2));
            bias.push_back(K_START + anchor_class_bias(2));
        }
        return bias;
    }

    /** @brief Expects every entry of @p actual within the float32 bound of @p expected. */
    void expect_biases_equal(const std::vector<double>& actual, const std::vector<double>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], K_BIAS_TOLERANCE) << "entry " << i;
    }

    /** @brief Provides the engine the states of the Medium cases are allocated on. */
    class Medium_AnchorBiasInit : public SaTest::EngineTest {};
} // namespace

/** @brief Tests that the objectness bias is log(8 / (640 / stride)^2) at strides 8, 16 and 32. */
TEST(Small_AnchorBiasInit, TheObjectnessBiasFollowsTheStride)
{
    EXPECT_NEAR(anchor_objectness_bias(8), -6.684611727667927, K_FORMULA_TOLERANCE);
    EXPECT_NEAR(anchor_objectness_bias(16), -5.298317366548036, K_FORMULA_TOLERANCE);
    EXPECT_NEAR(anchor_objectness_bias(32), -3.912023005428146, K_FORMULA_TOLERANCE);
}

/** @brief Tests that the class bias is log(0.6 / (classes - 0.99)) for 80 and 2 classes. */
TEST(Small_AnchorBiasInit, TheClassBiasFollowsTheClassCount)
{
    EXPECT_NEAR(anchor_class_bias(80), -4.880400050500633, K_FORMULA_TOLERANCE);
    EXPECT_NEAR(anchor_class_bias(2), -0.5207759546191588, K_FORMULA_TOLERANCE);
}

/**
 * @brief Tests that a head over a sigmoid and a head that is a convolution each name the bias of
 *        their convolution, with the stride the output's extent gives.
 */
TEST(Small_AnchorBiasInit, DerivesTheBiasEntryAndTheStrideOfEachHead)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::vector<AnchorBiasHead> heads,
        derive_anchor_bias_heads(spec_from(K_MODEL), Shape{64, 64, 3}, task_of_the_cases(),
                                 outputs_of_the_cases()));

    ASSERT_EQ(heads.size(), 2u);
    EXPECT_EQ(heads[0].bias, "0.bias");
    EXPECT_EQ(heads[0].stride, 4);
    EXPECT_EQ(heads[1].bias, "2.bias");
    EXPECT_EQ(heads[1].stride, 8);
}

/** @brief Tests that a head whose convolution has no bias is refused and the row is named. */
TEST(Small_AnchorBiasInit, ARowWithoutABiasIsRefused)
{
    const std::vector<Train::OutputShape> outputs = {
        Train::OutputShape{"head_0", Shape{2, 16, 16, 14}}};
    Vision::Detect::Anchor::AnchorTask task = task_of_the_cases();
    task.heads.resize(1);

    SA_EXPECT_FAILURE(derive_anchor_bias_heads(spec_from(K_UNBIASED_MODEL), Shape{64, 64, 3},
                                               task, outputs),
                      StatusCode::InvalidDescription, "row 0", "no bias");
}

/** @brief Tests that a head that ends in a row other than a convolution is refused. */
TEST(Small_AnchorBiasInit, ARowThatIsNoConvolutionIsRefused)
{
    const std::vector<Train::OutputShape> outputs = {
        Train::OutputShape{"head_0", Shape{2, 32, 32, 3}}};
    Vision::Detect::Anchor::AnchorTask task = task_of_the_cases();
    task.heads.resize(1);

    SA_EXPECT_FAILURE(derive_anchor_bias_heads(spec_from(K_POOLED_MODEL), Shape{64, 64, 3},
                                               task, outputs),
                      StatusCode::InvalidDescription, "head_0", "maxpool2d");
}

/** @brief Tests that an output the model does not publish is refused and named. */
TEST(Small_AnchorBiasInit, AnUnpublishedOutputIsRefused)
{
    const std::vector<Train::OutputShape> outputs = {
        Train::OutputShape{"head_0", Shape{2, 16, 16, 14}}};

    SA_EXPECT_FAILURE(derive_anchor_bias_heads(spec_from(K_MODEL), Shape{64, 64, 3},
                                               task_of_the_cases(), outputs),
                      StatusCode::InvalidDescription, "head_1");
}

/** @brief Tests that a level whose extent does not divide the input has no stride. */
TEST(Small_AnchorBiasInit, AnExtentThatDoesNotDivideTheInputIsRefused)
{
    const std::vector<Train::OutputShape> outputs = {
        Train::OutputShape{"head_0", Shape{2, 16, 16, 14}},
        Train::OutputShape{"head_1", Shape{2, 7, 7, 14}}};

    SA_EXPECT_FAILURE(derive_anchor_bias_heads(spec_from(K_MODEL), Shape{64, 64, 3},
                                               task_of_the_cases(), outputs),
                      StatusCode::InvalidDescription, "head_1", "stride");
}

/** @brief Tests that the first head gains the biases of stride 4 and keeps its box entries. */
TEST_F(Medium_AnchorBiasInit, TheFirstHeadGainsTheBiasesOfItsStride)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, initialized_state(*engine));

    expect_biases_equal(read_entry(state, "0.bias"), expected_bias(4));
}

/** @brief Tests that the second head gains the biases of stride 8 and keeps its box entries. */
TEST_F(Medium_AnchorBiasInit, TheSecondHeadGainsTheBiasesOfItsStride)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, initialized_state(*engine));

    expect_biases_equal(read_entry(state, "2.bias"), expected_bias(8));
}

/** @brief Tests that an entry of the wrong extent refuses the call and names the entry. */
TEST_F(Medium_AnchorBiasInit, AnEntryOfTheWrongExtentIsRefusedAndNamed)
{
    SA_ASSERT_OK_AND_ASSIGN(
        NN::ModelState state,
        NN::ModelState::from_entries(*engine, {make_entry("0.bias", 1, Shape{14}),
                                               make_entry("2.bias", 2, Shape{13})}));
    start_entry(state, "0.bias");
    start_entry(state, "2.bias");
    const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4},
                                               AnchorBiasHead{"2.bias", 8}};

    SA_EXPECT_FAILURE(initialize_anchor_biases(task_of_the_cases(), heads, state),
                      StatusCode::InvalidDescription, "2.bias", "[14]");
}

/** @brief Tests that the refusal of an entry of the wrong extent changes no bias. */
TEST_F(Medium_AnchorBiasInit, AnEntryOfTheWrongExtentLeavesEveryBiasUnchanged)
{
    SA_ASSERT_OK_AND_ASSIGN(
        NN::ModelState state,
        NN::ModelState::from_entries(*engine, {make_entry("0.bias", 1, Shape{14}),
                                               make_entry("2.bias", 2, Shape{13})}));
    start_entry(state, "0.bias");
    start_entry(state, "2.bias");
    const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4},
                                               AnchorBiasHead{"2.bias", 8}};

    EXPECT_FALSE(SaTest::status_is_ok(initialize_anchor_biases(task_of_the_cases(), heads, state)));

    EXPECT_EQ(read_entry(state, "0.bias"), (std::vector<double>(14, K_START)));
}

/** @brief Tests that an entry the state does not hold is refused and named. */
TEST_F(Medium_AnchorBiasInit, AnAbsentEntryIsRefusedAndNamed)
{
    SA_ASSERT_OK_AND_ASSIGN(
        NN::ModelState state,
        NN::ModelState::from_entries(*engine, {make_entry("0.bias", 1, Shape{14})}));
    const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4},
                                               AnchorBiasHead{"2.bias", 8}};

    SA_EXPECT_FAILURE(initialize_anchor_biases(task_of_the_cases(), heads, state),
                      StatusCode::InvalidDescription, "2.bias");
}

/** @brief Tests that a head list of another length than the task's is refused. */
TEST_F(Medium_AnchorBiasInit, AHeadCountThatDiffersFromTheTaskIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(
        NN::ModelState state,
        NN::ModelState::from_entries(*engine, {make_entry("0.bias", 1, Shape{14})}));
    const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4}};

    SA_EXPECT_FAILURE(initialize_anchor_biases(task_of_the_cases(), heads, state),
                      StatusCode::InvalidDescription, "1 bias heads", "2 heads");
}
