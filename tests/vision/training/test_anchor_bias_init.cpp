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
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>

using namespace SushiAI;
using Vision::Training::AnchorBiasHead;
using Vision::Training::anchor_class_bias;
using Vision::Training::anchor_objectness_bias;
using Vision::Training::derive_anchor_bias_heads;
using Vision::Training::initialize_anchor_biases;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the error of a float32 bias against the double formula. */
    constexpr double K_BIAS_TOLERANCE = 1e-5;

    /** @brief Bounds the error of the double formula against a value computed offline. */
    constexpr double K_FORMULA_TOLERANCE = 1e-12;

    /** @brief Holds the entries of one anchor slot with two classes. */
    constexpr std::int64_t K_SLOT_ENTRIES = 7;

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

/**
 * @brief Tests that the objectness entry and the class entries of both slots of both heads gain
 *        the bias of their stride and the box entries stay as they were.
 */
TEST_F(Medium_AnchorBiasInit, AddsBothBiasesToEverySlotOfEveryHead)
{
    SA_ASSERT_OK_AND_ASSIGN(
        NN::ModelState state,
        NN::ModelState::from_entries(*engine, {make_entry("0.bias", 1, Shape{14}),
                                               make_entry("2.bias", 2, Shape{14})}));
    start_entry(state, "0.bias");
    start_entry(state, "2.bias");
    const std::vector<AnchorBiasHead> heads = {AnchorBiasHead{"0.bias", 4},
                                               AnchorBiasHead{"2.bias", 8}};

    SA_EXPECT_OK(initialize_anchor_biases(task_of_the_cases(), heads, state));

    const std::int64_t strides[] = {4, 8};
    const std::string names[] = {"0.bias", "2.bias"};
    for (std::size_t head = 0; head < 2; ++head)
    {
        SCOPED_TRACE(names[head]);
        const std::vector<double> bias = read_entry(state, names[head]);
        ASSERT_EQ(bias.size(), 14u);
        for (std::int64_t slot = 0; slot < K_SLOTS; ++slot)
        {
            const auto first = static_cast<std::size_t>(slot * K_SLOT_ENTRIES);
            for (std::size_t entry = 0; entry < 4; ++entry)
                EXPECT_EQ(bias[first + entry], K_START);
            EXPECT_NEAR(bias[first + 4], K_START + anchor_objectness_bias(strides[head]),
                        K_BIAS_TOLERANCE);
            EXPECT_NEAR(bias[first + 5], K_START + anchor_class_bias(2), K_BIAS_TOLERANCE);
            EXPECT_NEAR(bias[first + 6], K_START + anchor_class_bias(2), K_BIAS_TOLERANCE);
        }
    }
}

/** @brief Tests that an entry of the wrong extent refuses the call and changes no bias. */
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

    SA_EXPECT_FAILURE(initialize_anchor_biases(task_of_the_cases(), heads, state),
                      StatusCode::InvalidDescription, "2.bias", "[14]");

    for (const double value : read_entry(state, "0.bias"))
        EXPECT_EQ(value, K_START);
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
