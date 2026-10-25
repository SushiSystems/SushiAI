/**************************************************************************/
/* test_free_bias_init.cpp                                                */
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
 * @file test_free_bias_init.cpp
 * @brief Tests the stride-derived start of a traced detect head's last biases.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/training/free_bias_init.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

using namespace SushiAI;
using Vision::Blocks::DetectHead;
using Vision::Training::derive_free_model_facts;
using Vision::Training::free_class_bias;
using Vision::Training::FreeModelFacts;
using Vision::Training::initialize_free_biases;

namespace
{
    /** @brief Runs each test on a fresh engine. */
    class Medium_TrainingFreeBiasInit : public SushiAI::Test::EngineTest
    {

    };

    /** @brief Holds the class count of the fixture. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Holds the batch extent of the trace. */
    constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the number of levels. */
    constexpr std::size_t K_LEVELS = 3;

    /** @brief Holds the H and the W of each level over a 64 by 64 input. */
    constexpr std::array<std::int64_t, K_LEVELS> K_SIDES = {8, 4, 2};

    /** @brief Holds the channels of each level's feature map. */
    constexpr std::array<std::int64_t, K_LEVELS> K_WIDTHS = {16, 32, 64};

    /** @brief Holds the strides the facts give: 64 divided by each side. */
    constexpr std::array<std::int64_t, K_LEVELS> K_STRIDES = {8, 16, 32};

    /** @brief Names the row the head's state hangs under. */
    constexpr std::string_view K_PREFIX = "23";

    /** @brief Holds the tolerance of a float32 bias against its double closed form. */
    constexpr double K_TOLERANCE = 1e-6;

    /** @brief Holds a traced head, its state and its facts. */
    struct TracedHead
    {
        std::unique_ptr<DetectHead> head;           /**< The head of both copies. */
        std::unique_ptr<NN::ModelState> state;      /**< Its state under K_PREFIX. */
        FreeModelFacts facts;                       /**< Facts of its ports. */
    };

    /** @brief Builds, traces and initialises a head over a 64 by 64 input. */
    Status trace_head(SushiBLAS::Engine& engine, TracedHead& traced)
    {
        std::vector<Shape> levels;
        for (std::size_t level = 0; level < K_LEVELS; ++level)
            levels.push_back(Shape{K_SIDES[level], K_SIDES[level], K_WIDTHS[level]});
        SA_TRY_ASSIGN(traced.head, DetectHead::create(NN::ShapeSpan(levels.data(), levels.size()),
                                                      K_CLASSES, NN::NormOptions{}));

        Graph::Graph graph;
        Graph::GraphBuilder builder(graph);
        std::vector<Graph::ValueID> features;
        for (std::size_t level = 0; level < K_LEVELS; ++level)
        {
            features.push_back(builder.input(
                Shape{K_BATCH, K_SIDES[level], K_SIDES[level], K_WIDTHS[level]}, "feature"));
        }
        std::vector<Graph::ValueID> outputs;
        SA_TRY(traced.head->forward(NN::Trace::evaluation(builder),
                                    NN::ValueSpan(features.data(), features.size()), outputs));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, *traced.head,
                                                                   K_PREFIX));
        traced.state = std::make_unique<NN::ModelState>(std::move(state));
        SA_TRY(NN::initialize(engine, *traced.state));

        std::vector<Train::OutputShape> published;
        for (const NN::Port& port : traced.head->ports())
        {
            published.push_back(Train::OutputShape{
                port.name, Shape{K_BATCH, port.shape.dim(0), port.shape.dim(1),
                                 port.shape.dim(2)}});
        }
        SA_TRY_ASSIGN(traced.facts, derive_free_model_facts(Shape{64, 64, 3}, published));
        return Status();
    }

    /** @brief Expects every element of entry @p name to be @p value. */
    void expect_bias(NN::ModelState& state, const std::string& name, double value)
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const tensor, state.tensor(name));
        const float* const data = tensor->data_as<float>();
        for (std::int64_t index = 0; index < tensor->num_elements; ++index)
            EXPECT_NEAR(static_cast<double>(data[index]), value, K_TOLERANCE) << name;
    }
} // namespace

/** @brief Tests that the class bias of stride 8 over 80 classes is log(5 / 80 / 6400). */
TEST(Small_TrainingFreeClassBias, MatchesTheClosedForm)
{
    EXPECT_NEAR(free_class_bias(80, 8), std::log(5.0 / 80.0 / 6400.0), 1e-12);
}

/** @brief Tests that both copies' last box biases are 1 and class biases follow the stride. */
TEST_F(Medium_TrainingFreeBiasInit, SetsTheLastBiasesOfBothCopies)
{
    TracedHead traced;
    ASSERT_TRUE(SushiAI::Test::status_is_ok(trace_head(*engine, traced)));
    flush();
    SA_EXPECT_OK(initialize_free_biases(traced.facts, K_PREFIX, *traced.state));

    for (std::size_t level = 0; level < K_LEVELS; ++level)
    {
        const std::string index = std::to_string(level);
        const double class_bias = std::log(5.0 / static_cast<double>(K_CLASSES) /
                                           std::pow(640.0 / K_STRIDES[level], 2.0));
        for (const std::string copy : {"", "one2one_"})
        {
            expect_bias(*traced.state, "23." + copy + "cv2." + index + ".2.bias", 1.0);
            expect_bias(*traced.state, "23." + copy + "cv3." + index + ".2.bias", class_bias);
        }
    }
}

/** @brief Tests that a prefix the state lacks is refused by entry and changes nothing. */
TEST_F(Medium_TrainingFreeBiasInit, RefusesAPrefixTheStateLacks)
{
    TracedHead traced;
    ASSERT_TRUE(SushiAI::Test::status_is_ok(trace_head(*engine, traced)));
    flush();
    SA_EXPECT_FAILURE(initialize_free_biases(traced.facts, "22", *traced.state),
                      StatusCode::InvalidDescription, "'22.cv2.0.2.bias'");
    expect_bias(*traced.state, "23.cv2.0.2.bias", 0.0);
}
