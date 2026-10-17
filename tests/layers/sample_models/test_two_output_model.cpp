/**************************************************************************/
/* test_two_output_model.cpp                                              */
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
 * @file test_two_output_model.cpp
 * @brief Tests the two outputs and the parameters of build_two_output_model().
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/sample_models/two_output_model.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows every trace reads. */
    constexpr int64_t K_ROWS = 3;
} // namespace

/** @brief Tests that a trace publishes "features" of width 8 and "logits" of width 3. */
TEST(Small_SampleModelsTwoOutput, ATracePublishesTheFeaturesAndTheLogits)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_two_output_model());
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_ROWS, 6}, "x");

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            model.trace(NN::Trace::evaluation(builder), x));

    ASSERT_EQ(outputs.all().size(), 2u);
    EXPECT_EQ(outputs.all()[0].first, "features");
    EXPECT_EQ(outputs.all()[1].first, "logits");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID features, outputs.at("features"));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID logits, outputs.at("logits"));
    EXPECT_EQ(graph.value(features).shape, (Shape{K_ROWS, 8}));
    EXPECT_EQ(graph.value(logits).shape, (Shape{K_ROWS, 3}));
}

/** @brief Tests that the two linear layers hold the four parameters, and the GELU none. */
TEST(Small_SampleModelsTwoOutput, TheLinearLayersHoldTheFourParameters)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_two_output_model());
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_ROWS, 6}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelOutputs outputs,
                            model.trace(NN::Trace::evaluation(builder), x));
    static_cast<void>(outputs);

    std::vector<std::string> shapes;
    const auto on_parameter = [&shapes](const std::string&, NN::Parameter& parameter)
    {
        shapes.push_back(parameter.shape().to_string());
    };
    const auto on_buffer = [](const std::string&, NN::Buffer&)
    {

    };
    model.walk_state("model", NN::StateVisitor(on_parameter, on_buffer));

    EXPECT_EQ(shapes, (std::vector<std::string>{"[8, 6]", "[8]", "[3, 8]", "[3]"}));
}
