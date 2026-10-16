/**************************************************************************/
/* test_trace_summary.cpp                                                 */
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
 * @file test_trace_summary.cpp
 * @brief Tests the state records and the summary trace_summary() writes of a training trace.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/layer_traces/trace_summary.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns whether @p text ends with @p suffix. */
    bool ends_with(const std::string& text, const std::string& suffix)
    {
        return text.size() >= suffix.size() &&
               text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
    }
} // namespace

/** @brief Tests that a record writes the name, the extents and the kind of the state. */
TEST(Small_LayerTracesSummary, AStateRecordNamesItsKind)
{
    EXPECT_EQ(SaTest::state_record("0.weight", Shape{4, 3}, false), "0.weight [4, 3] parameter");
    EXPECT_EQ(SaTest::state_record("1.running_mean", Shape{8}, true), "1.running_mean [8] buffer");
}

/** @brief Tests that the summary of a linear model lists its ops, its output and its state. */
TEST(Small_LayerTracesSummary, TheSummaryListsTheOpsTheOutputAndTheState)
{
    NN::ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef head,
                            model.add("head", NN::Linear{6, 4}, model.input()));
    SA_EXPECT_OK(model.output("logits", head));

    SA_ASSERT_OK_AND_ASSIGN(const SaTest::TraceSummary summary,
                            SaTest::trace_summary(model, Shape{6}, "model"));

    EXPECT_EQ(summary.outputs, (std::vector<std::string>{"logits [2, 4]"}));
    ASSERT_EQ(summary.state.size(), 2u);
    EXPECT_TRUE(ends_with(summary.state[0], "weight [4, 6] parameter")) << summary.state[0];
    EXPECT_TRUE(ends_with(summary.state[1], "bias [4] parameter")) << summary.state[1];
    EXPECT_EQ(summary.state, SaTest::state_records(model, "model"));

    EXPECT_NE(std::find(summary.ops.begin(), summary.ops.end(), Ops::K_MATMUL.id),
              summary.ops.end());
}

/** @brief Tests that the input of a summary carries K_TRACE_BATCH rows in front of the sample. */
TEST(Small_LayerTracesSummary, TheTraceRunsOverTheFixedBatch)
{
    NN::ModelGraph model(Shape{2, 2, 3});
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef flat,
                            model.add("flat", NN::Flatten{}, model.input()));
    SA_EXPECT_OK(model.output("flat", flat));

    SA_ASSERT_OK_AND_ASSIGN(const SaTest::TraceSummary summary,
                            SaTest::trace_summary(model, Shape{2, 2, 3}, "model"));

    EXPECT_EQ(summary.outputs, (std::vector<std::string>{"flat [2, 12]"}));
    EXPECT_TRUE(summary.state.empty());
    EXPECT_EQ(SaTest::K_TRACE_BATCH, 2);
}
