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
 * @file test_trace_summary.cpp
 * @brief Tests the state records and the summary trace_summary() writes of a training trace.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
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

namespace
{
    /** @brief Holds a state entry and the record written for it. */
    struct RecordRow
    {
        std::string name;         /**< The row's name. */
        std::string entry;        /**< The dotted name of the state. */
        Shape shape;              /**< The extents of the state. */
        bool buffer;              /**< Whether the state is a buffer, not a parameter. */
        std::string record;       /**< The record written for it. */
    };

    /** @brief Runs the state record over a parameter and a buffer. */
    class Small_LayerTracesStateRecord : public ::testing::TestWithParam<RecordRow>
    {
    };

    /** @brief Returns the summary of a model of one 6-to-4 linear layer publishing "logits". */
    SaTest::TraceSummary summary_of_linear_model()
    {
        NN::ModelGraph model(Shape{6});
        const NN::NodeRef head = SaTest::require_value(
            model.add("head", NN::Linear{6, 4}, model.input()));
        SA_EXPECT_OK(model.output("logits", head));

        return SaTest::require_value(SaTest::trace_summary(model, Shape{6}, "model"));
    }
} // namespace

/** @brief Tests that a record writes the name, the extents and the kind of the state. */
TEST_P(Small_LayerTracesStateRecord, AStateRecordNamesItsKind)
{
    const RecordRow& row = GetParam();

    EXPECT_EQ(SaTest::state_record(row.entry, row.shape, row.buffer), row.record);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LayerTracesStateRecord,
    ::testing::Values(RecordRow{"AParameter", "0.weight", Shape{4, 3}, false,
                                "0.weight [4, 3] parameter"},
                      RecordRow{"ABuffer", "1.running_mean", Shape{8}, true,
                                "1.running_mean [8] buffer"}),
    [](const ::testing::TestParamInfo<RecordRow>& info) { return info.param.name; });

/** @brief Tests that the summary of a linear model lists its output with the batch in front. */
TEST(Small_LayerTracesSummary, TheSummaryListsTheOutputs)
{
    EXPECT_EQ(summary_of_linear_model().outputs, (std::vector<std::string>{"logits [2, 4]"}));
}

/** @brief Tests that the summary of a linear model lists its weight and its bias. */
TEST(Small_LayerTracesSummary, TheSummaryListsTheParameters)
{
    const SaTest::TraceSummary summary = summary_of_linear_model();

    ASSERT_EQ(summary.state.size(), 2u);
    EXPECT_TRUE(ends_with(summary.state[0], "weight [4, 6] parameter")) << summary.state[0];
    EXPECT_TRUE(ends_with(summary.state[1], "bias [4] parameter")) << summary.state[1];
}

/** @brief Tests that the input of a summary carries the batch rows in front of the sample. */
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
}
