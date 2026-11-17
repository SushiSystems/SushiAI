/**************************************************************************/
/* test_end_padding.cpp                                        */
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
 * @file test_end_padding.cpp
 * @brief Tests the leading and end paddings of the max-pooling layer and its row keys.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/gradient_probe.hpp>
#include <support/host_row_values.hpp>
#include <support/test_names.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using NN::MaxPool2d;
namespace SaTest = SushiAI::Test;

namespace
{
    using SaTest::require_value;

    /** @brief Builds the layer of a maxpool2d row over @p input with @p attributes as JSON. */
    Result<std::unique_ptr<NN::Layer>> make_row(const Shape& input, const std::string& attributes)
    {
        const Config::Value row = require_value(Config::parse_json(attributes, "<test>"));

        return NN::make_maxpool2d(input, row, "layer 2 'maxpool2d'");
    }

    /** @brief Holds one row with the per-sample shape it gives over [5, 5, 3]. */
    struct ExtentRow
    {
        std::string name;       /**< The row's name. */
        std::string attributes; /**< The row's keys as JSON. */
        Shape expected;         /**< The per-sample result. */
    };

    /** @brief Names an ExtentRow for gtest. */
    std::string extent_name(const ::testing::TestParamInfo<ExtentRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each extent row. */
    class Small_MaxPool2dEndPaddingExtent : public ::testing::TestWithParam<ExtentRow>
    {

    };
} // namespace

/** @brief Tests that the leading and end paddings give the window count their sum gives. */
TEST_P(Small_MaxPool2dEndPaddingExtent, TheExtentFollowsTheLeadingAndTheEndPadding)
{
    const ExtentRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Layer> layer,
                            make_row(Shape{5, 5, 3}, row.attributes));

    EXPECT_EQ(layer->output_shape(), row.expected);
}

/**
 * Over 5 rows with a window of 3 at stride 1, the count is 5 + leading + end - 3 + 1, so the pads
 * 1 and 1 give 5, the pads 1 and 0 give 4, the pads 0 and 2 give 5, and the pads 2 and 0 give 5.
 * The second axis is worked the same way from its own pair.
 */
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MaxPool2dEndPaddingExtent,
    ::testing::Values(
        ExtentRow{"EqualPadsAreTheSymmetricPad",
                  R"({"kernel": 3, "stride": 1, "leading_padding": 1, "end_padding": 1})",
                  Shape{5, 5, 3}},
        ExtentRow{"NoPadAtTheEndDropsAWindow",
                  R"({"kernel": 3, "stride": 1, "leading_padding": 1, "end_padding": 0})",
                  Shape{4, 4, 3}},
        ExtentRow{"MorePadAtTheEndThanAtTheStart",
                  R"({"kernel": 3, "stride": 1, "leading_padding": 0, "end_padding": 2})",
                  Shape{5, 5, 3}},
        ExtentRow{"MorePadAtTheStartThanAtTheEnd",
                  R"({"kernel": 3, "stride": 1, "leading_padding": 2, "end_padding": 0})",
                  Shape{5, 5, 3}},
        ExtentRow{"PadsThatDifferPerAxis",
                  R"({"kernel": 3, "stride": 1, "leading_padding": [1, 0], "end_padding": [0, 2]})",
                  Shape{4, 5, 3}}),
    extent_name);

namespace
{
    /** @brief Holds one row the factory refuses and the words the message holds. */
    struct RefusalRow
    {
        std::string name;       /**< The row's name. */
        std::string attributes; /**< The row's keys as JSON. */
        std::string fragment;   /**< What the message holds besides the context. */
    };

    /** @brief Names a RefusalRow for gtest. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each refusal row. */
    class Small_MaxPool2dEndPaddingRefusal : public ::testing::TestWithParam<RefusalRow>
    {

    };
} // namespace

/** @brief Tests that the factory refuses a row that states the paddings in two forms or in half. */
TEST_P(Small_MaxPool2dEndPaddingRefusal, TheFactoryRefusesTheRow)
{
    const RefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(make_row(Shape{5, 5, 3}, row.attributes), StatusCode::InvalidDescription,
                      "layer 2 'maxpool2d'", row.fragment);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_MaxPool2dEndPaddingRefusal,
    ::testing::Values(
        RefusalRow{"OnlyTheLeadingPadding", R"({"kernel": 2, "leading_padding": 1})",
                   "needs both or neither"},
        RefusalRow{"OnlyTheEndPadding", R"({"kernel": 2, "end_padding": 1})",
                   "needs both or neither"},
        RefusalRow{"BesideThePaddingKey",
                   R"({"kernel": 2, "padding": 1, "leading_padding": 1, "end_padding": 0})",
                   "takes one form of the two"},
        RefusalRow{"BesideTheTrailingPaddingKey",
                   R"({"kernel": 2, "trailing_padding": 1, "leading_padding": 1, "end_padding": 1})",
                   "takes one form of the two"},
        RefusalRow{"ANegativeLeadingPadding",
                   R"({"kernel": 2, "leading_padding": -1, "end_padding": 0})",
                   "0 or more"},
        RefusalRow{"ANegativeEndPadding",
                   R"({"kernel": 2, "leading_padding": 0, "end_padding": [0, -1]})",
                   "0 or more"},
        RefusalRow{"ANegativeTrailingPaddingKey", R"({"kernel": 2, "trailing_padding": -1})",
                   "non-negative trailing padding"}),
    refusal_name);

namespace
{
    /**
     * @brief Returns a 2 by 2 window at stride one with a leading pad and a signed end pad.
     * @param end_pad The rows and columns by which the pad below and right exceeds the leading
     *        pad; negative when it falls short of it.
     */
    MaxPool2d make_pool(int64_t leading, int64_t end_pad)
    {
        Graph::ConvGeometry window;
        window.kernel_h = 2;
        window.kernel_w = 2;
        window.pad_top = leading;
        window.pad_left = leading;
        MaxPool2d layer{window};
        layer.end_pad_h = end_pad;
        layer.end_pad_w = end_pad;
        return layer;
    }

    /** @brief Holds the 3 by 3 input of the value cases, 9 1 2 over 3 8 4 over 5 6 7. */
    std::vector<double> value_input()
    {
        return {9, 1, 2, 3, 8, 4, 5, 6, 7};
    }

    /** @brief Pools the 3 by 3 input with @p layer on the host and returns the result. */
    Result<std::vector<double>> pool_on_host(const MaxPool2d& layer)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{1, 3, 3, 1}, "x", DType::Float64);
        SA_TRY_ASSIGN(const Graph::ValueID y, layer.forward(builder, x));

        return SaTest::evaluate_on_host(graph, {{x, value_input()}}, y);
    }

    /** @brief Expects every element of @p got to equal @p expected exactly. */
    void expect_exact(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_TRUE(TestSupport::is_within(
                TestSupport::measure_error(got[index], expected[index], 0.0)))
                << "element " << index;
    }
} // namespace

/**
 * @brief Tests the maxima of a window of two with a pad above and left of one and none below and
 *        right, worked by hand.
 *
 * Row i reads the input rows i - 1 and i that exist, and column j reads the columns j - 1 and j. Row
 * 0 reads {9, 1, 2} alone and gives 9, 9, 2; row 1 reads the first two rows and gives 9, 9, 8; row 2
 * reads the last two and gives 5, 8, 8.
 */
TEST(Small_MaxPool2dEndPadding, ALeadingPadWithNoEndPadDropsTheWindowsPastTheEnd)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got, pool_on_host(make_pool(1, -1)));

    expect_exact(got, {9, 9, 2, 9, 9, 8, 5, 8, 8});
}

/**
 * @brief Tests the maxima of a window of two with no leading pad and one below and right.
 *
 * Row i reads the input rows i and i + 1 that exist, and column j the columns j and j + 1. Row 0 gives
 * 9, 8, 4, row 1 gives 8, 8, 7 and row 2, which reads {5, 6, 7} alone, gives 6, 7, 7.
 */
TEST(Small_MaxPool2dEndPadding, AnEndPadWithNoLeadingPadKeepsTheLastWindows)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> got, pool_on_host(make_pool(0, 1)));

    expect_exact(got, {9, 8, 4, 8, 8, 7, 6, 7, 7});
}

/** @brief Tests that the row keys build the layer the fields describe. */
TEST(Small_MaxPool2dEndPadding, TheRowKeysSetTheLeadingPadAndTheSignedEndPad)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Layer> shortfall,
        make_row(Shape{3, 3, 1},
                 R"({"kernel": 2, "stride": 1, "leading_padding": 1, "end_padding": 0})"));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Layer> trailing,
        make_row(Shape{3, 3, 1},
                 R"({"kernel": 2, "stride": 1, "leading_padding": 0, "end_padding": 1})"));

    EXPECT_EQ(shortfall->output_shape(), (Shape{3, 3, 1}));
    EXPECT_EQ(trailing->output_shape(), (Shape{3, 3, 1}));
}

/** @brief Tests that an end pad below minus the leading pad is refused. */
TEST(Small_MaxPool2dEndPadding, AnEndPadBelowMinusTheLeadingPadIsRefused)
{
    SA_EXPECT_FAILURE(make_pool(1, -2).output_shape(Shape{3, 3, 1}),
                      StatusCode::InvalidDescription, "MaxPool2d", "end padding of 0 or more");
}

namespace
{
    /** @brief Provides the Engine every gradient case records on. */
    class Medium_MaxPool2dEndPaddingGradient : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the gradient of a pool with a negative end pad matches central differences. */
TEST_F(Medium_MaxPool2dEndPaddingGradient, TheGradientMatchesCentralDifferences)
{
    const MaxPool2d layer = make_pool(1, -1);

    SaTest::GradientProbe probe(backend(), Shape{1, 3, 3, 1},
                                [&](const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        const Graph::ValueID y = layer.forward(builder, x).value();
        return builder.reshape(y, Shape{1, 9});
    });

    std::vector<double> point = value_input();
    SaTest::expect_gradient_matches(probe, point);
}
