/**************************************************************************/
/* test_position_embedding.cpp                                            */
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
 * @file test_position_embedding.cpp
 * @brief Tests the sum, the state, the refusals and the node factory of PositionEmbedding.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include "blocks_test_support.hpp"

using namespace SushiAI;
using Language::Blocks::PositionEmbedding;
using NN::Trace;
namespace SaTest = SushiAI::Test;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the width of the sequence every case reads. */
    constexpr int64_t K_DIM = 6;

    /** @brief Holds the positions of the sequence every case reads. */
    constexpr int64_t K_LENGTH = 5;

    /** @brief Holds the samples of the sequence the numeric cases read. */
    constexpr int64_t K_SAMPLES = 2;

    /** @brief Returns @p x plus row t of @p table at every position t of every sample. */
    std::vector<double> added_on_host(const std::vector<double>& x,
                                      const std::vector<double>& table)
    {
        const std::size_t per_sample = static_cast<std::size_t>(K_LENGTH * K_DIM);
        std::vector<double> sum(x.size());
        for (std::size_t at = 0; at < x.size(); ++at)
            sum[at] = x[at] + table[at % per_sample];

        return sum;
    }

    /** @brief Holds one block the trace refuses and two words of its message. */
    struct BlockRow
    {
        std::string name;   /**< The row's name. */
        int64_t length;     /**< The table's rows. */
        int64_t dim;        /**< The table's width. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused block. */
    class Small_BlockPositionEmbeddingTable : public ::testing::TestWithParam<BlockRow>
    {

    };

    /** @brief Holds one factory input the factory refuses and two words of its message. */
    struct FactoryRow
    {
        std::string name;   /**< The row's name. */
        Shape input;        /**< The per-sample shape. */
        std::string json;   /**< The row's attributes. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message, or empty. */
    };

    /** @brief Runs one refused factory row. */
    class Small_BlockPositionEmbeddingFactory : public ::testing::TestWithParam<FactoryRow>
    {

    };

    /** @brief Holds one incremental input the table cannot serve and two words of its message. */
    struct WindowRow
    {
        std::string name;   /**< The row's name. */
        Shape slice;        /**< The traced slice's shape. */
        Shape sample;       /**< The per-sample shape of the window. */
        std::string first;  /**< A word of the message. */
        std::string second; /**< A second word of the message. */
    };

    /** @brief Runs one refused window. */
    class Small_BlockPositionEmbeddingWindow : public ::testing::TestWithParam<WindowRow>
    {

    };

    /** @brief Names a row by its name field. */
    template<typename Row>
    std::string row_name(const ::testing::TestParamInfo<Row>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a table of T rows adds row t to position t of every sample. */
TEST(Small_BlockPositionEmbedding, ATraceAddsTheTableToEveryRow)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_SAMPLES, K_LENGTH, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);
    const std::vector<double> inputs = SaTest::eighths(K_SAMPLES * K_LENGTH * K_DIM, 3);
    const std::vector<double> table = SaTest::eighths(K_LENGTH * K_DIM, 0);
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, inputs));
    SA_EXPECT_OK(evaluator.write(block.weight.id_in(graph), table));

    SA_EXPECT_OK(evaluator.run());

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{K_SAMPLES, K_LENGTH, K_DIM}));
    EXPECT_EQ(block.weight.shape(), (Shape{K_LENGTH, K_DIM}));
    EXPECT_EQ(block.weight.role(), NN::ParameterRole::Embedding);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> sum, evaluator.read(y));
    EXPECT_EQ(sum, added_on_host(inputs, table));
}

/** @brief Tests that a sequence shorter than the table adds the table's leading rows. */
TEST(Small_BlockPositionEmbedding, AShorterSequenceReadsTheLeadingRows)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_SAMPLES, K_LENGTH, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH + 3, K_DIM);
    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);
    const std::vector<double> inputs = SaTest::eighths(K_SAMPLES * K_LENGTH * K_DIM, 3);
    const std::vector<double> table = SaTest::eighths((K_LENGTH + 3) * K_DIM, 0);
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, inputs));
    SA_EXPECT_OK(evaluator.write(block.weight.id_in(graph), table));

    SA_EXPECT_OK(evaluator.run());

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(block.weight.shape(), (Shape{K_LENGTH + 3, K_DIM}));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> sum, evaluator.read(y));
    EXPECT_EQ(sum, added_on_host(inputs, table));
}

/** @brief Tests that a table that cannot fit the input fails and stays undeclared. */
TEST_P(Small_BlockPositionEmbeddingTable, ATableThatCannotFitTheInputIsRefused)
{
    const BlockRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_SAMPLES, K_LENGTH, K_DIM}, "x");
    PositionEmbedding block(row.length, row.dim);

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = block.forward(trace, x);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
    EXPECT_FALSE(block.weight.declared());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockPositionEmbeddingTable,
    ::testing::Values(BlockRow{"FewerRows", K_LENGTH - 1, K_DIM, "5 positions", "4 rows"},
                      BlockRow{"OtherWidth", K_LENGTH, K_DIM + 1, "PositionEmbedding",
                               "[N, T, 7]"},
                      BlockRow{"NoExtent", 0, 0, "positive extents", ""}),
    row_name<BlockRow>);

/** @brief Tests that the walk names the one table. */
TEST(Small_BlockPositionEmbedding, TheWalkYieldsTheTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    PositionEmbedding block(K_LENGTH, K_DIM);
    static_cast<void>(block.forward(Trace::evaluation(builder),
                                    builder.input(Shape{2, K_LENGTH, K_DIM}, "x")));

    std::vector<std::string> names;
    NN::for_each_parameter(block, "positions", [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    });

    EXPECT_EQ(names, (std::vector<std::string>{"positions.weight"}));
}

/** @brief Tests that an absent "length" takes T and gives a one-port node of the input. */
TEST(Small_BlockPositionEmbedding, AnAbsentLengthTakesThePositions)
{
    const Shape input{K_LENGTH, K_DIM};
    const Config::Value absent = require_value(Config::parse_json("{}", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), absent,
                                                       "layer 1"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<PositionEmbedding>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().length, K_LENGTH);
    EXPECT_EQ(wrapped->layer().dim, K_DIM);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, input);
}

/** @brief Tests that an explicit "length" above T is read into the layer. */
TEST(Small_BlockPositionEmbedding, AnExplicitLengthIsRead)
{
    const Shape input{K_LENGTH, K_DIM};
    const Config::Value longer = require_value(Config::parse_json(R"({"length": 9})", "<test>"));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&input, 1), longer,
                                                       "layer 1"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<PositionEmbedding>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().length, 9);
}

/** @brief Tests that the factory refuses a length below T, a stray key and a rank-one input. */
TEST_P(Small_BlockPositionEmbeddingFactory, TheFactoryRefusesARowThatDoesNotFit)
{
    const FactoryRow& row = GetParam();
    const Config::Value attributes = require_value(Config::parse_json(row.json, "<test>"));

    SA_EXPECT_FAILURE(
        Language::Blocks::make_position_embedding_node(NN::ShapeSpan(&row.input, 1), attributes,
                                                       "layer 1"),
        StatusCode::InvalidDescription, "layer 1", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockPositionEmbeddingFactory,
    ::testing::Values(FactoryRow{"ShorterLength", Shape{K_LENGTH, K_DIM}, R"({"length": 4})",
                                 "'length' is 4", "5 positions"},
                      FactoryRow{"StrayKey", Shape{K_LENGTH, K_DIM}, R"({"rows": 9})", "'rows'",
                                 ""},
                      FactoryRow{"RankOne", Shape{K_DIM}, "{}", "rank-2 [T, C] input", ""}),
    row_name<FactoryRow>);

/** @brief Tests that one position adds the table row of the position fed to it. */
TEST(Small_BlockPositionEmbedding, TheIncrementalFormAddsTheRowOfTheFedPosition)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{K_LENGTH, K_DIM});
    const std::vector<double> slice{1, 2, 3, 4, 5, 6};
    const std::vector<double> table = SaTest::eighths(K_LENGTH * K_DIM, 0);
    Graph::HostEvaluator evaluator(graph);
    SA_EXPECT_OK(evaluator.write(x, slice));
    SA_EXPECT_OK(evaluator.write(block.weight.id_in(graph), table));
    ASSERT_EQ(fed.leaves().size(), 1u);
    SA_EXPECT_OK(evaluator.write(fed.leaves()[0].value, {3}));

    SA_EXPECT_OK(evaluator.run());

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<double> sum, evaluator.read(y));
    const std::vector<double> row = SaTest::row_of(table, 3, K_DIM);
    EXPECT_EQ(sum, (std::vector<double>{1 + row[0], 2 + row[1], 3 + row[2], 4 + row[3],
                                        5 + row[4], 6 + row[5]}));
}

/** @brief Tests that one position is fed an int32 [1, 1] index and carries no value. */
TEST(Small_BlockPositionEmbedding, TheIncrementalFormIsFedItsPositionAlone)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{1, 1, K_DIM}, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;

    const Trace trace = Trace::carrying(builder, fed, carried);
    const Graph::ValueID y = block.forward_incremental(trace, x, Shape{K_LENGTH, K_DIM});

    SA_EXPECT_OK(trace.status());
    EXPECT_EQ(graph.value(y).shape, (Shape{1, 1, K_DIM}));
    ASSERT_EQ(fed.leaves().size(), 1u);
    EXPECT_EQ(fed.leaves()[0].name, Language::Blocks::K_POSITION_LEAF);
    EXPECT_EQ(graph.value(fed.leaves()[0].value).shape, (Shape{1, 1}));
    EXPECT_EQ(graph.value(fed.leaves()[0].value).dtype, DType::Int32);
    EXPECT_TRUE(carried.values().empty());
}

/** @brief Tests that a window longer than the table and a longer slice are refused. */
TEST_P(Small_BlockPositionEmbeddingWindow, TheIncrementalFormRefusesAWindowItsTableCannotServe)
{
    const WindowRow& row = GetParam();
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(row.slice, "x");
    PositionEmbedding block(K_LENGTH, K_DIM);
    NN::FedLeafList fed;
    NN::CarriedValueList carried;
    const Trace trace = Trace::carrying(builder, fed, carried);

    const Graph::ValueID y = block.forward_incremental(trace, x, row.sample);

    EXPECT_EQ(y, x);
    SA_EXPECT_FAILURE(trace.status(), StatusCode::InvalidDescription, row.first, row.second);
    EXPECT_TRUE(fed.leaves().empty());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BlockPositionEmbeddingWindow,
    ::testing::Values(WindowRow{"LongerWindow", Shape{1, 1, K_DIM}, Shape{K_LENGTH + 1, K_DIM},
                                "6 positions", "5 rows"},
                      WindowRow{"LongerSlice", Shape{1, 2, K_DIM}, Shape{K_LENGTH, K_DIM},
                                "one position", "[1, 2, 6]"}),
    row_name<WindowRow>);
