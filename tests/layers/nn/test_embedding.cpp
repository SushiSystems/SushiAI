/**************************************************************************/
/* test_embedding.cpp                                                     */
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
 * @file test_embedding.cpp
 * @brief Tests the embedding layer: its trace, its parameters and the row gather it runs on a
 *        device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using NN::Embedding;
using NN::Trace;
// SaTest stays reachable inside a TEST_F body; see tests/layers/nn/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::read_tensor_bytes;
using SaTest::write_tensor;

namespace
{
    /** @brief Holds the row count of the table every case builds. */
    constexpr int64_t K_ROWS = 7;

    /** @brief Holds the row length of the table every case builds. */
    constexpr int64_t K_LENGTH = 3;

    /** @brief Returns a layer over a [K_ROWS, K_LENGTH] table of @p dtype. */
    Embedding make_layer(DType dtype = DType::Float32)
    {
        Embedding layer{};
        layer.num_embeddings = K_ROWS;
        layer.embedding_dim = K_LENGTH;
        layer.dtype = dtype;
        return layer;
    }

    /** @brief Counts the nodes of @p graph that carry @p op. */
    std::size_t count_nodes(const Graph::Graph& graph, OpID op)
    {
        std::size_t count = 0;
        for (const Graph::Node& node : graph.nodes())
        {
            if (node.op == op)
                ++count;
        }

        return count;
    }

    /** @brief Traces @p layer over @p x, expects @p x back, and returns what was recorded. */
    Status refused(Embedding& layer, const Graph::GraphBuilder& builder, Graph::ValueID x)
    {
        const Trace trace = Trace::evaluation(builder);
        EXPECT_EQ(layer.forward(trace, x), x);
        return trace.status();
    }

    /** @brief Traces a layer over a value id the one-value graph does not hold. */
    void trace_over_a_value_the_graph_lacks()
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        static_cast<void>(builder.input(Shape{2, 5}, "indices", DType::Int32));
        Embedding layer = make_layer();
        static_cast<void>(layer.forward(Trace::evaluation(builder), Graph::ValueID{7}));
    }

    /** @brief Asks for the result shape of a per-sample index of rank five. */
    void shape_of_a_rank_five_sample()
    {
        const Embedding layer = make_layer();
        static_cast<void>(layer.output_shape(Shape{1, 1, 1, 1, 1}));
    }
}

static_assert(NN::TraceModule<Embedding> && !NN::Module<Embedding> && !NN::ModalModule<Embedding>,
              "Embedding has the trace form alone");

/** @brief Tests that a trace adds the table leaf and one GATHER_ROWS over [table, indices]. */
TEST(Unit_Embedding, ATraceDeclaresTheTableAndEmitsOneGather)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices = builder.input(Shape{2, 5}, "indices", DType::Int32);
    Embedding layer = make_layer();
    const std::size_t nodes_before = graph.node_count();

    const Trace trace = Trace::evaluation(builder);
    const Graph::ValueID y = layer.forward(trace, indices);

    SA_EXPECT_OK(trace.status());
    EXPECT_FALSE(graph.has_refusal());
    EXPECT_EQ(graph.node_count(), nodes_before + 2u);
    ASSERT_EQ(count_nodes(graph, Ops::K_GATHER_ROWS.id), 1u);

    const Graph::Node& node = graph.node(graph.value(y).producer);
    EXPECT_EQ(node.op, Ops::K_GATHER_ROWS.id);
    ASSERT_EQ(node.inputs.size(), 2u);
    EXPECT_EQ(node.inputs[0], layer.weight.id_in(graph));
    EXPECT_EQ(node.inputs[1], indices);
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 5, K_LENGTH}));
    EXPECT_EQ(graph.value(y).dtype, DType::Float32);
}

/** @brief Tests that the table is a [V, D] Embedding parameter drawn from a normal of 0.02. */
TEST(Unit_Embedding, TheTableIsAnEmbeddingDrawnFromANormalOfFixedDeviation)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Embedding layer = make_layer(DType::Float64);
    const Graph::ValueID y = layer.forward(
        Trace::evaluation(builder), builder.input(Shape{4}, "indices", DType::Int32));

    ASSERT_TRUE(layer.weight.declared());
    EXPECT_EQ(layer.weight.shape(), (Shape{K_ROWS, K_LENGTH}));
    EXPECT_EQ(layer.weight.dtype(), DType::Float64);
    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Embedding);
    EXPECT_EQ(layer.weight.init().kind, NN::ParameterInit::Kind::Normal);
    EXPECT_NEAR(layer.weight.init().deviation, NN::K_EMBEDDING_DEVIATION, 0.0);
    EXPECT_NEAR(NN::K_EMBEDDING_DEVIATION, 0.02, 0.0);
    EXPECT_EQ(layer.weight.fan_in(), K_LENGTH);
    EXPECT_EQ(layer.weight.fan_out(), K_LENGTH);
    EXPECT_EQ(graph.value(y).dtype, DType::Float64);
    EXPECT_EQ(graph.value(layer.weight.id_in(graph)).lifetime, Graph::Lifetime::Parameter);
    EXPECT_TRUE(graph.value(layer.weight.id_in(graph)).requires_grad);
}

/** @brief Tests that the state walk yields the one parameter, named weight. */
TEST(Unit_Embedding, TheWalkYieldsTheTable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Embedding layer = make_layer();
    static_cast<void>(layer.forward(Trace::evaluation(builder),
                                    builder.input(Shape{4}, "indices", DType::Int32)));

    std::vector<std::string> names;
    const auto on_parameter = [&names](const std::string& name, NN::Parameter& parameter)
    {
        names.push_back(name);
        EXPECT_EQ(parameter.shape(), (Shape{K_ROWS, K_LENGTH}));
    };
    const auto on_buffer = [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    };
    const NN::StateVisitor visitor(on_parameter, on_buffer);
    NN::walk_state(layer, "tokens", visitor);

    EXPECT_EQ(names, (std::vector<std::string>{"tokens.weight"}));
}

/** @brief Tests that traces into two graphs name one state and declare one leaf in each. */
TEST(Unit_Embedding, TwoGraphsNameTheSameTable)
{
    Embedding layer = make_layer();

    Graph::Graph first;
    const Graph::GraphBuilder first_builder(first);
    static_cast<void>(layer.forward(Trace::evaluation(first_builder),
                                    first_builder.input(Shape{2, 5}, "indices", DType::Int32)));
    const Graph::StateID state = layer.weight.state();

    Graph::Graph second;
    const Graph::GraphBuilder second_builder(second);
    static_cast<void>(layer.forward(Trace::evaluation(second_builder),
                                    second_builder.input(Shape{3, 2}, "indices", DType::Int32)));

    EXPECT_NE(state, Graph::K_NO_STATE);
    EXPECT_EQ(layer.weight.state(), state);
    EXPECT_NE(layer.weight.id_in(first), Graph::K_NO_VALUE);
    EXPECT_NE(layer.weight.id_in(second), Graph::K_NO_VALUE);
}

/** @brief Tests that a FLOAT32 input and an INT64 input fail and declare nothing. */
TEST(Unit_Embedding, AnInputThatIsNotInt32Fails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID real = builder.input(Shape{2, 5}, "real");
    const Graph::ValueID wide = builder.input(Shape{2, 5}, "wide", DType::Int64);
    Embedding layer = make_layer();
    const std::size_t values_before = graph.value_count();
    const std::size_t nodes_before = graph.node_count();

    SA_EXPECT_FAILURE(refused(layer, builder, real), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "int32 input of indices", "float32");
    SA_EXPECT_FAILURE(refused(layer, builder, wide), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "int32 input of indices", "int64");

    EXPECT_EQ(graph.value_count(), values_before);
    EXPECT_EQ(graph.node_count(), nodes_before);
    EXPECT_FALSE(layer.weight.declared());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that an index of the highest rank fails, as the result has no rank left. */
TEST(Unit_Embedding, AnInputOfTheHighestRankFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices =
        builder.input(Shape{1, 1, 1, 1, 1, 1}, "indices", DType::Int32);
    Embedding layer = make_layer();

    SA_EXPECT_FAILURE(refused(layer, builder, indices), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "rank 1 to 5", "[1, 1, 1, 1, 1, 1]");
    EXPECT_FALSE(layer.weight.declared());
}

/** @brief Tests that an extent that is not positive and an integer table each fail. */
TEST(Unit_Embedding, ALayerDescribedWrongFails)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID indices = builder.input(Shape{2, 5}, "indices", DType::Int32);

    Embedding no_rows = make_layer();
    no_rows.num_embeddings = 0;
    Embedding no_length = make_layer();
    no_length.embedding_dim = -1;
    Embedding integer_table = make_layer(DType::Int32);

    SA_EXPECT_FAILURE(refused(no_rows, builder, indices), StatusCode::InvalidDescription,
                      "positive extents", "num_embeddings=0");
    SA_EXPECT_FAILURE(refused(no_length, builder, indices), StatusCode::InvalidDescription,
                      "positive extents", "embedding_dim=-1");
    SA_EXPECT_FAILURE(refused(integer_table, builder, indices), StatusCode::InvalidDescription,
                      "Embedding(7, 3)", "half, float32 or float64 table", "int32");
    EXPECT_FALSE(no_rows.weight.declared());
    EXPECT_FALSE(integer_table.weight.declared());
}

/** @brief Tests that output_shape appends the row length to a per-sample shape of any rank. */
TEST(Unit_Embedding, OutputShapeAppendsTheRowLength)
{
    const Embedding layer = make_layer();

    EXPECT_EQ(layer.output_shape(Shape{5}), (Shape{5, K_LENGTH}));
    EXPECT_EQ(layer.output_shape(Shape{4, 2}), (Shape{4, 2, K_LENGTH}));
    EXPECT_EQ(layer.output_shape(Shape{1, 1, 1, 1}), (Shape{1, 1, 1, 1, K_LENGTH}));
}

/** @brief Tests that the factory reads both keys and builds a node over [T] into [T, D]. */
TEST(Unit_Embedding, TheFactoryReadsTheTwoExtents)
{
    const Shape input{5};
    const Config::Value attributes =
        Config::parse_json(R"({"num_embeddings": 11, "embedding_dim": 4})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<NN::Node> node,
        NN::make_embedding_node(NN::ShapeSpan(&input, 1), attributes, "layer 0"));

    const auto* wrapped = dynamic_cast<const NN::ModuleNode<Embedding>*>(node.get());
    ASSERT_NE(wrapped, nullptr);
    EXPECT_EQ(wrapped->layer().num_embeddings, 11);
    EXPECT_EQ(wrapped->layer().embedding_dim, 4);
    EXPECT_EQ(wrapped->layer().dtype, DType::Float32);
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{5, 4}));
    EXPECT_EQ(node->input_count(), 1u);
}

/** @brief Tests that the factory refuses a stray key, an absent key and an input that is off. */
TEST(Unit_Embedding, TheFactoryRefusesARowThatDoesNotFit)
{
    const Shape input{5};
    const Shape pair[] = {Shape{5}, Shape{5}};
    const Shape deep{1, 1, 1, 1, 1};
    const Config::Value both =
        Config::parse_json(R"({"num_embeddings": 11, "embedding_dim": 4})", "<test>");
    const Config::Value stray = Config::parse_json(
        R"({"num_embeddings": 11, "embedding_dim": 4, "padding": 0})", "<test>");
    const Config::Value no_length = Config::parse_json(R"({"num_embeddings": 11})", "<test>");
    const Config::Value zero_rows =
        Config::parse_json(R"({"num_embeddings": 0, "embedding_dim": 4})", "<test>");

    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), stray, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'padding'");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), no_length, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'embedding_dim'", "absent");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&input, 1), zero_rows, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "'num_embeddings'", "positive");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(pair, 2), both, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "reads one input", "given 2");
    SA_EXPECT_FAILURE(NN::make_embedding_node(NN::ShapeSpan(&deep, 1), both, "layer 3"),
                      StatusCode::InvalidDescription, "layer 3", "[1, 1, 1, 1, 1]", "rank 1 to 4");
}

/** @brief Tests that a trace over a value the graph does not hold stops the process. */
TEST(Unit_EmbeddingDeathTest, ATraceOverAValueTheGraphLacksIsFatal)
{
    SA_EXPECT_FATAL(trace_over_a_value_the_graph_lacks(), "Embedding(7, 3)", "value 7",
                    "holds 1 values");
}

/** @brief Tests that output_shape over a per-sample index of rank five stops the process. */
TEST(Unit_EmbeddingDeathTest, OutputShapeOverARankFiveSampleIsFatal)
{
    SA_EXPECT_FATAL(shape_of_a_rank_five_sample(), "Embedding::output_shape", "at most 4");
}

namespace
{
    /** @brief Provides the Engine every gather case records on. */
    class Integration_GatherRows : public SaTest::EngineTest
    {
    };

    /** @brief Bounds a copied element and a sum of dyadic values, both of which are exact. */
    constexpr double K_EXACT_TOLERANCE = 0.0;

    /** @brief Holds how often the determinism case replays its step. */
    constexpr int K_REPLAYS = 4;

    /** @brief Writes @p indices into an INT32 leaf; no task is recorded while the host writes. */
    void write_indices(SushiBLAS::Tensor& tensor, const std::vector<std::int32_t>& indices)
    {
        ASSERT_EQ(tensor.dtype, DType::Int32);
        ASSERT_EQ(static_cast<std::size_t>(tensor.num_elements), indices.size());

        std::int32_t* const data = tensor.data_as<std::int32_t>();
        for (std::size_t i = 0; i < indices.size(); ++i)
            data[i] = indices[i];
    }

    /** @brief Returns the rows of a [rows, width] @p table that @p indices names, in order. */
    std::vector<double> gathered_on_host(const std::vector<double>& table, std::size_t width,
                                         const std::vector<std::int32_t>& indices)
    {
        std::vector<double> rows;
        rows.reserve(indices.size() * width);
        for (const std::int32_t index : indices)
        {
            const std::size_t first = static_cast<std::size_t>(index) * width;
            rows.insert(rows.end(), table.begin() + static_cast<std::ptrdiff_t>(first),
                        table.begin() + static_cast<std::ptrdiff_t>(first + width));
        }

        return rows;
    }

    /** @brief Returns a zero [table_rows, width] with each of @p rows added in index order. */
    std::vector<double> scattered_on_host(const std::vector<double>& rows, std::size_t width,
                                          const std::vector<std::int32_t>& indices,
                                          std::size_t table_rows)
    {
        std::vector<double> sums(table_rows * width, 0.0);
        for (std::size_t n = 0; n < indices.size(); ++n)
        {
            const std::size_t first = static_cast<std::size_t>(indices[n]) * width;
            for (std::size_t column = 0; column < width; ++column)
                sums[first + column] += rows[n * width + column];
        }

        return sums;
    }

    /** @brief Expects @p got to equal @p expected element by element, to K_EXACT_TOLERANCE. */
    void expect_exact(const std::vector<double>& got, const std::vector<double>& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], K_EXACT_TOLERANCE) << "at element " << i;
    }

    /** @brief Returns [0.25, 0.5, 0.75, ...], @p count dyadic values whose sums are exact. */
    std::vector<double> dyadic_ramp(std::size_t count)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = 0.25 * static_cast<double>(i + 1);

        return values;
    }
}

/** @brief Tests that a gather over a [2, 3] index copies the named rows, a repeated one too. */
TEST_F(Integration_GatherRows, GatherCopiesTheRowsItsIndexNames)
{
    const std::vector<std::int32_t> indices = {4, 0, 2, 2, 2, 1};
    const std::vector<double> table = dyadic_ramp(5 * 3);

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table_id = builder.input(Shape{5, 3}, "table");
    const ValueID index_id = builder.input(Shape{2, 3}, "indices", DType::Int32);
    const Graph::Traced gathered = builder.gather_rows(table_id, index_id);
    SA_EXPECT_OK(gathered);
    ASSERT_TRUE(gathered.has_value());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table_id), table);
    write_indices(pool.tensor(index_id), indices);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    expect_exact(read_tensor(pool.tensor(gathered.value())), gathered_on_host(table, 3, indices));
}

/** @brief Tests that the step reads the index of each run, not the one it was compiled over. */
TEST_F(Integration_GatherRows, AReplayReadsTheIndexOfItsOwnRun)
{
    const std::vector<double> table = dyadic_ramp(6 * 2);

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table_id = builder.input(Shape{6, 2}, "table", DType::Float64);
    const ValueID index_id = builder.input(Shape{4}, "indices", DType::Int32);
    const Graph::Traced gathered = builder.gather_rows(table_id, index_id);
    ASSERT_TRUE(gathered.has_value());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(table_id), table);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));

    for (const std::vector<std::int32_t>& indices :
         {std::vector<std::int32_t>{0, 1, 2, 3}, std::vector<std::int32_t>{5, 5, 0, 4}})
    {
        write_indices(pool.tensor(index_id), indices);
        SA_EXPECT_OK(step->run());
        expect_exact(read_tensor(pool.tensor(gathered.value())),
                     gathered_on_host(table, 2, indices));
    }
}

/** @brief Tests that a scatter-add sums the rows of a repeated index and zeroes unnamed rows. */
TEST_F(Integration_GatherRows, ScatterAddSumsTheRowsOfARepeatedIndex)
{
    const std::vector<std::int32_t> indices = {3, 1, 3, 3, 0, 1};
    const std::vector<double> rows = dyadic_ramp(6 * 2);

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID rows_id = builder.input(Shape{6, 2}, "rows", DType::Float64);
    const ValueID index_id = builder.input(Shape{6}, "indices", DType::Int32);
    const Graph::Traced sums = builder.scatter_add_rows(rows_id, index_id, 5);
    SA_EXPECT_OK(sums);
    ASSERT_TRUE(sums.has_value());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(rows_id), rows);
    write_indices(pool.tensor(index_id), indices);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));

    // The second run proves the result starts from zero and not from the first run's sums.
    SA_EXPECT_OK(step->run());
    SA_EXPECT_OK(step->run());

    const std::vector<double> expected = scattered_on_host(rows, 2, indices, 5);
    expect_exact(read_tensor(pool.tensor(sums.value())), expected);
    EXPECT_NEAR(expected[2 * 2], 0.0, K_EXACT_TOLERANCE) << "row 2 is named by no index";
    EXPECT_NEAR(expected[4 * 2], 0.0, K_EXACT_TOLERANCE) << "row 4 is named by no index";
}

/**
 * @brief Tests that the table gradient of a FLOAT32 gather with a heavily repeated index has
 *        the same bytes on every replay.
 */
TEST_F(Integration_GatherRows, TheTableGradientIsBitIdenticalAcrossReplays)
{
    constexpr std::int64_t K_TABLE_ROWS = 13;
    constexpr std::int64_t K_WIDTH = 8;
    constexpr std::int64_t K_COUNT = 256;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID table_id = builder.parameter(Shape{K_TABLE_ROWS, K_WIDTH}, "table");
    const ValueID index_id = builder.input(Shape{K_COUNT}, "indices", DType::Int32);
    const ValueID targets = builder.input(Shape{K_COUNT, K_WIDTH}, "targets");
    const Graph::Traced gathered = builder.gather_rows(table_id, index_id);
    ASSERT_TRUE(gathered.has_value());
    const ValueID loss = builder.nll_loss(gathered.value(), targets);

    const std::vector<ValueID> wrt = {table_id};
    const std::vector<ValueID> gradients = Autograd::differentiate(
        graph, loss, SushiRuntime::span<const ValueID>(wrt.data(), wrt.size()));
    ASSERT_NE(gradients.at(0), Graph::K_NO_VALUE);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    std::vector<double> table(static_cast<std::size_t>(K_TABLE_ROWS * K_WIDTH));
    std::vector<double> target_values(static_cast<std::size_t>(K_COUNT * K_WIDTH));
    SaTest::HostNoise(4101).fill(table, 1.0);
    SaTest::HostNoise(4102).fill(target_values, 1.0);
    write_tensor(pool.tensor(table_id), table);
    write_tensor(pool.tensor(targets), target_values);

    // Every row of the table is named about twenty times, so every sum has many terms.
    std::vector<std::int32_t> indices(static_cast<std::size_t>(K_COUNT));
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = static_cast<std::int32_t>((i * 7) % static_cast<std::size_t>(K_TABLE_ROWS));
    write_indices(pool.tensor(index_id), indices);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());
    const std::vector<unsigned char> first = read_tensor_bytes(pool.tensor(gradients[0]));

    for (int replay = 1; replay < K_REPLAYS; ++replay)
    {
        SA_EXPECT_OK(step->run());
        const std::vector<unsigned char> again = read_tensor_bytes(pool.tensor(gradients[0]));
        ASSERT_EQ(again.size(), first.size());
        EXPECT_EQ(SaTest::first_difference(first, again), first.size()) << "replay " << replay;
    }
}

/** @brief Tests that an Embedding over a [2, 3] batch of ids yields the rows of its table. */
TEST_F(Integration_GatherRows, AnEmbeddingYieldsTheRowsOfItsTable)
{
    const std::vector<std::int32_t> ids = {6, 6, 0, 3, 1, 6};

    NN::Embedding layer{};
    layer.num_embeddings = 7;
    layer.embedding_dim = 4;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID id_leaf = builder.input(Shape{2, 3}, "ids", DType::Int32);
    const NN::Trace trace = NN::Trace::evaluation(builder);
    const ValueID embedded = layer.forward(trace, id_leaf);
    SA_EXPECT_OK(trace.status());
    ASSERT_EQ(graph.value(embedded).shape, (Shape{2, 3, 4}));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));

    const std::vector<double> table = dyadic_ramp(7 * 4);
    write_tensor(pool.tensor(layer.weight.id_in(graph)), table);
    write_indices(pool.tensor(id_leaf), ids);

    std::unique_ptr<CompiledStep> step;
    SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(*engine, graph, pool));
    SA_EXPECT_OK(step->run());

    expect_exact(read_tensor(pool.tensor(embedded)), gathered_on_host(table, 4, ids));
}
