/**************************************************************************/
/* test_gather_rows.cpp                                                   */
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
 * @file test_gather_rows.cpp
 * @brief Checks GATHER_ROWS, SCATTER_ADD_ROWS and the Embedding layer on an engine against the
 *        rows and the sums a host loop gives.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
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
