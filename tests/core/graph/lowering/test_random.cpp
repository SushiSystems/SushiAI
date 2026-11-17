/**************************************************************************/
/* test_random.cpp                                                        */
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
 * @file test_random.cpp
 * @brief Tests the shape rule and the emitted IR of DROPOUT, its row of the lowering table, and
 *        the masks its lowered step draws on an engine.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/random.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

#include <support/test_names.hpp>


using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::DropoutResult;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Holds the rate every accepted case passes. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds a site seed with bits set in both of its 32-bit halves. */
    constexpr std::uint64_t K_SEED = 0xFEDCBA9876543210ull;

    /** @brief Pairs a label with a site seed. */
    struct SeedCase
    {
        const char* label;   /**< Names the row. */
        std::uint64_t seed; /**< The site seed. */
    };

    /** @brief Pairs a label with a rate the emitter refuses. */
    struct RateCase
    {
        const char* label; /**< Names the row. */
        double rate;       /**< The refused rate. */
    };

    /** @brief Lists the seeds that must survive the two attribute slots. */
    constexpr SeedCase K_SEEDS[] = {
        {"BitsSetInBothHalves", K_SEED},
        {"Zero", 0},
        {"AllBitsSet", ~std::uint64_t{0}},
    };

    /** @brief Lists the rates outside the open interval (0, 1). */
    constexpr RateCase K_REFUSED_RATES[] = {
        {"Zero", 0.0},
        {"One", 1.0},
        {"Negative", -0.5},
    };

    /** @brief Runs once per row of K_SEEDS. */
    class Small_DropoutSeed : public ::testing::TestWithParam<SeedCase>
    {

    };

    /** @brief Runs once per row of K_REFUSED_RATES. */
    class Small_DropoutRate : public ::testing::TestWithParam<RateCase>
    {

    };

    /** @brief Holds the side of the square mask of the kept-fraction case. */
    constexpr int64_t K_SIDE = 1024;

    /** @brief Holds the probability that an element is kept. */
    constexpr double K_KEEP = 1.0 - K_RATE;

    /** @brief Holds the engine seed of the cases that compare two engines' draws. */
    constexpr std::uint64_t K_ENGINE_SEED = 20261116ull;

    /**
     * @brief Bounds the distance of the kept fraction from K_KEEP: ten standard deviations of a
     * fraction of K_SIDE * K_SIDE draws, whose deviation is at most 0.5 / K_SIDE. The two-sided
     * false-failure rate of a fair draw is below 1e-22, and the bound lies under the design's 0.005.
     */
    constexpr double K_FRACTION_BOUND = 10.0 * 0.5 / static_cast<double>(K_SIDE);

    /** @brief Provides the Engine every drawing case compiles on. */
    class Medium_DropoutDraw : public SaTest::EngineTest
    {

    };

    /** @brief Holds a compiled dropout over a leaf, and what a case reads of it. */
    struct MaskedLeaf
    {
        /** @brief Builds the pool on @p engine; make_masked_leaf() fills the rest. */
        explicit MaskedLeaf(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief Runs the step once and returns the mask it saved. */
        std::vector<double> run_and_read_mask()
        {
            SA_EXPECT_OK(step->run());
            return SaTest::read_tensor(pool.tensor(mask));
        }

        /** @brief Holds the forward node. */
        Graph::Graph graph;

        /** @brief Holds the tensor of each value. */
        TensorPool pool;

        /** @brief Names the mask the node saves. */
        ValueID mask = Graph::K_NO_VALUE;

        /** @brief Holds the compiled step. */
        std::unique_ptr<CompiledStep> step;
    };

    /** @brief Compiles a dropout of rate K_RATE and site seed K_SEED over a leaf of @p shape. */
    Result<std::unique_ptr<MaskedLeaf>> make_masked_leaf(SushiAI::ExecutionBackend& backend,
                                                         const Shape& shape)
    {
        std::unique_ptr<MaskedLeaf> made = std::make_unique<MaskedLeaf>(backend.engine());
        const GraphBuilder builder(made->graph);

        const ValueID x = builder.parameter(shape, "x");
        SA_TRY_ASSIGN(const DropoutResult dropped, builder.dropout(x, K_RATE, K_SEED));
        made->mask = dropped.saved_mask;

        SA_TRY(made->pool.materialise_leaves(made->graph));
        SA_TRY_ASSIGN(made->step, CompiledStep::compile(backend, made->graph, made->pool));
        return std::move(made);
    }

    /** @brief Returns the fraction of elements of @p mask that are not zero. */
    double kept_fraction(const std::vector<double>& mask)
    {
        const auto kept = std::count_if(mask.begin(), mask.end(),
                                        [](double value) { return value != 0.0; });
        return static_cast<double>(kept) / static_cast<double>(mask.size());
    }

    /** @brief Expects @p id to be shaped like [2, 3, 4], typed float64 and an activation. */
    void expect_dropout_output(const Graph::Graph& graph, ValueID id)
    {
        EXPECT_EQ(graph.value(id).shape, (Shape{2, 3, 4}));
        EXPECT_EQ(graph.value(id).dtype, DType::Float64);
        EXPECT_EQ(graph.value(id).lifetime, Graph::Lifetime::Activation);
    }

    /** @brief Expects @p refused to name @p part, and the graph to hold no node beyond @p nodes. */
    void expect_refused(const Graph::Graph& graph, const Result<DropoutResult>& refused,
                        std::size_t nodes, const char* part)
    {
        SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, part);
        EXPECT_EQ(graph.node_count(), nodes) << "a refused call left a node behind";
        EXPECT_TRUE(graph.has_refusal());
    }
}

/** @brief Tests that the node reads [x] and writes the value and a mask, both shaped like x. */
TEST(Small_DropoutOps, DropoutWritesItsValueAndTheMaskItSaves)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 3, 4}, "x", DType::Float64);

    const Result<DropoutResult> out = builder.dropout(x, K_RATE, K_SEED);

    SA_EXPECT_OK(out);
    ASSERT_TRUE(out.has_value());
    const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
    EXPECT_EQ(node.op, Ops::K_DROPOUT.id);
    EXPECT_EQ(node.inputs, (std::vector<ValueID>{x}));
    EXPECT_EQ(node.outputs,
              (std::vector<ValueID>{out.value().output, out.value().saved_mask}));
    EXPECT_EQ(node.attributes.values[Graph::K_DROPOUT_RATE_SLOT], K_RATE);

    expect_dropout_output(graph, out.value().output);
    expect_dropout_output(graph, out.value().saved_mask);

    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that a 64-bit seed comes back whole from the two attribute slots it is put in. */
TEST_P(Small_DropoutSeed, TheSiteSeedSurvivesItsAttributeSlots)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4}, "x");

    const Result<DropoutResult> out = builder.dropout(x, K_RATE, GetParam().seed);
    ASSERT_TRUE(out.has_value());

    const Graph::Node& node = graph.node(graph.value(out.value().output).producer);
    EXPECT_EQ(Graph::dropout_seed(node.attributes), GetParam().seed);
}

INSTANTIATE_TEST_SUITE_P(, Small_DropoutSeed, ::testing::ValuesIn(K_SEEDS),
                         [](const ::testing::TestParamInfo<SeedCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.label);
                         });

/** @brief Tests that the emitter refuses an input whose element type cannot carry a mask. */
TEST(Small_DropoutOps, DropoutRefusesAnInputOfAnIntegerType)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID whole = builder.input(Shape{2, 4}, "whole", DType::Int32);
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.dropout(whole, K_RATE, K_SEED), nodes,
                   "half, float32 or float64 input");
}

/** @brief Tests that the emitter refuses a rate outside the open interval (0, 1). */
TEST_P(Small_DropoutRate, DropoutRefusesARateOutsideZeroToOne)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 4}, "x");
    const std::size_t nodes = graph.node_count();

    expect_refused(graph, builder.dropout(x, GetParam().rate, K_SEED), nodes, "rate in (0, 1)");
}

INSTANTIATE_TEST_SUITE_P(, Small_DropoutRate, ::testing::ValuesIn(K_REFUSED_RATES),
                         [](const ::testing::TestParamInfo<RateCase>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.label);
                         });

/** @brief Tests that DROPOUT has the one row of the random table and no other. */
TEST(Small_DropoutOps, DropoutHasTheRowOfTheRandomTable)
{
    ASSERT_EQ(Graph::K_RANDOM_LOWERINGS.size(), 1u);
    EXPECT_EQ(Graph::K_RANDOM_LOWERINGS[0].op, Ops::K_DROPOUT.id);
    EXPECT_EQ(Graph::find_lowering(Ops::K_DROPOUT.id), &Graph::lower_dropout);
    EXPECT_EQ(Graph::lowering_row_count(Ops::K_DROPOUT.id), 1u);
    EXPECT_FALSE(Graph::is_unlowered(Ops::K_DROPOUT.id));
}

/** @brief Tests that one compiled step draws a new mask at its second run, with one compile. */
TEST_F(Medium_DropoutDraw, OneCompiledStepDrawsADifferentMaskAtEachRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> leaf,
                            make_masked_leaf(backend(), Shape{64, 64}));

    const std::vector<double> first = leaf->run_and_read_mask();
    const std::vector<double> second = leaf->run_and_read_mask();

    EXPECT_NE(first, second);
    EXPECT_EQ(leaf->step->compile_count(), 1u);
    EXPECT_EQ(leaf->step->replay_prelude_count(), 1u);
}

/** @brief Tests that a stream seeded alike gives equal masks at equal run indices. */
TEST_F(Medium_DropoutDraw, TwoStepsOnAStreamSeededAlikeGiveEqualMasksAtEqualRunIndices)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> first,
                            make_masked_leaf(backend(), Shape{64, 64}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> second,
                            make_masked_leaf(backend(), Shape{64, 64}));

    engine->rng().reseed(K_ENGINE_SEED);
    const std::vector<double> first_run0 = first->run_and_read_mask();
    const std::vector<double> first_run1 = first->run_and_read_mask();

    engine->rng().reseed(K_ENGINE_SEED);
    EXPECT_EQ(second->run_and_read_mask(), first_run0);
    EXPECT_EQ(second->run_and_read_mask(), first_run1);
}

/** @brief Tests that another stream seed gives another mask at the same run index. */
TEST_F(Medium_DropoutDraw, AnotherStreamSeedGivesAnotherMaskAtTheSameRunIndex)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> first,
                            make_masked_leaf(backend(), Shape{64, 64}));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> second,
                            make_masked_leaf(backend(), Shape{64, 64}));

    engine->rng().reseed(K_ENGINE_SEED);
    const std::vector<double> seeded = first->run_and_read_mask();

    engine->rng().reseed(K_ENGINE_SEED + 1);
    EXPECT_NE(second->run_and_read_mask(), seeded);
}

/** @brief Tests that the kept fraction of a [1024, 1024] mask is within K_FRACTION_BOUND of keep. */
TEST_F(Medium_DropoutDraw, TheKeptFractionOfALargeMaskIsTheKeepProbability)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedLeaf> leaf,
                            make_masked_leaf(backend(), Shape{K_SIDE, K_SIDE}));
    engine->rng().reseed(K_ENGINE_SEED);

    EXPECT_NEAR(kept_fraction(leaf->run_and_read_mask()), K_KEEP, K_FRACTION_BOUND);
}
