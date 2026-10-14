/**************************************************************************/
/* test_compiled_step.cpp                                                 */
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
 * @file test_compiled_step.cpp
 * @brief Tests that a compiled step restores its run count once, before its first run.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every case compiles on. */
    class Unit_CompiledStep : public SaTest::EngineTest
    {

    };

    /** @brief Holds the element count of the masked value. */
    constexpr std::size_t K_ELEMENTS = 256;

    /** @brief Holds the drop rate; a kept element is scaled by 4/3. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds the probability that an element is kept. */
    constexpr double K_KEEP = 1.0 - K_RATE;

    /** @brief Holds the site seed of the one dropout of every case. */
    constexpr std::uint64_t K_SEED = 0x5EEDC0DE20261011ull;

    /** @brief Bounds a FLOAT32 scale read back as a double. */
    constexpr double K_TOLERANCE = 1e-6;

    /** @brief Holds a compiled step that masks a leaf of ones, so its output is the mask. */
    struct MaskedOnes
    {
        /** @brief Builds the pool on @p engine; make_masked_ones() fills the rest. */
        explicit MaskedOnes(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief Holds the leaf and the dropout node. */
        Graph::Graph graph;

        /** @brief Holds the tensor of each value. */
        TensorPool pool;

        /** @brief Names the masked value. */
        ValueID masked = Graph::K_NO_VALUE;

        /** @brief Holds the compiled step. */
        std::unique_ptr<CompiledStep> step;
    };

    /** @brief Compiles a dropout over K_ELEMENTS ones with the site seed K_SEED. */
    Result<std::unique_ptr<MaskedOnes>> make_masked_ones(SushiBLAS::Engine& engine)
    {
        std::unique_ptr<MaskedOnes> made = std::make_unique<MaskedOnes>(engine);
        const GraphBuilder builder(made->graph);

        const ValueID ones =
            builder.parameter(Shape{static_cast<int64_t>(K_ELEMENTS)}, "ones");
        SA_TRY_ASSIGN(const Graph::DropoutResult dropped, builder.dropout(ones, K_RATE, K_SEED));
        made->masked = dropped.output;

        SA_TRY(made->pool.materialise_leaves(made->graph));
        SA_TRY_ASSIGN(made->step, CompiledStep::compile(engine, made->graph, made->pool));
        SaTest::write_tensor(made->pool.tensor(ones), std::vector<double>(K_ELEMENTS, 1.0));
        return std::move(made);
    }

    /** @brief Expects @p mask to hold what the generator draws for K_SEED in run @p run. */
    void expect_mask_of_run(const std::vector<double>& mask, std::uint64_t run)
    {
        ASSERT_EQ(mask.size(), K_ELEMENTS);
        for (std::size_t element = 0; element < K_ELEMENTS; ++element)
        {
            const double expected =
                Graph::dropout_keeps(K_SEED, run, element, K_KEEP) ? 1.0 / K_KEEP : 0.0;
            EXPECT_NEAR(mask[element], expected, K_TOLERANCE) << "at element " << element;
        }
    }
} // namespace

/** @brief Tests that a step restored to a count draws the mask of that run next. */
TEST_F(Unit_CompiledStep, ARestoredRunCountIsTheIndexOfTheNextRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> whole, make_masked_ones(*engine));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> resumed,
                            make_masked_ones(*engine));
    ASSERT_EQ(whole->step->replay_prelude_count(), 1u);

    for (int run = 0; run < 3; ++run)
        SA_EXPECT_OK(whole->step->run());
    const std::vector<double> third = SaTest::read_tensor(whole->pool.tensor(whole->masked));

    EXPECT_EQ(resumed->step->run_count(), 0u);
    SA_EXPECT_OK(resumed->step->restore_run_count(2));
    EXPECT_EQ(resumed->step->run_count(), 2u);
    SA_EXPECT_OK(resumed->step->run());

    EXPECT_EQ(resumed->step->run_count(), 3u);
    EXPECT_EQ(SaTest::read_tensor(resumed->pool.tensor(resumed->masked)), third);
    expect_mask_of_run(third, 2);
}

/** @brief Tests that a step that has run refuses a restore and keeps its count. */
TEST_F(Unit_CompiledStep, ARestoreAfterARunIsRefusedWithTheCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    SA_EXPECT_OK(made->step->run());
    SA_EXPECT_FAILURE(made->step->restore_run_count(4), StatusCode::InvalidDescription,
                      "before its first run", "run 1");
    EXPECT_EQ(made->step->run_count(), 1u);
}

/** @brief Tests that a count above 0 is restored once, and that 0 leaves a new step new. */
TEST_F(Unit_CompiledStep, ASecondRestoreIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    SA_EXPECT_OK(made->step->restore_run_count(0));
    SA_EXPECT_OK(made->step->restore_run_count(5));
    SA_EXPECT_FAILURE(made->step->restore_run_count(7), StatusCode::InvalidDescription,
                      "run 5");
    EXPECT_EQ(made->step->run_count(), 5u);
}

/** @brief Tests that two ranges of a graph run as two steps over the tensors of one pool. */
TEST_F(Unit_CompiledStep, TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    const ValueID doubled = builder.scale(leaf, 2.0);
    const ValueID tripled = builder.scale(doubled, 3.0);
    const Graph::NodeID cut = graph.value(doubled).producer + 1;
    ASSERT_LT(cut, graph.node_count());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> first,
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                              Graph::NodeRange{0, cut}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> second,
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                              Graph::NodeRange{cut, Graph::K_NO_NODE}));
    EXPECT_GE(first->operation_count(), 1u);
    EXPECT_GE(second->operation_count(), 1u);

    SaTest::write_tensor(pool.tensor(leaf), std::vector<double>{1.0, 2.0, 3.0, 4.0});
    SA_EXPECT_OK(first->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(doubled)),
              (std::vector<double>{2.0, 4.0, 6.0, 8.0}));

    // The host writes between the two steps, and the second reads what it wrote.
    SaTest::write_tensor(pool.tensor(doubled), std::vector<double>(4, 10.0));
    SA_EXPECT_OK(second->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(tripled)),
              (std::vector<double>{30.0, 30.0, 30.0, 30.0}));
}

/** @brief Tests that a range that ends behind the last node is refused with its bounds. */
TEST_F(Unit_CompiledStep, ARangeOutsideTheGraphIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    static_cast<void>(builder.scale(leaf, 2.0));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    const Graph::NodeID count = static_cast<Graph::NodeID>(graph.node_count());
    SA_EXPECT_FAILURE(CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                                            Graph::NodeRange{0, count + 1}),
                      StatusCode::InvalidDescription, "does not lie in a graph");
    SA_EXPECT_FAILURE(
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr, Graph::NodeRange{1, 0}),
        StatusCode::InvalidDescription, "[1, 0)");
}
