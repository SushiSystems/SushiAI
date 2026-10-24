/**************************************************************************/
/* test_context.cpp                                                       */
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
 * @file test_context.cpp
 * @brief Tests what a lowering context gives a row: its node, its tensors, counts and lists.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/graph/replay_prelude.hpp>
#include <SushiAI/graph/tensor_pool.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::FusedForm;
using Graph::FusionChoice;
using Graph::FusionQuery;
using Graph::FusionSelection;
using Graph::GraphBuilder;
using Graph::LoweringContext;
using Graph::ReplayPrelude;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Answers every query with the form it was built with. */
    class FixedSelector final : public Graph::FusionSelector
    {
        public:
            /** @brief Builds a selector that always answers @p form. */
            explicit FixedSelector(FusedForm form) noexcept : form_(form)
            {

            }

            /** @brief Returns the fixed form with a fixed reason. */
            FusionChoice choose(const FusionQuery&) const override
            {
                return FusionChoice{form_, "the test fixed the answer"};
            }

        private:
            /** @brief Holds the form every answer names. */
            FusedForm form_;
    };

    /** @brief Gives each case a graph of one ReLU over a [2, 3] input, a pool and pass state. */
    class Medium_LoweringContext : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces the ReLU and binds its input. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                const GraphBuilder builder(graph);
                x = builder.input(Shape{2, 3}, "x");
                y = builder.relu(x);
                node = graph.value(y).producer;
                pool = std::make_unique<TensorPool>(*engine);
                SA_EXPECT_OK(pool->materialise_leaves(graph));
                recording = std::make_unique<SaTest::TestGraph>(*engine);
            }

            /** @brief Releases the graph and the pool before the Engine they draw from. */
            void TearDown() override
            {
                recording.reset();
                pool.reset();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Builds a context over the ReLU node with no list of preludes. */
            std::unique_ptr<LoweringContext> context_over(const Graph::FusionSelector& chooser)
            {
                return std::make_unique<LoweringContext>(recording->recorder(), engine->rng(),
                                                         *pool, graph, node, chooser,
                                                         selections, emitted);
            }

            /** @brief Holds the traced graph. */
            Graph::Graph graph;

            /** @brief Holds the input of the ReLU. */
            ValueID x = Graph::K_NO_VALUE;

            /** @brief Holds the result of the ReLU. */
            ValueID y = Graph::K_NO_VALUE;

            /** @brief Holds the ReLU node. */
            Graph::NodeID node = Graph::K_NO_NODE;

            /** @brief Holds the pool the context hands tensors out of. */
            std::unique_ptr<TensorPool> pool;

            /** @brief Holds the graph whose recorder the context records through. */
            std::unique_ptr<SaTest::TestGraph> recording;

            /** @brief Holds the selector a context consults; it takes a GEMM, bias and ReLU. */
            FixedSelector selector{FusedForm::GEMMBiasReLU};

            /** @brief Holds the pass's list of fusion decisions. */
            std::vector<FusionSelection> selections;

            /** @brief Holds the pass's count of recorded tasks. */
            std::size_t emitted = 0;
    };
} // namespace

/** @brief Tests that the transpose flag maps onto the two SushiBLAS enumerators. */
TEST(Small_LoweringContext, TheTransposeFlagMapsOntoSushiBLAS)
{
    static_assert(Graph::transpose_of(true) == SushiBLAS::Core::Transpose::Trans,
                  "true reads the operand transposed");
    static_assert(Graph::transpose_of(false) == SushiBLAS::Core::Transpose::NoTrans,
                  "false reads it as stored");
    SUCCEED();
}

/** @brief Tests that the context names its node, its operation and the values it reads. */
TEST_F(Medium_LoweringContext, TheContextDescribesItsNode)
{
    const std::unique_ptr<LoweringContext> context = context_over(selector);

    EXPECT_EQ(context->id(), node);
    EXPECT_EQ(&context->graph(), &graph);
    EXPECT_EQ(context->node().op, Ops::K_RELU.id);
    EXPECT_EQ(context->name(), Ops::K_RELU.name);
    EXPECT_EQ(context->input_value(0).shape, Shape({2, 3}));
    EXPECT_EQ(context->output_value(0).shape, Shape({2, 3}));
    EXPECT_EQ(&context->input(0), &pool->tensor(x));
}

/** @brief Tests that a result gets its tensor on the first request and keeps it after. */
TEST_F(Medium_LoweringContext, AResultIsAllocatedOnTheFirstRequest)
{
    const std::unique_ptr<LoweringContext> context = context_over(selector);

    EXPECT_FALSE(context->output_is_bound(0));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & first, context->output(0));
    EXPECT_TRUE(context->output_is_bound(0));
    EXPECT_TRUE(pool->is_bound(y));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & second, context->output(0));
    EXPECT_EQ(&first, &second);
}

/** @brief Tests that an aliased result shares the storage of the operand it names. */
TEST_F(Medium_LoweringContext, AnAliasedResultSharesItsOperandsStorage)
{
    const std::unique_ptr<LoweringContext> context = context_over(selector);

    SA_EXPECT_OK(context->alias_output(0, 0));

    ASSERT_TRUE(context->output_is_bound(0));
    EXPECT_EQ(pool->tensor(y).storage, pool->tensor(x).storage);
}

/** @brief Tests that count() adds to the pass's own counter, which emitted() reads. */
TEST_F(Medium_LoweringContext, CountAddsToThePassTotal)
{
    emitted = 2;
    const std::unique_ptr<LoweringContext> context = context_over(selector);

    context->count(3);

    EXPECT_EQ(context->emitted(), 5u);
    EXPECT_EQ(emitted, 5u);
}

/** @brief Tests that select() records the decision and close_selection() sets its task count. */
TEST_F(Medium_LoweringContext, ASelectionIsRecordedWithTheTasksItCost)
{
    const std::unique_ptr<LoweringContext> context = context_over(selector);

    FusionQuery query;
    query.matched = FusedForm::GEMMBiasReLU;
    query.rows = 2;
    query.columns = 3;

    const std::size_t before = context->emitted();
    const FusionChoice choice = context->select(query);
    context->count(4);
    context->close_selection(before);

    EXPECT_EQ(choice.form, FusedForm::GEMMBiasReLU);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].matched, FusedForm::GEMMBiasReLU);
    EXPECT_EQ(selections[0].chosen, FusedForm::GEMMBiasReLU);
    EXPECT_FALSE(selections[0].reason.empty());
    EXPECT_EQ(selections[0].tasks, 4u);
}

/** @brief Tests that a selector may narrow a GEMM epilogue to the plain bias form. */
TEST_F(Medium_LoweringContext, ASelectorMayNarrowAGEMMEpilogueToTheBiasForm)
{
    const FixedSelector narrowing(FusedForm::GEMMBias);
    const std::unique_ptr<LoweringContext> context = context_over(narrowing);

    FusionQuery query;
    query.matched = FusedForm::GEMMBiasTanh;

    EXPECT_EQ(context->select(query).form, FusedForm::GEMMBias);
    ASSERT_EQ(selections.size(), 1u);
    EXPECT_EQ(selections[0].matched, FusedForm::GEMMBiasTanh);
    EXPECT_EQ(selections[0].chosen, FusedForm::GEMMBias);
}

/** @brief Tests that a prelude is refused without a list and appended when there is one. */
TEST_F(Medium_LoweringContext, APreludeNeedsAListToGoInto)
{
    std::uint64_t seen = 0;
    const ReplayPrelude prelude = [&seen](std::uint64_t run)
    {
        seen = run;
    };

    const std::unique_ptr<LoweringContext> without = context_over(selector);
    SA_EXPECT_FAILURE(without->before_each_replay(prelude), StatusCode::Unsupported,
                      "CompiledStep");

    std::vector<ReplayPrelude> preludes;
    const LoweringContext with(recording->recorder(), engine->rng(), *pool, graph, node,
                               selector, selections, emitted, preludes);
    SA_EXPECT_OK(with.before_each_replay(prelude));

    ASSERT_EQ(preludes.size(), 1u);
    preludes[0](7);
    EXPECT_EQ(seen, 7u);
}
