/**************************************************************************/
/* test_tensor_pool.cpp                                                   */
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
 * @file test_tensor_pool.cpp
 * @brief Tests how a tensor pool hands out, shares and counts the tensors behind graph values.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every pool draws its memory from. */
    class Medium_TensorPool : public SaTest::EngineTest
    {

    };

    /** @brief Bounds the values of a constant fill, which FLOAT32 holds exactly. */
    constexpr double K_EXACT = 0.0;

    /** @brief Returns the bytes of a FLOAT32 tensor of @p elements elements. */
    constexpr std::size_t float_bytes(std::size_t elements)
    {
        return elements * sizeof(float);
    }
} // namespace

/** @brief Tests that a value is allocated on its first request and handed back after. */
TEST_F(Medium_TensorPool, AValueIsAllocatedOnceAndKeepsItsTensor)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{4, 6}, "x");

    TensorPool pool(*engine);
    EXPECT_FALSE(pool.is_bound(x));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & first, pool.materialise(x, graph.value(x)));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & second, pool.materialise(x, graph.value(x)));

    EXPECT_EQ(&first, &second);
    EXPECT_TRUE(pool.is_bound(x));
    EXPECT_EQ(&pool.tensor(x), &first);
    EXPECT_EQ(pool.tensor_count(), 1u);
    EXPECT_EQ(pool.bytes().values, float_bytes(24));
    EXPECT_EQ(pool.allocated_bytes(), pool.unpacked_bytes());
    EXPECT_FALSE(pool.has_plan());
}

/** @brief Tests that the ones tensor is shared per shape and type and holds ones after a flush. */
TEST_F(Medium_TensorPool, OnesAreSharedPerShapeAndTypeAndFilledByTheFlush)
{
    TensorPool pool(*engine);

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & first, pool.ones(Shape{3}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & again, pool.ones(Shape{3}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & wider, pool.ones(Shape{3}, DType::Float64));

    EXPECT_EQ(&first, &again);
    EXPECT_NE(&first, &wider);
    EXPECT_EQ(pool.bytes().ones, float_bytes(3) + 3 * sizeof(double));

    SA_EXPECT_OK(pool.flush_constants());

    for (const double value : SaTest::read_tensor(first))
        EXPECT_NEAR(value, 1.0, K_EXACT);
    for (const double value : SaTest::read_tensor(wider))
        EXPECT_NEAR(value, 1.0, K_EXACT);
}

/** @brief Tests that a scratch buffer is keyed by node and slot and outside a plan owns bytes. */
TEST_F(Medium_TensorPool, ScratchIsKeyedByNodeAndSlot)
{
    TensorPool pool(*engine);

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & first,
                            pool.scratch(4, 0, Shape{2, 2}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & again,
                            pool.scratch(4, 0, Shape{2, 2}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & other_slot,
                            pool.scratch(4, 1, Shape{2, 2}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor & other_node,
                            pool.scratch(5, 0, Shape{2, 2}, DType::Float32));

    EXPECT_EQ(&first, &again);
    EXPECT_NE(&first, &other_slot);
    EXPECT_NE(&first, &other_node);
    EXPECT_EQ(pool.tensor_count(), 3u);
    EXPECT_EQ(pool.bytes().scratch, float_bytes(12));
}

/** @brief Tests that an alias reads what was written through the value it aliases. */
TEST_F(Medium_TensorPool, AnAliasSharesTheStorageOfItsSource)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID source = builder.input(Shape{3}, "source");
    const ValueID target = builder.input(Shape{3}, "target");

    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor created, pool.create(Shape{3}, DType::Float32));
    SA_EXPECT_OK(pool.bind(source, created));
    SA_EXPECT_OK(pool.alias(target, source));
    SA_EXPECT_OK(pool.alias(target, source));

    const std::vector<double> values = {1.5, -2.0, 0.25};
    SaTest::write_tensor(pool.tensor(source), values);

    EXPECT_EQ(pool.tensor(target).storage, pool.tensor(source).storage);
    const std::vector<double> got = SaTest::read_tensor(pool.tensor(target));
    ASSERT_EQ(got.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        EXPECT_NEAR(got[i], values[i], K_EXACT) << "at " << i;
}

/** @brief Tests that the flush writes a queued constant, and that an empty flush succeeds. */
TEST_F(Medium_TensorPool, AQueuedConstantIsWrittenByTheFlush)
{
    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.flush_constants());

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor tensor, pool.create(Shape{4}, DType::Float32));
    SA_EXPECT_OK(pool.request_constant(tensor, 2.5));
    SA_EXPECT_OK(pool.flush_constants());

    for (const double value : SaTest::read_tensor(tensor))
        EXPECT_NEAR(value, 2.5, K_EXACT);
}
