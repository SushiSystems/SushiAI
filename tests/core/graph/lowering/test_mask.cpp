/**************************************************************************/
/* test_mask.cpp                                                          */
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
 * @file test_mask.cpp
 * @brief Tests the causal mask lowering on a device: a HALF mask holds the lowest finite value.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/mask.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/test_names.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::LoweringRule;
using Graph::TensorPool;
using Graph::ValueID;
using SaTest::read_tensor;
using SaTest::write_tensor;

namespace
{
    /** @brief Provides the Engine every causal mask case records on. */
    class Medium_CausalMaskOperations : public SaTest::EngineTest
    {

    };

    /**
     * @brief Returns @p count values of a fixed pattern, scale * ((i * stride) % 11 - 5).
     *
     * The pattern is deterministic, so no seed is needed, and it is symmetric about zero.
     */
    std::vector<double> patterned(std::size_t count, std::size_t stride, double scale)
    {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = scale * (static_cast<double>((i * stride) % 11) - 5.0);
        return values;
    }

    /**
     * @brief Lowers @p graph into a step over @p pool and runs it once.
     *
     * The leaves of @p graph must already be materialised in @p pool and written.
     */
    void compile_and_run(SushiAI::ExecutionBackend& backend, const Graph::Graph& graph,
                         TensorPool& pool)
    {
        std::unique_ptr<CompiledStep> step;
        SA_ASSERT_OK_AND_ASSIGN(step, CompiledStep::compile(backend, graph, pool));
        SA_EXPECT_OK(step->run());
    }

    /** @brief Pairs an operation with the row the header names for it. */
    struct ExpectedRow
    {
        OpID op;           /**< The operation. */
        LoweringRule rule; /**< The row that lowers it. */
    };

    /** @brief Runs once per row the header names for the table. */
    class Small_MaskRow : public ::testing::TestWithParam<ExpectedRow>
    {

    };

    /** @brief Lists the rows of the table. */
    const std::array<ExpectedRow, 1> K_EXPECTED_ROWS = {{
        {Ops::K_CAUSAL_MASK.id, &Graph::lower_causal_mask},
    }};

    /** @brief Returns the row of @p op in the table, or nullptr when none names it. */
    const Graph::LoweringEntry* find_entry(OpID op)
    {
        for (const Graph::LoweringEntry& entry : Graph::K_MASK_LOWERINGS)
        {
            if (entry.op == op)
                return &entry;
        }
        return nullptr;
    }
}

/** @brief Tests that a HALF causal mask holds the lowest finite HALF above the diagonal. */
TEST_F(Medium_CausalMaskOperations, HalfCausalMaskFillsTheUpperTriangleWithTheLowestFiniteValue)
{
    constexpr std::size_t K_BATCHES = 2;
    constexpr std::size_t K_LENGTH = 4;
    constexpr double K_REQUESTED_FILL = -1.0e9;
    constexpr double K_LOWEST_HALF = -65504.0;

    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID x = builder.input(Shape{2, 4, 4}, "x");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID narrowed, builder.cast(x, DType::Half));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID masked, builder.causal_mask(narrowed, K_REQUESTED_FILL));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID y, builder.cast(masked, DType::Float32));

    const std::vector<double> scores = patterned(K_BATCHES * K_LENGTH * K_LENGTH, 3, 0.25);

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    write_tensor(pool.tensor(x), scores);

    ASSERT_NO_FATAL_FAILURE(compile_and_run(backend(), graph, pool));

    const std::vector<double> got = read_tensor(pool.tensor(y));
    ASSERT_EQ(got.size(), scores.size());

    for (std::size_t batch = 0; batch < K_BATCHES; ++batch)
    {
        for (std::size_t row = 0; row < K_LENGTH; ++row)
        {
            for (std::size_t column = 0; column < K_LENGTH; ++column)
            {
                const std::size_t at = (batch * K_LENGTH + row) * K_LENGTH + column;
                EXPECT_FALSE(std::isinf(got[at])) << "at " << at;
                if (column > row)
                    EXPECT_EQ(got[at], K_LOWEST_HALF) << "at " << at;
                else
                    EXPECT_EQ(got[at], scores[at]) << "at " << at;
            }
        }
    }
}

/** @brief Tests that the table holds one row for each row the header names, and no other. */
TEST(Small_MaskLowerings, TheTableHoldsExactlyTheRowsTheHeaderNames)
{
    EXPECT_EQ(Graph::K_MASK_LOWERINGS.size(), K_EXPECTED_ROWS.size());
}

/** @brief Tests that a row pairs its operation with the function the header names. */
TEST_P(Small_MaskRow, EachRowPairsItsOperationWithItsLowering)
{
    const ExpectedRow row = GetParam();
    const std::string name(Ops::name_of(row.op));
    const Graph::LoweringEntry* entry = find_entry(row.op);
    ASSERT_NE(entry, nullptr) << name;

    EXPECT_EQ(entry->rule, row.rule) << name;
    EXPECT_EQ(entry->scratch, nullptr) << name;
    EXPECT_EQ(Graph::find_lowering(row.op), row.rule) << name;
}

INSTANTIATE_TEST_SUITE_P(, Small_MaskRow, ::testing::ValuesIn(K_EXPECTED_ROWS),
                         [](const ::testing::TestParamInfo<ExpectedRow>& info)
                         {
                             std::string name(Ops::name_of(info.param.op));
                             name.replace(0, name.find('.') + 1, "");
                             return ::SushiAI::Test::parameter_name(name);
                         });
