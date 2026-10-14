/**************************************************************************/
/* test_region_executor.cpp                                               */
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
 * @file test_region_executor.cpp
 * @brief Tests that RegionExecutor builds, keeps, re-records and drops regions as a step says.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_executor.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/graph/ir.hpp>

#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>

#include "eager_run_support.hpp"
#include "eager_test_support.hpp"

using namespace SushiAI;
using Eager::RegionAction;
using Eager::RegionExecutor;
using Eager::StepPlan;
using Eager::StepWork;
using Eager::StorageID;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Unit_RegionExecutor : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a FLOAT32 result against the same sum computed in double. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Plans host-traced steps and hands them to an executor, as a session does. */
    struct Harness
    {
        /** @brief Builds the storage, the table and the executor on @p engine. */
        explicit Harness(SushiBLAS::Engine& engine)
            : bank(engine), table(ids), executor(engine)
        {

        }

        /**
         * @brief Plans @p step and returns the work an executor is handed for it.
         * @return The work, with a filled tensor behind every leaf.
         */
        [[nodiscard]] Result<StepWork> prepare(const Support::TracedStep& step)
        {
            const Eager::BoundaryMap boundaries = Support::boundaries_of(step);
            const std::vector<StorageID> preset = Support::presets_of(step);
            for (std::size_t value = 0; value < preset.size(); ++value)
            {
                if (preset[value] == Eager::K_NO_STORAGE || bank.holds(preset[value]))
                    continue;

                const Graph::Value& leaf = step.graph.value(static_cast<Graph::ValueID>(value));
                SA_TRY(bank.allocate(preset[value], leaf.shape, leaf.dtype));
                Support::fill_wave(bank.tensor(preset[value]), static_cast<double>(value), 0.5);
            }

            plan = table.plan(step.graph, step.partition, boundaries,
                              Support::span_of(preset));
            for (const Eager::SlotRecord& record : plan.created)
                SA_TRY(bank.allocate(record.id, record.shape, record.dtype));

            StepWork work;
            work.regions.resize(plan.regions.size());
            for (std::size_t region = 0; region < plan.regions.size(); ++region)
            {
                work.regions[region].order = plan.regions[region];
                if (plan.regions[region].action == RegionAction::Build)
                    work.regions[region].program = Eager::compile_region(
                        step.graph, step.partition, plan.regions[region].key,
                        boundaries.regions[region]);
            }

            work.dropped = plan.dropped;
            work.storage_of_value = Support::span_of(plan.storage_of_value);
            work.run_index = runs++;
            return work;
        }

        /** @brief Plans @p step and executes it. */
        [[nodiscard]] Status run(const Support::TracedStep& step)
        {
            SA_TRY_ASSIGN(StepWork work, prepare(step));
            return executor.execute(work, bank);
        }

        /** @brief Returns the tensor the last planned step placed @p value in. */
        [[nodiscard]] SushiBLAS::Tensor& tensor_of(Graph::ValueID value)
        {
            return bank.tensor(plan.storage_of_value[value]);
        }

        /** @brief Returns relu(x * weight + bias) of the last planned linear_then_relu(). */
        [[nodiscard]] std::vector<double> expected(const Support::TracedStep& step)
        {
            // The leaves of linear_then_relu() are traced in the order x, weight, bias.
            const std::vector<double> x =
                SaTest::read_tensor(bank.tensor(Support::K_LEAF_STORAGE));
            const std::vector<double> weight = SaTest::read_tensor(tensor_of(step.parameters[0]));
            const std::vector<double> bias = SaTest::read_tensor(tensor_of(step.parameters[1]));
            return Support::linear_relu(x, weight, bias);
        }

        Eager::StorageIDSource ids; /**< Hands out the id of every boundary slot. */
        Eager::StorageBank bank;    /**< Holds every tensor the steps name. */
        Eager::RegionTable table;   /**< Compares each step with the one before. */
        StepPlan plan;              /**< The decisions of the last planned step. */
        RegionExecutor executor;    /**< The executor under test. */
        std::uint64_t runs = 0;     /**< Counts the steps prepared. */
    };

    /** @brief Expects @p result to equal @p expected within K_TOLERANCE, element by element. */
    void expect_near(const std::vector<double>& result, const std::vector<double>& expected)
    {
        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t i = 0; i < result.size(); ++i)
            EXPECT_NEAR(result[i], expected[i], K_TOLERANCE) << "element " << i;
    }
} // namespace

TEST_F(Unit_RegionExecutor, ABuiltStepWritesItsObservedValue)
{
    Harness harness(*engine);
    const Support::TracedStep step = Support::linear_then_relu(false);

    SA_EXPECT_OK(harness.run(step));

    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
    EXPECT_EQ(harness.executor.region_count(), 1u);
    EXPECT_EQ(harness.executor.emit_count(), 1u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
    EXPECT_GT(harness.executor.operation_count(), 0u);
}

TEST_F(Unit_RegionExecutor, AKeptStepRunsAgainWithoutBeingComposedOrRecorded)
{
    Harness harness(*engine);
    SA_EXPECT_OK(harness.run(Support::linear_then_relu(false)));
    const std::size_t allocated = harness.executor.allocated_bytes();

    // The second step reads another input through the storage the region was recorded with.
    Support::fill_wave(harness.bank.tensor(Support::K_LEAF_STORAGE), 2.5, 1.5);
    const Support::TracedStep again = Support::linear_then_relu(false);
    SA_EXPECT_OK(harness.run(again));

    expect_near(SaTest::read_tensor(harness.tensor_of(again.hidden)), harness.expected(again));
    EXPECT_EQ(harness.executor.emit_count(), 1u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
    EXPECT_EQ(harness.executor.allocated_bytes(), allocated);
}

TEST_F(Unit_RegionExecutor, TwoRegionsRunInKeyOrder)
{
    Harness harness(*engine);
    const Support::TracedStep step = Support::linear_then_relu(true);

    SA_EXPECT_OK(harness.run(step));

    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
    EXPECT_EQ(harness.executor.region_count(), 2u);
    EXPECT_EQ(harness.executor.emit_count(), 2u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
}

TEST_F(Unit_RegionExecutor, ARegionRecordedUnderAnotherKeyKeepsItsProgram)
{
    Harness harness(*engine);
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_EXPECT_OK(harness.run(step));
    const Eager::RegionOrder built = harness.plan.regions[0];
    const Eager::RegionProgram* const program = harness.executor.program_of(built.key);
    ASSERT_NE(program, nullptr);
    const std::size_t nodes = program->graph.node_count();

    // The same region under key 3: the table says Record when only the numbering moved.
    StepWork work;
    work.regions.resize(1);
    work.regions[0].order = built;
    work.regions[0].order.previous_key = built.key;
    work.regions[0].order.key = built.key + 2;
    work.regions[0].order.action = RegionAction::Record;
    work.storage_of_value = Support::span_of(harness.plan.storage_of_value);
    work.run_index = 1;
    SA_EXPECT_OK(harness.executor.execute(work, harness.bank));

    EXPECT_EQ(harness.executor.program_of(built.key), nullptr);
    ASSERT_NE(harness.executor.program_of(built.key + 2), nullptr);
    EXPECT_EQ(harness.executor.program_of(built.key + 2)->graph.node_count(), nodes);
    EXPECT_EQ(harness.executor.emit_count(built.id), 2u);
    EXPECT_EQ(harness.executor.compose_count(), 2u);
    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
}

TEST_F(Unit_RegionExecutor, ARegionThatLeftIsDropped)
{
    Harness harness(*engine);
    SA_EXPECT_OK(harness.run(Support::linear_then_relu(true)));
    ASSERT_EQ(harness.executor.region_count(), 2u);

    const Support::TracedStep joined = Support::linear_then_relu(false);
    SA_EXPECT_OK(harness.run(joined));

    EXPECT_EQ(harness.plan.dropped.size(), 1u);
    EXPECT_EQ(harness.executor.region_count(), 1u);
    expect_near(SaTest::read_tensor(harness.tensor_of(joined.hidden)), harness.expected(joined));
}

TEST_F(Unit_RegionExecutor, TheTrailingWorkRunsBehindEveryRegionOnEveryRun)
{
    Harness harness(*engine);
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_ASSERT_OK_AND_ASSIGN(StepWork work, harness.prepare(step));

    // The update subtracts the region's result from a zeroed tensor of the same shape.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor negated,
                            allocate_like(*engine, harness.tensor_of(step.hidden)));
    SaTest::write_tensor(negated, std::vector<double>(32, 0.0));
    Support::PlainDescent update(1.0f);
    update.add(negated, harness.tensor_of(step.hidden));

    work.trailing = &update;
    work.trailing_key = step.partition.optimizer_key;
    SA_EXPECT_OK(harness.executor.execute(work, harness.bank));

    const std::vector<double> result = SaTest::read_tensor(harness.tensor_of(step.hidden));
    std::vector<double> expected(result.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        expected[i] = -result[i];
    expect_near(SaTest::read_tensor(negated), expected);
    EXPECT_EQ(update.record_count(), 1u);
    EXPECT_EQ(update.replay_count(), 1u);

    // The same update under the same key is replayed, not recorded again.
    SA_ASSERT_OK_AND_ASSIGN(StepWork again, harness.prepare(Support::linear_then_relu(false)));
    again.trailing = &update;
    again.trailing_key = step.partition.optimizer_key;
    SA_EXPECT_OK(harness.executor.execute(again, harness.bank));

    EXPECT_EQ(update.record_count(), 1u);
    EXPECT_EQ(update.replay_count(), 2u);
    EXPECT_EQ(harness.executor.trailing_record_count(), 1u);
}

TEST_F(Unit_RegionExecutor, DiscardDropsEveryRegionAndAKeptRegionIsThenRefused)
{
    Harness harness(*engine);
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_EXPECT_OK(harness.run(step));

    SA_EXPECT_OK(harness.executor.discard());
    EXPECT_EQ(harness.executor.region_count(), 0u);

    // The table still believes the region is recorded, so it plans a Keep.
    SA_EXPECT_FAILURE(harness.run(Support::linear_then_relu(false)),
                      StatusCode::InvalidDescription, "keeps the region");
}
