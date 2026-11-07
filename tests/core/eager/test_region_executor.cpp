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
 * @file test_region_executor.cpp
 * @brief Tests that RegionExecutor builds, keeps, re-records and drops regions as a step says.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/core/span.hpp>
#include "eager_run_support.hpp"
#include "eager_test_support.hpp"
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_executor.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/step_executor.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_descent_recorder.hpp>
#include <SushiAI/graph/in_memory_late_increment_recorder.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Eager::RegionAction;
using Eager::RegionExecutor;
using Eager::StepPlan;
using Eager::StepWork;
using Eager::StorageID;
// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_RegionExecutor : public SaTest::EngineTest
    {

    };

    /**
     * @brief Bounds a FLOAT32 result against the same sum computed in double; 256 unit roundoffs of
     * float, the smallest power of two that covers 1e-5.
     */
    constexpr double K_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Plans host-traced steps and hands them to an executor, as a session does. */
    struct Harness
    {
        /** @brief Builds the storage, the table and the executor on @p backend. */
        explicit Harness(RuntimeExecutionBackend& backend)
            : bank(backend.engine()), table(ids), executor(backend)
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
                {
                    SA_TRY_ASSIGN(work.regions[region].program,
                                  Eager::compile_region(step.graph, step.partition,
                                                        plan.regions[region].key,
                                                        boundaries.regions[region]));
                }
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

    /** @brief Returns @p values negated. */
    [[nodiscard]] std::vector<double> negated_of(const std::vector<double>& values)
    {
        std::vector<double> negated(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            negated[i] = -values[i];
        return negated;
    }

    /** @brief Expects @p result to equal @p expected within K_TOLERANCE, element by element. */
    void expect_near(const std::vector<double>& result, const std::vector<double>& expected)
    {
        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t i = 0; i < result.size(); ++i)
            EXPECT_NEAR(result[i], expected[i], K_TOLERANCE) << "element " << i;
    }

    /**
     * @brief Runs linear_then_relu() on @p harness, refills the input and runs it again.
     * @param allocated_after_first Receives the pool bytes the executor held after the first run.
     * @return The step of the second run.
     */
    [[nodiscard]] Support::TracedStep run_kept_step_again(Harness& harness,
                                                          std::size_t& allocated_after_first)
    {
        SA_EXPECT_OK(harness.run(Support::linear_then_relu(false)));
        allocated_after_first = harness.executor.allocated_bytes();

        // The second step reads another input through the storage the region was recorded with.
        Support::fill_wave(harness.bank.tensor(Support::K_LEAF_STORAGE), 2.5, 1.5);
        Support::TracedStep again = Support::linear_then_relu(false);
        SA_EXPECT_OK(harness.run(again));
        return again;
    }

    /** @brief Holds a region built under its table key and the node count of its program. */
    struct BuiltRegion
    {
        Eager::RegionOrder order; /**< The order the table gave the region. */
        std::size_t nodes = 0;    /**< The node count of the region's program. */
    };

    /**
     * @brief Builds @p step on @p harness, then records its region again under a key raised by 2.
     * @return The order the region was built with and the node count of its program.
     */
    [[nodiscard]] BuiltRegion run_then_rekey(Harness& harness, const Support::TracedStep& step)
    {
        SA_EXPECT_OK(harness.run(step));
        BuiltRegion built;
        built.order = harness.plan.regions[0];
        const Eager::RegionProgram* const program = harness.executor.program_of(built.order.key);
        EXPECT_NE(program, nullptr);
        if (program != nullptr)
            built.nodes = program->graph.node_count();

        // The same region under another key: the table says Record when only the numbering moved.
        StepWork work;
        work.regions.resize(1);
        work.regions[0].order = built.order;
        work.regions[0].order.previous_key = built.order.key;
        work.regions[0].order.key = built.order.key + 2;
        work.regions[0].order.action = RegionAction::Record;
        work.storage_of_value = Support::span_of(harness.plan.storage_of_value);
        work.run_index = 1;
        SA_EXPECT_OK(harness.executor.execute(work, harness.bank));
        return built;
    }

    /** @brief Hands @p update to a fresh step of @p harness as its trailing work and runs it. */
    void run_with_trailing(Harness& harness, Graph::StepRecorder& update)
    {
        const Support::TracedStep step = Support::linear_then_relu(false);
        SA_ASSERT_OK_AND_ASSIGN(StepWork work, harness.prepare(step));
        work.trailing = &update;
        work.trailing_key = step.partition.optimizer_key;
        SA_EXPECT_OK(harness.executor.execute(work, harness.bank));
    }
} // namespace

/** @brief Tests that a built step writes the value the caller observes. */
TEST_F(Medium_RegionExecutor, ABuiltStepWritesItsObservedValue)
{
    Harness harness(backend());
    const Support::TracedStep step = Support::linear_then_relu(false);

    SA_EXPECT_OK(harness.run(step));

    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
    EXPECT_EQ(harness.executor.region_count(), 1u);
    EXPECT_EQ(harness.executor.emit_count(), 1u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
    EXPECT_GT(harness.executor.operation_count(), 0u);
}

/** @brief Tests that a kept step reads the storage its kernels were recorded with. */
TEST_F(Medium_RegionExecutor, AKeptStepFollowsTheInputItsKernelsRead)
{
    Harness harness(backend());
    std::size_t allocated = 0;

    const Support::TracedStep again = run_kept_step_again(harness, allocated);

    expect_near(SaTest::read_tensor(harness.tensor_of(again.hidden)), harness.expected(again));
}

/** @brief Tests that a kept step is neither composed nor recorded again. */
TEST_F(Medium_RegionExecutor, AKeptStepIsNeitherComposedNorRecordedAgain)
{
    Harness harness(backend());
    std::size_t allocated = 0;

    (void)run_kept_step_again(harness, allocated);

    EXPECT_EQ(harness.executor.emit_count(), 1u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
}

/** @brief Tests that a kept step allocates no pool byte. */
TEST_F(Medium_RegionExecutor, AKeptStepAllocatesNoPoolByte)
{
    Harness harness(backend());
    std::size_t allocated = 0;

    (void)run_kept_step_again(harness, allocated);

    EXPECT_EQ(harness.executor.allocated_bytes(), allocated);
}

/** @brief Tests that two regions give the result of one, so the lower key ran first. */
TEST_F(Medium_RegionExecutor, TwoRegionsRunInKeyOrder)
{
    Harness harness(backend());
    const Support::TracedStep step = Support::linear_then_relu(true);

    SA_EXPECT_OK(harness.run(step));

    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
    EXPECT_EQ(harness.executor.region_count(), 2u);
    EXPECT_EQ(harness.executor.emit_count(), 2u);
    EXPECT_EQ(harness.executor.compose_count(), 1u);
}

/** @brief Tests that a region recorded under another key keeps its program under the new key. */
TEST_F(Medium_RegionExecutor, ARegionRecordedUnderAnotherKeyKeepsItsProgram)
{
    Harness harness(backend());
    const BuiltRegion built = run_then_rekey(harness, Support::linear_then_relu(false));

    EXPECT_EQ(harness.executor.program_of(built.order.key), nullptr);
    ASSERT_NE(harness.executor.program_of(built.order.key + 2), nullptr);
    EXPECT_EQ(harness.executor.program_of(built.order.key + 2)->graph.node_count(), built.nodes);
}

/** @brief Tests that a region recorded under another key is lowered and composed again. */
TEST_F(Medium_RegionExecutor, ARegionRecordedUnderAnotherKeyIsLoweredAndComposedAgain)
{
    Harness harness(backend());
    const BuiltRegion built = run_then_rekey(harness, Support::linear_then_relu(false));

    EXPECT_EQ(harness.executor.emit_count(built.order.id), 2u);
    EXPECT_EQ(harness.executor.compose_count(), 2u);
}

/** @brief Tests that a region recorded under another key still writes the host result. */
TEST_F(Medium_RegionExecutor, ARegionRecordedUnderAnotherKeyKeepsItsResult)
{
    Harness harness(backend());
    const Support::TracedStep step = Support::linear_then_relu(false);

    (void)run_then_rekey(harness, step);

    expect_near(SaTest::read_tensor(harness.tensor_of(step.hidden)), harness.expected(step));
}

/** @brief Tests that the step which joins two regions drops the one that left. */
TEST_F(Medium_RegionExecutor, ARegionThatLeftIsDropped)
{
    Harness harness(backend());
    SA_EXPECT_OK(harness.run(Support::linear_then_relu(true)));
    ASSERT_EQ(harness.executor.region_count(), 2u);

    const Support::TracedStep joined = Support::linear_then_relu(false);
    SA_EXPECT_OK(harness.run(joined));

    EXPECT_EQ(harness.plan.dropped.size(), 1u);
    EXPECT_EQ(harness.executor.region_count(), 1u);
    expect_near(SaTest::read_tensor(harness.tensor_of(joined.hidden)), harness.expected(joined));
}

/** @brief Tests that the executor tells the trailing recorder each run has finished. */
TEST_F(Medium_RegionExecutor, TheTrailingRecorderIsToldAfterEveryRun)
{
    Harness harness(backend());
    Graph::InMemoryStepRecorder update;

    run_with_trailing(harness, update);
    run_with_trailing(harness, update);

    const std::vector<Graph::InMemoryStepRecorder::Call>& calls = update.calls();
    EXPECT_EQ(
        std::count(calls.begin(), calls.end(), Graph::InMemoryStepRecorder::Call::AfterReplay),
        2);
}

/** @brief Tests that the trailing work runs behind every region of the step. */
TEST_F(Medium_RegionExecutor, TheTrailingWorkRunsBehindEveryRegion)
{
    Harness harness(backend());
    const Support::TracedStep step = Support::linear_then_relu(false);
    SA_ASSERT_OK_AND_ASSIGN(StepWork work, harness.prepare(step));

    // The update subtracts the region's result from a zeroed tensor of the same shape.
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor negated,
                            allocate_like(*engine, harness.tensor_of(step.hidden)));
    SaTest::write_tensor(negated, std::vector<double>(32, 0.0));
    Graph::InMemoryDescentRecorder update(1.0f);
    update.add(negated, harness.tensor_of(step.hidden));

    work.trailing = &update;
    work.trailing_key = step.partition.optimizer_key;
    SA_EXPECT_OK(harness.executor.execute(work, harness.bank));

    const std::vector<double> result = SaTest::read_tensor(harness.tensor_of(step.hidden));
    expect_near(SaTest::read_tensor(negated), negated_of(result));
}

/** @brief Tests that a late trailing task replays after the recorder that made it is gone. */
TEST_F(Medium_RegionExecutor, ALateTrailingTaskReplaysAfterItsRecorderIsGone)
{
    Harness harness(backend());
    SushiBLAS::Tensor out = engine->create_tensor({4}, SushiBLAS::Core::DataType::Float32);
    SaTest::write_tensor(out, std::vector<double>(4, 0.0));
    SushiBLAS::Tensor count = engine->create_tensor({1}, SushiBLAS::Core::DataType::Int32,
                                                    SushiRuntime::Residency::Host);
    count.data_as<std::int32_t>()[0] = 2;
    Graph::InMemoryLateIncrementRecorder update(out, count);

    // The first step records the update through a recorder that dies with record_trailing().
    run_with_trailing(harness, update);
    run_with_trailing(harness, update);

    expect_near(SaTest::read_tensor(out), {2.0, 2.0, 0.0, 0.0});
}

/** @brief Tests that discard() drops every region the executor holds. */
TEST_F(Medium_RegionExecutor, DiscardDropsEveryRegion)
{
    Harness harness(backend());
    SA_EXPECT_OK(harness.run(Support::linear_then_relu(false)));

    SA_EXPECT_OK(harness.executor.discard());

    EXPECT_EQ(harness.executor.region_count(), 0u);
}

/** @brief Tests that a Keep planned after discard() is refused with InvalidDescription. */
TEST_F(Medium_RegionExecutor, AKeptRegionAfterDiscardIsRefused)
{
    Harness harness(backend());
    SA_EXPECT_OK(harness.run(Support::linear_then_relu(false)));
    SA_EXPECT_OK(harness.executor.discard());

    // The table still believes the region is recorded, so it plans a Keep.
    SA_EXPECT_FAILURE(harness.run(Support::linear_then_relu(false)),
                      StatusCode::InvalidDescription, "keeps the region");
}
