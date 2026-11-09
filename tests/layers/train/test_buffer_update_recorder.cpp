/**************************************************************************/
/* test_buffer_update_recorder.cpp                                        */
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
 * @file test_buffer_update_recorder.cpp
 * @brief Tests the binding of buffer updates and the blend a recorder replays into a buffer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/execution/step_tasks.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The weight the blend gives the buffer's current content. */
    constexpr double K_MOMENTUM = 0.1;

    /** @brief The factor the batch value is scaled by before it is blended in. */
    constexpr double K_SOURCE_SCALE = 2.0;

    /**
     * @brief The rounded operations on the longest path to one blended element: the scale
     *        rounds 1 - momentum and then its product with the running value, the axpy rounds
     *        momentum * scale and then its product with the batch value, and the sum of the two
     *        rounds once; each path is 3 deep, and the paths run side by side.
     */
    constexpr std::size_t K_BLEND_OPERATIONS = 3;

    /** @brief The state identity the bind cases give the buffer their update targets. */
    constexpr std::uint64_t K_TARGET_STATE = 7;

    /** @brief The value identity the bind cases bind their update sources under. */
    constexpr Graph::ValueID K_SOURCE_VALUE = 0;

    /** @brief Allocates a vector of @p values with dtype @p dtype and fills it with them. */
    Result<SushiBLAS::Tensor> vector_of(SushiBLAS::Engine& engine,
                                        const std::vector<double>& values, DType dtype)
    {
        SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                      allocate_tensor(engine,
                                      Shape{static_cast<std::int64_t>(values.size())}, dtype));
        SaTest::write_tensor(tensor,
                             SushiRuntime::span<const double>(values.data(), values.size()));
        return std::move(tensor);
    }

    /** @brief Builds a buffer entry of a given name, identity and dtype, of extent two. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, DType dtype)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.kind = NN::StateKind::Buffer;
        entry.state = Graph::StateID{state};
        entry.shape = Shape{2};
        entry.dtype = dtype;
        return entry;
    }

    /** @brief Provides the host backend the targets, the sources and the recordings live on. */
    class HostBlend
    {
        protected:
            /** @brief Replays @p tasks once and returns its status. */
            static Status replay_once(StepTasks& tasks)
            {
                SushiRuntime::RunReport report;
                return tasks.replay(report);
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Records and replays buffer updates on the host backend. */
    class Small_BufferUpdateRecorder : public HostBlend, public ::testing::Test
    {
    };

    /** @brief Holds a count of slots; each blends a two-element source into a two-element target. */
    struct SlotCountRow
    {
        std::string name;               /**< The row's name. */
        std::size_t slots;              /**< The slots the recorder holds. */
        std::size_t recorded_tasks;     /**< The tasks it records: two for each slot. */
    };

    /** @brief Records a recorder of a given slot count. */
    class Small_BufferUpdateRecorderCount : public HostBlend,
                                            public ::testing::TestWithParam<SlotCountRow>
    {
    };

    /** @brief Holds one element of a running statistic and the batch statistic blended into it. */
    struct BlendRow
    {
        std::string name;       /**< The row's name. */
        double running;         /**< The element the target holds. */
        double batch;           /**< The element the source holds. */
    };

    /** @brief Replays one blend of one element. */
    class Small_BufferUpdateBlend : public HostBlend, public ::testing::TestWithParam<BlendRow>
    {
    };
} // namespace

/** @brief Tests that a recorder records two tasks for every slot it holds, and none for none. */
TEST_P(Small_BufferUpdateRecorderCount, TwoTasksPerSlotAndNoneForNoSlot)
{
    const SlotCountRow& row = GetParam();
    std::vector<SushiBLAS::Tensor> tensors;
    std::vector<Train::BufferUpdateSlot> slots;
    for (std::size_t slot = 0; slot < row.slots; ++slot)
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target,
                                vector_of(backend.engine(), {1.0, 2.0}, DType::Float64));
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                                vector_of(backend.engine(), {3.0, 4.0}, DType::Float64));
        slots.push_back(Train::BufferUpdateSlot{target, source, K_MOMENTUM, K_SOURCE_SCALE});
    }
    Train::BufferUpdateRecorder recorder(std::move(slots));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t recorded, recorder.record_into_step(step));

    EXPECT_EQ(recorder.slot_count(), row.slots);
    EXPECT_EQ(recorded, row.recorded_tasks);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BufferUpdateRecorderCount,
    ::testing::Values(SlotCountRow{"NoSlot", 0, 0}, SlotCountRow{"OneSlot", 1, 2},
                      SlotCountRow{"TwoSlots", 2, 4}),
    [](const ::testing::TestParamInfo<SlotCountRow>& info) { return info.param.name; });

/**
 * @brief Tests that one replay of a recorder leaves the target at the value
 *        NN::updated_statistic gives for its running and batch elements.
 */
TEST_P(Small_BufferUpdateBlend, OneReplayBlendsTheSourceIntoTheTarget)
{
    const BlendRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target,
                            vector_of(backend.engine(), {row.running}, DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(backend.engine(), {row.batch}, DType::Float64));

    std::vector<Train::BufferUpdateSlot> slots;
    slots.push_back(Train::BufferUpdateSlot{target, source, K_MOMENTUM, K_SOURCE_SCALE});
    Train::BufferUpdateRecorder recorder(std::move(slots));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StepTasks> tasks, backend.make_tasks());
    const Graph::StepRecording step(tasks->recorder(), tasks->runtime_recorder());
    SA_EXPECT_OK(recorder.record_into_step(step));
    SA_EXPECT_OK(tasks->seal());
    SA_EXPECT_OK(replay_once(*tasks));

    const std::vector<double> blended = SaTest::read_tensor(target);
    ASSERT_EQ(blended.size(), 1u);
    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
        blended.front(),
        NN::updated_statistic(row.running, row.batch, K_MOMENTUM, K_SOURCE_SCALE),
        K_BLEND_OPERATIONS)));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BufferUpdateBlend,
    ::testing::Values(BlendRow{"AboveTheBatch", 1.0, 0.25}, BlendRow{"BelowTheBatch", -2.0, 3.0},
                      BlendRow{"BatchBelowZero", 0.5, -1.0}, BlendRow{"BatchAboveRunning", 4.0, 2.0}),
    [](const ::testing::TestParamInfo<BlendRow>& info) { return info.param.name; });

/** @brief Tests that binding refuses an update whose target the state does not hold. */
TEST_F(Small_BufferUpdateRecorder, BindRefusesATargetTheStateLacks)
{
    std::vector<NN::StateEntry> entries;
    entries.push_back(make_entry("bn.running_mean", 1, DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(backend.engine(), std::move(entries)));

    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(backend.engine(), {1.0, 2.0}, DType::Float64));
    SA_EXPECT_OK(pool.bind(K_SOURCE_VALUE, std::move(source)));

    NN::BufferUpdateList updates;
    updates.add(NN::BufferUpdate{Graph::StateID{K_TARGET_STATE}, K_SOURCE_VALUE, K_MOMENTUM,
                                 K_SOURCE_SCALE});

    SA_EXPECT_FAILURE(Train::bind_buffer_updates(updates, state, pool),
                      StatusCode::Incompatible, "no entry");
}

/** @brief Tests that binding refuses an update whose target and source differ in dtype. */
TEST_F(Small_BufferUpdateRecorder, BindRefusesADTypeMismatch)
{
    std::vector<NN::StateEntry> entries;
    entries.push_back(make_entry("bn.running_mean", K_TARGET_STATE, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(backend.engine(), std::move(entries)));

    Graph::TensorPool pool(backend.engine());
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(backend.engine(), {1.0, 2.0}, DType::Float64));
    SA_EXPECT_OK(pool.bind(K_SOURCE_VALUE, std::move(source)));

    NN::BufferUpdateList updates;
    updates.add(NN::BufferUpdate{Graph::StateID{K_TARGET_STATE}, K_SOURCE_VALUE, K_MOMENTUM,
                                 K_SOURCE_SCALE});

    SA_EXPECT_FAILURE(Train::bind_buffer_updates(updates, state, pool),
                      StatusCode::Incompatible, dtype_name(DType::Float32),
                      dtype_name(DType::Float64));
}
