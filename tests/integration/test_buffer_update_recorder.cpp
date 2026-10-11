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
 * @file test_buffer_update_recorder.cpp
 * @brief Tests the binding of buffer updates and the blend a recorder replays into a buffer.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

using SushiAI::Detail::record_and_run;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_BufferUpdateRecorder : public SaTest::EngineTest {};

    /** @brief The weight the blend gives the buffer's current content. */
    constexpr double K_MOMENTUM = 0.1;

    /** @brief The factor the batch value is scaled by before it is blended in. */
    constexpr double K_SOURCE_SCALE = 2.0;

    /** @brief The absolute tolerance of one blended element, in double precision. */
    constexpr double K_BLEND_TOLERANCE = 1e-9;

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

} // namespace

/** @brief Tests that a recorder records two device tasks for every slot it holds. */
TEST_F(Integration_BufferUpdateRecorder, RecordsTwoTasksPerSlot)
{
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target,
                            vector_of(*engine, {1.0, 2.0}, DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(*engine, {3.0, 4.0}, DType::Float64));

    std::vector<Train::BufferUpdateSlot> slots;
    slots.push_back(Train::BufferUpdateSlot{target, source, K_MOMENTUM, K_SOURCE_SCALE});
    slots.push_back(Train::BufferUpdateSlot{source, target, K_MOMENTUM, K_SOURCE_SCALE});
    Train::BufferUpdateRecorder recorder(std::move(slots));

    std::size_t recorded = 0;
    const Status run = record_and_run(*engine, [&]() -> Status
    {
        SA_TRY_ASSIGN(const std::size_t count,
                      recorder.record_into_step(engine->recorder()));
        recorded = count;
        return Status();
    });
    SA_EXPECT_OK(run);
    EXPECT_EQ(recorder.slot_count(), 2u);
    EXPECT_EQ(recorded, 4u);
}

/**
 * @brief Tests that one replay of a recorder leaves each target at the value
 *        NN::updated_statistic gives for its running and batch elements.
 */
TEST_F(Integration_BufferUpdateRecorder, OneReplayBlendsTheSourceIntoTheTarget)
{
    const std::vector<double> running = {1.0, -2.0, 0.5, 4.0};
    const std::vector<double> batch = {0.25, 3.0, -1.0, 2.0};
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor target,
                            vector_of(*engine, running, DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(*engine, batch, DType::Float64));

    std::vector<Train::BufferUpdateSlot> slots;
    slots.push_back(Train::BufferUpdateSlot{target, source, K_MOMENTUM, K_SOURCE_SCALE});
    Train::BufferUpdateRecorder recorder(std::move(slots));

    const Status run = record_and_run(*engine, [&]() -> Status
    {
        SA_TRY_ASSIGN(const std::size_t count,
                      recorder.record_into_step(engine->recorder()));
        static_cast<void>(count);
        return Status();
    });
    SA_EXPECT_OK(run);

    const std::vector<double> blended = SaTest::read_tensor(target);
    ASSERT_EQ(blended.size(), running.size());
    for (std::size_t i = 0; i < running.size(); ++i)
    {
        const double expected =
            NN::updated_statistic(running[i], batch[i], K_MOMENTUM, K_SOURCE_SCALE);
        EXPECT_NEAR(blended[i], expected, K_BLEND_TOLERANCE) << "element " << i;
    }
}

/** @brief Tests that a recorder holding no slot records no task and runs cleanly. */
TEST_F(Integration_BufferUpdateRecorder, NoSlotRecordsNothing)
{
    Train::BufferUpdateRecorder recorder(std::vector<Train::BufferUpdateSlot>{});

    std::size_t recorded = 1;
    const Status run = record_and_run(*engine, [&]() -> Status
    {
        SA_TRY_ASSIGN(const std::size_t count,
                      recorder.record_into_step(engine->recorder()));
        recorded = count;
        return Status();
    });
    SA_EXPECT_OK(run);
    EXPECT_EQ(recorder.slot_count(), 0u);
    EXPECT_EQ(recorded, 0u);
}

/** @brief Tests that binding refuses an update whose target the state does not hold. */
TEST_F(Integration_BufferUpdateRecorder, BindRefusesATargetTheStateLacks)
{
    std::vector<NN::StateEntry> entries;
    entries.push_back(make_entry("bn.running_mean", 1, DType::Float64));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, std::move(entries)));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(*engine, {1.0, 2.0}, DType::Float64));
    SA_EXPECT_OK(pool.bind(K_SOURCE_VALUE, std::move(source)));

    NN::BufferUpdateList updates;
    updates.add(NN::BufferUpdate{Graph::StateID{K_TARGET_STATE}, K_SOURCE_VALUE, K_MOMENTUM,
                                 K_SOURCE_SCALE});

    SA_EXPECT_FAILURE(Train::bind_buffer_updates(updates, state, pool),
                      StatusCode::Incompatible, "no entry");
}

/** @brief Tests that binding refuses an update whose target and source differ in dtype. */
TEST_F(Integration_BufferUpdateRecorder, BindRefusesADTypeMismatch)
{
    std::vector<NN::StateEntry> entries;
    entries.push_back(make_entry("bn.running_mean", K_TARGET_STATE, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, std::move(entries)));

    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor source,
                            vector_of(*engine, {1.0, 2.0}, DType::Float64));
    SA_EXPECT_OK(pool.bind(K_SOURCE_VALUE, std::move(source)));

    NN::BufferUpdateList updates;
    updates.add(NN::BufferUpdate{Graph::StateID{K_TARGET_STATE}, K_SOURCE_VALUE, K_MOMENTUM,
                                 K_SOURCE_SCALE});

    SA_EXPECT_FAILURE(Train::bind_buffer_updates(updates, state, pool),
                      StatusCode::Incompatible, dtype_name(DType::Float32),
                      dtype_name(DType::Float64));
}
