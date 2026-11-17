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
 * @brief Tests the binding of buffer updates to the tensors they touch.
 * @author Mustafa Garip
 */

#include <cstdint>
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
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The weight the blend gives the buffer's current content. */
    constexpr double K_MOMENTUM = 0.1;

    /** @brief The factor the batch value is scaled by before it is blended in. */
    constexpr double K_SOURCE_SCALE = 2.0;

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

    /** @brief Binds buffer updates on the host backend. */
    class Small_BufferUpdateRecorder : public ::testing::Test
    {
        protected:
            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

} // namespace

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
