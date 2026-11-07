/**************************************************************************/
/* test_batch_allocator.cpp                                               */
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


/**
 * @file test_batch_allocator.cpp
 * @brief Tests what the two batch allocators give: the layout the engine allocates and the
 *        layout the in-memory allocator describes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/data/batch_allocator.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the engine allocator draws from. */
    class Medium_EngineBatchAllocator : public SaTest::EngineTest
    {

    };

    /** @brief The extents of every tensor below. */
    constexpr int64_t K_DIMS[2] = {3, 4};
} // namespace

/** @brief Tests that the in-memory allocator describes the extents and the element type. */
TEST(Small_InMemoryBatchAllocator, ATensorHoldsTheExtentsAndTheTypeItWasAskedFor)
{
    Data::InMemoryBatchAllocator allocator;

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            allocator.allocate(SushiRuntime::span<const int64_t>(K_DIMS, 2),
                                               DType::Int32));

    EXPECT_EQ(tensor.rank, 2);
    EXPECT_EQ(tensor.shape[0], 3);
    EXPECT_EQ(tensor.shape[1], 4);
    EXPECT_EQ(tensor.dtype, DType::Int32);
}

/** @brief Tests that the in-memory allocator gives a tensor no storage to read. */
TEST(Small_InMemoryBatchAllocator, ATensorHoldsNoStorage)
{
    Data::InMemoryBatchAllocator allocator;

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            allocator.allocate(SushiRuntime::span<const int64_t>(K_DIMS, 2),
                                               DType::Float32));

    EXPECT_FALSE(tensor.storage);
}

/** @brief Tests that the in-memory allocator refuses an extent that is not positive. */
TEST(Small_InMemoryBatchAllocator, AnExtentThatIsNotPositiveIsRefused)
{
    Data::InMemoryBatchAllocator allocator;
    const int64_t dims[2] = {3, 0};

    SA_EXPECT_FAILURE(allocator.allocate(SushiRuntime::span<const int64_t>(dims, 2),
                                         DType::Float32),
                      StatusCode::InvalidDescription, "positive");
}

/** @brief Tests that the engine allocator gives the layout the in-memory one describes. */
TEST_F(Medium_EngineBatchAllocator, ATensorHoldsTheExtentsTheInMemoryAllocatorDescribes)
{
    Data::EngineBatchAllocator on_the_engine(*engine);
    Data::InMemoryBatchAllocator in_memory;
    const SushiRuntime::span<const int64_t> dims(K_DIMS, 2);

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor real,
                            on_the_engine.allocate(dims, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor described,
                            in_memory.allocate(dims, DType::Float32));

    EXPECT_TRUE(real.storage);
    EXPECT_EQ(real.rank, described.rank);
    EXPECT_EQ(real.shape[0], described.shape[0]);
    EXPECT_EQ(real.shape[1], described.shape[1]);
    EXPECT_EQ(real.dtype, described.dtype);
}
