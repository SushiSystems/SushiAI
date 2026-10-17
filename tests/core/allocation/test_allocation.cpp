/**************************************************************************/
/* test_allocation.cpp                                                    */
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
 * @file test_allocation.cpp
 * @brief Tests what allocate_tensor, allocate_like and allocate_arena give back or refuse.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::DType;
using SushiAI::Shape;
using SushiAI::StatusCode;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine every allocation case allocates on. */
    class Medium_Allocation : public SaTest::EngineTest
    {

    };

    /** @brief Holds the byte count the arena case asks for; not a multiple of any alignment. */
    constexpr std::size_t K_ARENA_BYTES = 1000;
} // namespace

/** @brief Tests that a tensor has the extents and type it was asked for, and its own storage. */
TEST_F(Medium_Allocation, ATensorHasTheExtentsAndTypeItWasAskedFor)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            SushiAI::allocate_tensor(*engine, Shape{2, 3, 4}, DType::Float64));

    EXPECT_EQ(tensor.rank, 3);
    EXPECT_EQ(tensor.shape[0], 2);
    EXPECT_EQ(tensor.shape[1], 3);
    EXPECT_EQ(tensor.shape[2], 4);
    EXPECT_EQ(tensor.num_elements, 24);
    EXPECT_EQ(tensor.dtype, DType::Float64);
    ASSERT_TRUE(tensor.storage);
    EXPECT_GE(tensor.storage->size_bytes(), 24u * SushiAI::dtype_size(DType::Float64));
}

/** @brief Tests that an extent that is not positive is refused as InvalidDescription. */
TEST_F(Medium_Allocation, AnExtentThatIsNotPositiveIsRefused)
{
    const std::int64_t dims[2] = {4, 0};

    SA_EXPECT_FAILURE(
        SushiAI::allocate_tensor(*engine, SushiRuntime::span<const std::int64_t>(dims, 2)),
        StatusCode::InvalidDescription);
}

/** @brief Tests that allocate_like copies the description and shares no storage. */
TEST_F(Medium_Allocation, ALikeTensorCopiesTheDescriptionAndSharesNothing)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor original,
                            SushiAI::allocate_tensor(*engine, Shape{5, 2}, DType::Int32));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor copy,
                            SushiAI::allocate_like(*engine, original));

    EXPECT_EQ(copy.rank, original.rank);
    EXPECT_EQ(copy.shape, original.shape);
    EXPECT_EQ(copy.dtype, original.dtype);
    EXPECT_EQ(copy.layout, original.layout);
    ASSERT_TRUE(copy.storage);
    EXPECT_NE(copy.storage.get(), original.storage.get());
}

/** @brief Tests that an arena holds at least the bytes it was asked for. */
TEST_F(Medium_Allocation, AnArenaHoldsAtLeastTheBytesAskedFor)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                            SushiAI::allocate_arena(*engine, K_ARENA_BYTES));

    ASSERT_TRUE(arena);
    EXPECT_GE(arena->size_bytes(), K_ARENA_BYTES);
}
