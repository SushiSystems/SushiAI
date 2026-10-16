/**************************************************************************/
/* test_feed.cpp                                                          */
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
 * @file test_feed.cpp
 * @brief Tests check_feed, which compares a batch tensor with the leaf a step copies it into.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/feed.hpp>

using namespace SushiAI;

namespace
{
    /**
     * @brief Returns a tensor header of @p extents and @p dtype that owns no storage.
     *
     * check_feed reads the rank, the extents and the element type only, so no device is needed.
     */
    SushiBLAS::Tensor header_of(std::initializer_list<std::int64_t> extents, DType dtype)
    {
        SushiBLAS::Tensor tensor{};
        tensor.rank = static_cast<std::int32_t>(extents.size());
        tensor.dtype = dtype;
        std::size_t axis = 0;
        for (const std::int64_t extent : extents)
            tensor.shape[axis++] = extent;
        return tensor;
    }
} // namespace

/** @brief Tests that tensors of one rank, one set of extents and one element type pass. */
TEST(Small_Feed, MatchingTensorsAreAccepted)
{
    SA_EXPECT_OK(Train::check_feed(header_of({4, 6}, DType::Float32),
                                   header_of({4, 6}, DType::Float32), "inputs"));
}

/** @brief Tests that other extents or another rank are refused with both shapes named. */
TEST(Small_Feed, AnotherShapeIsIncompatibleAndBothShapesAreNamed)
{
    SA_EXPECT_FAILURE(Train::check_feed(header_of({4, 5}, DType::Float32),
                                        header_of({4, 6}, DType::Float32), "inputs"),
                      StatusCode::Incompatible, "inputs", "[4, 5]", "[4, 6]");
    SA_EXPECT_FAILURE(Train::check_feed(header_of({24}, DType::Float32),
                                        header_of({4, 6}, DType::Float32), "inputs"),
                      StatusCode::Incompatible, "[24]", "[4, 6]");
}

/** @brief Tests that equal shapes of two element types are refused with both types named. */
TEST(Small_Feed, AnotherElementTypeIsIncompatibleAndBothTypesAreNamed)
{
    SA_EXPECT_FAILURE(Train::check_feed(header_of({4, 3}, DType::Int32),
                                        header_of({4, 3}, DType::Float32), "targets"),
                      StatusCode::Incompatible, "targets", "int32", "float32");
}

/** @brief Tests that an extent no Shape can hold is reported, not thrown. */
TEST(Small_Feed, AnExtentNoShapeHoldsIsAnInvalidDescription)
{
    SA_EXPECT_FAILURE(Train::check_feed(header_of({4, 0}, DType::Float32),
                                        header_of({4, 6}, DType::Float32), "inputs"),
                      StatusCode::InvalidDescription, "positive");
}
