/**************************************************************************/
/* test_tensor_source.cpp                                                 */
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
 * @file test_tensor_source.cpp
 * @brief Tests that the in-memory tensor source describes each tensor it is asked for and
 *        keeps the requests.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_source.hpp>

using namespace SushiAI;

/** @brief Tests that a tensor holds the extents and the type it was asked for. */
TEST(Small_InMemoryTensorSource, ATensorHoldsTheExtentsAndTheTypeItWasAskedFor)
{
    Graph::InMemoryTensorSource source;

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            source.create(Shape{2, 5}, DType::Float64));

    EXPECT_EQ(tensor.rank, 2);
    EXPECT_EQ(tensor.shape[0], 2);
    EXPECT_EQ(tensor.shape[1], 5);
    EXPECT_EQ(tensor.dtype, DType::Float64);
    EXPECT_EQ(tensor.num_elements, 10);
}

/** @brief Tests that a tensor holds no storage to read. */
TEST(Small_InMemoryTensorSource, ATensorHoldsNoStorage)
{
    Graph::InMemoryTensorSource source;

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor tensor,
                            source.create(Shape{4}, DType::Float32));

    EXPECT_FALSE(tensor.storage);
}

/** @brief Tests that the source keeps each request, in the order it came. */
TEST(Small_InMemoryTensorSource, EveryRequestIsKeptInOrder)
{
    Graph::InMemoryTensorSource source;

    SA_EXPECT_OK(source.create(Shape{3}, DType::Float32));
    SA_EXPECT_OK(source.create(Shape{2, 2}, DType::Int32));

    ASSERT_EQ(source.requests().size(), 2u);
    EXPECT_EQ(source.requests()[0].shape, Shape{3});
    EXPECT_EQ(source.requests()[0].dtype, DType::Float32);
    EXPECT_EQ(source.requests()[1].shape, (Shape{2, 2}));
    EXPECT_EQ(source.requests()[1].dtype, DType::Int32);
}
