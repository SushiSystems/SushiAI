/**************************************************************************/
/* test_tensor_bytes.cpp                                                  */
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
 * @file test_tensor_bytes.cpp
 * @brief Tests that TensorBytes either views the bytes of a file or owns a copy of them.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/interchange/tensor_bytes.hpp>

using namespace SushiAI;
using Interchange::TensorBytes;

namespace
{
    /** @brief Returns the bytes 1, 2, 3, 4, 5 that the cases view or own. */
    std::vector<std::byte> five_bytes()
    {
        return {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
    }
} // namespace

/** @brief Tests that a default tensor holds no byte. */
TEST(Small_TensorBytes, ADefaultTensorHoldsNoByte)
{
    const TensorBytes tensor;

    EXPECT_EQ(tensor.size(), 0u);
    EXPECT_TRUE(tensor.bytes().empty());
}

/** @brief Tests that a view reports the file's bytes by address and its offset. */
TEST(Small_TensorBytes, ViewingKeepsTheCallersBytesWithoutACopy)
{
    const std::vector<std::byte> file = five_bytes();
    const SushiRuntime::span<const std::byte> range(file.data() + 1, 3);

    const TensorBytes tensor = TensorBytes::viewing(range, 1);

    EXPECT_TRUE(tensor.is_view());
    EXPECT_EQ(tensor.bytes().data(), file.data() + 1);
    EXPECT_EQ(tensor.size(), 3u);
    EXPECT_EQ(tensor.offset(), 1u);
}

/** @brief Tests that an owned tensor holds the bytes it was given and no file offset. */
TEST(Small_TensorBytes, OwningHoldsTheGivenBytesAndHasNoOffset)
{
    const TensorBytes tensor = TensorBytes::owning(five_bytes());

    EXPECT_FALSE(tensor.is_view());
    EXPECT_EQ(tensor.size(), 5u);
    EXPECT_EQ(tensor.offset(), 0u);
    EXPECT_EQ(std::vector<std::byte>(tensor.bytes().begin(), tensor.bytes().end()), five_bytes());
}

/** @brief Tests that an owned tensor keeps its bytes when the source vector it took is gone. */
TEST(Small_TensorBytes, OwningTakesTheBytesOutOfTheCallersVector)
{
    std::vector<std::byte> source = five_bytes();

    const TensorBytes tensor = TensorBytes::owning(std::move(source));
    source.clear();

    EXPECT_EQ(tensor.size(), 5u);
    EXPECT_EQ(tensor.bytes()[4], std::byte{5});
}
