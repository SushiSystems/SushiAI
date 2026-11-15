/**************************************************************************/
/* test_ultralytics_tensors.cpp                                           */
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
 * @file test_ultralytics_tensors.cpp
 * @brief Tests the weight records of Ultralytics tensors: their layouts and the refused groupings.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/importers/ultralytics_checkpoint.hpp>
#include <SushiAI/vision/importers/ultralytics_tensors.hpp>

using namespace SushiAI;
using Vision::Importers::UltralyticsTensor;

namespace
{
    /** @brief Returns a tensor named "w" of @p shape and @p groups holding 0, 1, 2 and so on. */
    UltralyticsTensor counting_tensor(const Shape& shape, std::int64_t groups)
    {
        UltralyticsTensor tensor;
        tensor.name = "w";
        tensor.groups = groups;
        tensor.shape = shape;
        tensor.values.resize(static_cast<std::size_t>(shape.num_elements()));
        for (std::size_t i = 0; i < tensor.values.size(); ++i)
            tensor.values[i] = static_cast<float>(i);
        return tensor;
    }

    /** @brief Returns a one-element bias tensor named @p name. */
    UltralyticsTensor bias_tensor(const std::string& name)
    {
        UltralyticsTensor tensor = counting_tensor(Shape({1}), 1);
        tensor.name = name;
        return tensor;
    }

    /** @brief Returns the float32 elements of the payload of @p tensor. */
    std::vector<float> floats_of(const IO::ImportedTensor& tensor)
    {
        std::vector<float> out(tensor.payload.size() / sizeof(float));
        std::memcpy(out.data(), tensor.payload.data(), tensor.payload.size());
        return out;
    }

    /** @brief Returns list_ultralytics_tensors of the one tensor @p tensor. */
    Result<std::vector<IO::ImportedTensor>> list_one(const UltralyticsTensor& tensor)
    {
        return Vision::Importers::list_ultralytics_tensors(
            SushiRuntime::span<const UltralyticsTensor>(&tensor, 1),
            Vision::Importers::TensorRules{});
    }
} // namespace

/** @brief Tests that a [O, I, kh, kw] weight is laid out [O, kh, kw, I]. */
TEST(Small_UltralyticsTensors, ADenseWeightIsLaidOutOutputTapsInput)
{
    // Element (o, i, t) of a [2, 3, 1, 2] weight holds 6 o + 2 i + t; the record runs o, t, i.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> records,
                            list_one(counting_tensor(Shape{2, 3, 1, 2}, 1)));

    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].record.shape, (Shape{2, 1, 2, 3}));
    const std::vector<float> expected = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11};
    EXPECT_EQ(floats_of(records[0]), expected);
}

/** @brief Tests that a depthwise [C, 1, kh, kw] weight is laid out [kh, kw, C]. */
TEST(Small_UltralyticsTensors, ADepthwiseWeightIsLaidOutTapsChannel)
{
    // Element (c, t) of a [3, 1, 1, 2] weight of 3 groups holds 2 c + t; the record runs t, c.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> records,
                            list_one(counting_tensor(Shape{3, 1, 1, 2}, 3)));

    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].record.shape, (Shape{1, 2, 3}));
    const std::vector<float> expected = {0, 2, 4, 1, 3, 5};
    EXPECT_EQ(floats_of(records[0]), expected);
}

/** @brief Tests that a tensor of any other rank keeps its layout and its values. */
TEST(Small_UltralyticsTensors, ABiasKeepsItsLayout)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> records,
                            list_one(counting_tensor(Shape{3}, 1)));

    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].record.name, "w");
    EXPECT_EQ(records[0].record.shape, (Shape{3}));
    const std::vector<float> expected = {0, 1, 2};
    EXPECT_EQ(floats_of(records[0]), expected);
}

/** @brief Tests that a grouped convolution that is not depthwise is refused by name. */
TEST(Small_UltralyticsTensors, AGroupedWeightThatIsNotDepthwiseIsRefused)
{
    SA_EXPECT_FAILURE(list_one(counting_tensor(Shape{4, 2, 1, 1}, 2)), StatusCode::Unsupported,
                      "'w'", "2 groups");
}

/** @brief Tests that a tensor under a renamed prefix takes the new prefix and keeps the rest. */
TEST(Small_UltralyticsTensors, ARenamedPrefixIsReplacedAndTheRestKept)
{
    const UltralyticsTensor tensor = bias_tensor("24.m.1.bias");
    Vision::Importers::TensorRules rules;
    rules.renames.emplace_back("24.m.1.", "27.");

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> records,
                            Vision::Importers::list_ultralytics_tensors(
                                SushiRuntime::span<const UltralyticsTensor>(&tensor, 1), rules));

    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].record.name, "27.bias");
}

/** @brief Tests that a tensor under a dropped prefix gives no record. */
TEST(Small_UltralyticsTensors, ADroppedPrefixGivesNoRecord)
{
    const UltralyticsTensor tensor = bias_tensor("22.dfl.conv.weight");
    Vision::Importers::TensorRules rules;
    rules.dropped.emplace_back("22.dfl.");

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> records,
                            Vision::Importers::list_ultralytics_tensors(
                                SushiRuntime::span<const UltralyticsTensor>(&tensor, 1), rules));

    EXPECT_TRUE(records.empty());
}
