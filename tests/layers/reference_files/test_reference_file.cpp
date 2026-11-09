/**************************************************************************/
/* test_reference_file.cpp                                                */
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
 * @file test_reference_file.cpp
 * @brief Tests the absent-file path of ReferenceFile, ReferenceTensor and the comparisons.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;
using SaTest::ReferenceFile;
using SaTest::ReferenceTensor;

namespace
{
    /**
     * @brief Holds the tolerance near_reference() is called with.
     *
     * It is an input of the function under test, not a claim about an accuracy: the data of the
     * cases below sit half of it, five times it and ten times it from the reference.
     */
    constexpr float K_TOLERANCE_UNDER_TEST = 0.1f;

    /** @brief Holds the tolerance that demands equal values. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Returns a found float32 tensor of [@p values.size()] holding @p values. */
    ReferenceTensor float_tensor(const std::vector<float>& values)
    {
        ReferenceTensor tensor;
        tensor.found = true;
        tensor.dtype = DType::Float32;
        tensor.shape = Shape{static_cast<int64_t>(values.size())};
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Views @p values as a span of floats. */
    SushiRuntime::span<const float> view(const std::vector<float>& values)
    {
        return SushiRuntime::span<const float>(values.data(), values.size());
    }
} // namespace

/** @brief Tests that a file the opener does not find is reported with its path, not read. */
TEST(Small_ReferenceFile, AnAbsentFileIsReportedWithItsPath)
{
    ReferenceFile file("fixtures", "no_such_reference.sacp",
                       [](const std::string&) { return std::unique_ptr<std::istream>(); });

    EXPECT_FALSE(file.is_present());
    EXPECT_NE(file.reason().find("missing"), std::string::npos) << file.reason();
    EXPECT_NE(file.reason().find("fixtures/no_such_reference.sacp"), std::string::npos)
        << file.reason();

    const ReferenceTensor tensor = file.read("model.0.weight");
    EXPECT_FALSE(tensor.found);
    EXPECT_EQ(tensor.reason, file.reason());
}

/** @brief Tests that a tensor yields its elements exactly as its own type. */
TEST(Small_ReferenceFile, ATensorIsReadAsItsOwnType)
{
    const std::vector<float> values{1.5f, -2.0f, 0.25f};
    const ReferenceTensor tensor = float_tensor(values);

    EXPECT_TRUE(SaTest::near_reference(view(tensor.floats()), view(values), K_EXACT));
}

/** @brief Tests that a tensor of one type stops the process when it is read as another. */
TEST(Medium_ReferenceFileDeathTest, ATensorReadAsAnotherTypeStopsTheProcess)
{
    const ReferenceTensor tensor = float_tensor({1.5f, -2.0f, 0.25f});

    SA_EXPECT_FATAL((void)tensor.int32s(), "float32", "int32");
}

/** @brief Tests that a tensor that was not found stops the process when it is read. */
TEST(Medium_ReferenceFileDeathTest, ATensorThatWasNotFoundStopsTheProcessWhenRead)
{
    ReferenceTensor absent;
    absent.reason = "the tensor is not in the file";

    SA_EXPECT_FATAL((void)absent.floats(), "not found", "the tensor is not in the file");
}

/** @brief Tests that near_reference() passes within the tolerance and names the worst miss. */
TEST(Small_ReferenceFile, NearReferenceNamesTheFirstAndTheWorstMiss)
{
    // Each close value is half the tolerance off; the far ones are five times and ten times it.
    const std::vector<float> reference{1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> close{1.05f, 1.95f, 3.0f, 4.05f};
    const std::vector<float> far{1.0f, 2.5f, 3.0f, 5.0f};

    EXPECT_TRUE(SaTest::near_reference(view(close), view(reference), K_TOLERANCE_UNDER_TEST));
    EXPECT_TRUE(SaTest::near_reference(view(reference), view(reference), K_EXACT));

    const ::testing::AssertionResult missed =
        SaTest::near_reference(view(far), view(reference), K_TOLERANCE_UNDER_TEST);
    EXPECT_FALSE(missed);
    EXPECT_NE(std::string(missed.message()).find("2 of 4"), std::string::npos)
        << missed.message();
    EXPECT_NE(std::string(missed.message()).find("worst at 3"), std::string::npos)
        << missed.message();

    const std::vector<float> shorter{1.0f, 2.0f, 3.0f};
    EXPECT_FALSE(SaTest::near_reference(view(close), view(shorter), K_TOLERANCE_UNDER_TEST));
}

/** @brief Tests that equal_reference() compares across integer types and names the first miss. */
TEST(Small_ReferenceFile, EqualReferenceComparesAcrossIntegerTypes)
{
    const std::vector<std::int32_t> actual{3, 1, 4, 1};
    const std::vector<std::int64_t> same{3, 1, 4, 1};
    const std::vector<std::int64_t> other{3, 1, 5, 9};

    EXPECT_TRUE(SaTest::equal_reference(actual, same));

    const ::testing::AssertionResult missed = SaTest::equal_reference(actual, other);
    EXPECT_FALSE(missed);
    EXPECT_NE(std::string(missed.message()).find("first at 2"), std::string::npos)
        << missed.message();

    EXPECT_FALSE(SaTest::equal_reference(actual, std::vector<std::int64_t>{3, 1}));
}
