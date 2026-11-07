/**************************************************************************/
/* test_exchange_targets.cpp                                              */
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
 * @file test_exchange_targets.cpp
 * @brief Tests the checks of the tensors a gradient exchange sums into, on shapes alone.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/exchange_targets.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using SushiBLAS::Core::DataType;

    /** @brief Holds the elements of each tensor of a case. */
    constexpr std::int64_t K_ELEMENTS = 3;

    /** @brief Returns a tensor of @p elements elements of @p dtype that holds no storage. */
    SushiBLAS::Tensor shape_of(DataType dtype, std::int64_t elements = K_ELEMENTS)
    {
        const std::int64_t dims[1] = {elements};
        return SushiBLAS::Tensor::make_shape_only(SushiRuntime::span<const std::int64_t>(dims, 1),
                                                  dtype);
    }

    /** @brief Returns the layout of @p tensors. */
    SushiBLAS::PackLayout layout_of(const std::vector<SushiBLAS::Tensor>& tensors)
    {
        return SushiBLAS::PackLayout::of(
            SushiRuntime::span<const SushiBLAS::Tensor>(tensors.data(), tensors.size()));
    }
} // namespace

/** @brief Tests that targets of the layout's types and shapes are accepted. */
TEST(Small_ExchangeTargets, TargetsOfTheLayoutAreAccepted)
{
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32),
                                                    shape_of(DataType::Int64)};

    SA_EXPECT_OK(check_exchange_targets(layout_of(targets), targets));
}

/** @brief Tests that a target count that differs from the layout's entries is refused. */
TEST(Small_ExchangeTargets, ATargetCountThatDiffersFromTheLayoutIsRefused)
{
    const std::vector<SushiBLAS::Tensor> entries = {shape_of(DataType::Float32),
                                                    shape_of(DataType::Float32)};
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32)};

    SA_EXPECT_FAILURE(check_exchange_targets(layout_of(entries), targets),
                      StatusCode::InvalidDescription, "holds 2 tensors and 1 targets");
}

/** @brief Tests that a target of another type than its entry is refused, naming it. */
TEST(Small_ExchangeTargets, ATargetOfAnotherTypeIsRefused)
{
    const std::vector<SushiBLAS::Tensor> entries = {shape_of(DataType::Float32),
                                                    shape_of(DataType::Float32)};
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32),
                                                    shape_of(DataType::Float64)};

    SA_EXPECT_FAILURE(check_exchange_targets(layout_of(entries), targets),
                      StatusCode::InvalidDescription, "Target 1 differs");
}

/** @brief Tests that a target of another shape than its entry is refused, naming it. */
TEST(Small_ExchangeTargets, ATargetOfAnotherShapeIsRefused)
{
    const std::vector<SushiBLAS::Tensor> entries = {shape_of(DataType::Float32)};
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32, K_ELEMENTS + 1)};

    SA_EXPECT_FAILURE(check_exchange_targets(layout_of(entries), targets),
                      StatusCode::InvalidDescription, "Target 0 differs");
}

/** @brief Tests that targets of types ordered_sum() sums pass the type check. */
TEST(Small_ExchangeTargets, TheTypesOrderedSumSumsPassTheTypeCheck)
{
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32),
                                                    shape_of(DataType::Int32)};

    SA_EXPECT_OK(check_exchange_dtypes(targets));
}

/** @brief Tests that a uint32 target is unsupported, naming it. */
TEST(Small_ExchangeTargets, AUInt32TargetIsUnsupported)
{
    const std::vector<SushiBLAS::Tensor> targets = {shape_of(DataType::Float32),
                                                    shape_of(DataType::UInt32)};

    SA_EXPECT_FAILURE(check_exchange_dtypes(targets), StatusCode::Unsupported,
                      "Target 1 is uint32");
}
