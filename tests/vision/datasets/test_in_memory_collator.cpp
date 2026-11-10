/**************************************************************************/
/* test_in_memory_collator.cpp                                            */
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
 * @file test_in_memory_collator.cpp
 * @brief Tests that InMemoryCollator keeps the promise of the Collator interface.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/in_memory_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Returns the one role of every case: an int32 label per sample over 4 classes. */
    std::vector<Data::TargetDescription> labels()
    {
        return {Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                        4}};
    }
} // namespace

/** @brief Tests that the task and the roles are the ones the collator was built with. */
TEST(Small_InMemoryCollator, TheTaskAndTheRolesAreTheOnesItWasBuiltWith)
{
    const InMemoryCollator concrete("fake_classes", labels());
    const Collator& collator = concrete;

    EXPECT_EQ(collator.task(), "fake_classes");
    ASSERT_EQ(collator.targets().size(), 1u);
    EXPECT_EQ(collator.targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator.targets()[0].classes, 4);
}

/** @brief Tests that collate succeeds and writes no element of the batch. */
TEST(Small_InMemoryCollator, CollateSucceedsAndWritesNothing)
{
    const InMemoryCollator collator("fake_classes", labels());
    const SampleList list;
    Data::Batch batch;

    SA_EXPECT_OK(collator.collate(list, SushiRuntime::span<const FittedSample>(), batch));

    EXPECT_TRUE(batch.targets.empty());
}

/** @brief Tests that a scripted failure is what every collate returns. */
TEST(Small_InMemoryCollator, AScriptedFailureIsWhatCollateReturns)
{
    const InMemoryCollator collator(
        "fake_classes", labels(),
        Status::from_message(StatusCode::InvalidDescription, []()
        {
            return std::string("'second.png': the sample has no class");
        }));
    const SampleList list;
    Data::Batch batch;

    SA_EXPECT_FAILURE(collator.collate(list, SushiRuntime::span<const FittedSample>(), batch),
                      StatusCode::InvalidDescription, "second.png", "no class");
}

/** @brief Tests that destroying through the interface runs the destructor of the adapter. */
TEST(Small_InMemoryCollator, DestroyingThroughTheBaseRunsTheDerivedDestructor)
{
    std::size_t ended = 0;
    std::unique_ptr<const Collator> collator =
        std::make_unique<InMemoryCollator>("fake_classes", labels(), Status(), &ended);

    collator.reset();

    EXPECT_EQ(ended, 1u);
}
