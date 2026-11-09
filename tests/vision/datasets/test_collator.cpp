/**************************************************************************/
/* test_collator.cpp                                                      */
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
 * @file test_collator.cpp
 * @brief Tests the collator interface through a real collator and, for its destructor, a counting fake task.
 * @author Mustafa Garip
 */

#include <memory>
#include <utility>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;

namespace
{
    /** @brief Collates a class per sample, refusing a sample without one, and counts its end. */
    class ClassCollator final : public Collator
    {
        public:
            /** @brief Keeps the counter the destructor raises. */
            explicit ClassCollator(int& destroyed)
                : destroyed_(destroyed),
                  targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{},
                                                   DType::Int32, 4}}
            {

            }

            /** @brief Raises the counter it was given. */
            ~ClassCollator() override
            {
                ++destroyed_;
            }

            /** @brief Forbids a copy, which would count one end twice. */
            ClassCollator(const ClassCollator&) = delete;

            /** @brief Forbids copy assignment. */
            ClassCollator& operator=(const ClassCollator&) = delete;

            /** @brief Forbids a move. */
            ClassCollator(ClassCollator&&) = delete;

            /** @brief Forbids move assignment. */
            ClassCollator& operator=(ClassCollator&&) = delete;

            /** @brief Returns "fake_classes". */
            [[nodiscard]] std::string_view task() const noexcept override
            {
                return "fake_classes";
            }

            /** @brief Returns one role of int32 labels over four classes. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override
            {
                return targets_;
            }

            /** @brief Refuses the first sample without a class, opening with its file. */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override
            {
                static_cast<void>(batch);
                for (const FittedSample& sample : samples)
                {
                    SA_FAIL_IF(!sample.class_index.has_value(), StatusCode::InvalidDescription,
                               "{}: the sample has no class", list.files[sample.file].string());
                }

                return Status();
            }

        private:
            /// @brief Holds the counter the destructor raises.
            int& destroyed_;

            /// @brief Holds the one role collate() writes.
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Provides a host engine, a two-file list and two gray samples. */
    class Small_DatasetsCollator : public Support::CollatorBench
    {
    };
} // namespace

/** @brief Tests that the task and the roles of a collator are read through the interface. */
TEST(Small_DatasetsCollatorBase, TheTaskAndTheRolesAreReadThroughTheBase)
{
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BinaryCollator> made, BinaryCollator::make(1));
    const std::unique_ptr<const Collator> collator = std::move(made);

    EXPECT_EQ(collator->task(), K_TASK_BINARY);
    ASSERT_EQ(collator->targets().size(), 1u);
    EXPECT_EQ(collator->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(collator->targets()[0].dtype, DType::Float32);
}

/** @brief Tests that a sample the task cannot use is refused with its file first. */
TEST_F(Small_DatasetsCollator, ASampleTheTaskCannotUseIsRefusedNamingItsFile)
{
    samples[0].class_index = 1;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BinaryCollator> made, BinaryCollator::make(1));
    const std::unique_ptr<const Collator> collator = std::move(made);
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_FAILURE(collator->collate(list, span(), batch), StatusCode::InvalidDescription,
                      "second.png", "no class index");
}

/** @brief Tests that a sample the task can use is collated through the interface. */
TEST_F(Small_DatasetsCollator, ASampleTheTaskCanUseIsCollated)
{
    samples[0].class_index = 1;
    samples[1].class_index = 0;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BinaryCollator> made, BinaryCollator::make(1));
    const std::unique_ptr<const Collator> collator = std::move(made);
    Data::Batch batch = allocate_images(collator->targets());

    SA_EXPECT_OK(collator->collate(list, span(), batch));
}

/** @brief Tests that destroying through the interface runs the destructor of the task. */
TEST(Small_DatasetsCollatorBase, DestroyingThroughTheBaseRunsTheDerivedDestructor)
{
    int destroyed = 0;
    std::unique_ptr<const Collator> collator = std::make_unique<ClassCollator>(destroyed);

    collator.reset();

    EXPECT_EQ(destroyed, 1);
}
