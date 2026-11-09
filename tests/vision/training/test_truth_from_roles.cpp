/**************************************************************************/
/* test_truth_from_roles.cpp                                              */
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
 * @file test_truth_from_roles.cpp
 * @brief Tests the roles a detection step reads and their copy into a ground truth.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/input.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

using namespace SushiAI;
using Vision::Detect::GroundTruthBatch;
using Vision::Training::describe_truth_roles;
using Vision::Training::K_TRUTH_BOX_VALUES;
using Vision::Training::truth_from_roles;

namespace
{
    /** @brief Holds the images of every batch of these tests, N. */
    constexpr std::int64_t K_IMAGES = 2;

    /** @brief Holds the object slots of one image, M. */
    constexpr std::int64_t K_SLOTS = 3;

    /** @brief Holds the classes the classes role is described over. */
    constexpr std::int64_t K_CLASSES = 5;

    /** @brief Holds the corners of the six slots; a padding slot holds zeros. */
    const std::vector<float> K_CORNERS = {1.5f,  2.5f, 30.25f, 40.75f, 0.0f, 0.0f,  0.0f,  0.0f,
                                          8.0f,  9.0f, 16.0f,  17.0f,  0.0f, 0.0f,  0.0f,  0.0f,
                                          0.0f,  0.0f, 0.0f,   0.0f,   3.0f, 4.0f,  63.5f, 64.0f};

    /** @brief Holds the class of each slot; a padding slot holds 0. */
    const std::vector<std::int32_t> K_INDICES = {4, 0, 2, 0, 0, 1};

    /** @brief Holds 1 for each slot in use and 0 for each padding slot. */
    const std::vector<float> K_FLAGS = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    /** @brief Provides an engine and batches whose roles hold the values above. */
    class Medium_TruthFromRoles : public Test::EngineTest
    {
        protected:
            /** @brief Returns the three roles of K_SLOTS slots over K_CLASSES classes. */
            static std::vector<Data::TargetDescription> roles()
            {
                Result<std::vector<Data::TargetDescription>> described =
                    describe_truth_roles(K_SLOTS, K_CLASSES);
                EXPECT_TRUE(described.status().is_ok());
                return std::move(described).value();
            }

            /** @brief Allocates a batch of K_IMAGES one-value inputs over @p targets. */
            Data::Batch allocate(const std::vector<Data::TargetDescription>& targets)
            {
                Result<Data::Batch> batch =
                    Data::allocate_batch(
                        *engine, K_IMAGES, Data::InputDescription{Shape{1}, DType::Float32}, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Allocates the three roles and writes the values above into them. */
            Data::Batch filled_batch()
            {
                Data::Batch batch = allocate(roles());
                std::copy(K_CORNERS.begin(), K_CORNERS.end(),
                          batch.targets[0].values.data_as<float>());
                std::copy(K_INDICES.begin(), K_INDICES.end(),
                          batch.targets[1].values.data_as<std::int32_t>());
                std::copy(K_FLAGS.begin(), K_FLAGS.end(),
                          batch.targets[2].values.data_as<float>());
                return batch;
            }

            /** @brief Returns the truth read from the filled batch. */
            GroundTruthBatch truth_of_filled_batch()
            {
                const Data::Batch batch = filled_batch();
                GroundTruthBatch truth;
                SA_EXPECT_OK(truth_from_roles(batch, truth));
                return truth;
            }
    };

    /** @brief Provides the batch that lacks the role at one index. */
    class Medium_TruthFromRolesMissing : public Medium_TruthFromRoles,
                                         public ::testing::WithParamInterface<std::size_t>
    {

    };

    /** @brief Names a change to the roles of a batch and what the refusal names. */
    struct BadRole
    {
        const char* name;                                        /**< Names the row. */
        void (*change)(std::vector<Data::TargetDescription>&);   /**< Breaks one role. */
        const char* needles[3];                                  /**< What the message names. */
    };

    /** @brief Provides the roles of another element type, rank, width or slot count. */
    class Medium_TruthFromRolesBadRole : public Medium_TruthFromRoles,
                                         public ::testing::WithParamInterface<BadRole>
    {

    };

    /** @brief Names a change to a filled batch whose truth must be left as it was. */
    struct BadFlag
    {
        const char* name;        /**< Names the row. */
        float flag;              /**< The value written to slot 1 of image 1. */
    };

    /** @brief Provides the flags that are neither zero nor one. */
    class Medium_TruthFromRolesFlag : public Medium_TruthFromRoles,
                                      public ::testing::WithParamInterface<BadFlag>
    {

    };
} // namespace

/** @brief Tests that the description is boxes, classes and valid over M slots. */
TEST(Small_TruthRoles, DescribesBoxesClassesAndValidOverTheSlots)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Data::TargetDescription> roles,
                            describe_truth_roles(K_SLOTS, K_CLASSES));

    ASSERT_EQ(roles.size(), 3u);
    EXPECT_EQ(roles[0].role, Data::K_ROLE_BOXES);
    EXPECT_EQ(roles[0].sample_shape, (Shape{K_SLOTS, K_TRUTH_BOX_VALUES}));
    EXPECT_EQ(roles[0].dtype, DType::Float32);
    EXPECT_EQ(roles[0].classes, 0);

    EXPECT_EQ(roles[1].role, Data::K_ROLE_CLASSES);
    EXPECT_EQ(roles[1].sample_shape, Shape{K_SLOTS});
    EXPECT_EQ(roles[1].dtype, DType::Int32);
    EXPECT_EQ(roles[1].classes, K_CLASSES);

    EXPECT_EQ(roles[2].role, Data::K_ROLE_VALID);
    EXPECT_EQ(roles[2].sample_shape, Shape{K_SLOTS});
    EXPECT_EQ(roles[2].dtype, DType::Float32);
    EXPECT_EQ(roles[2].classes, 0);
}

/** @brief Tests that a class count of 0 is described, for a dataset of any count. */
TEST(Small_TruthRoles, DescribesAClassCountOfZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Data::TargetDescription> roles,
                            describe_truth_roles(1, 0));
    EXPECT_EQ(roles[1].classes, 0);
}

/** @brief Tests that no slot is refused with its value. */
TEST(Small_TruthRoles, RefusesNoSlot)
{
    SA_EXPECT_FAILURE(describe_truth_roles(0, K_CLASSES), StatusCode::InvalidDescription,
                      "object slot", "got 0");
}

/** @brief Tests that a negative class count is refused with its value. */
TEST(Small_TruthRoles, RefusesANegativeClassCount)
{
    SA_EXPECT_FAILURE(describe_truth_roles(K_SLOTS, -1), StatusCode::InvalidDescription,
                      "classes", "got -1");
}

/** @brief Tests that the corners are copied and the batch states its size. */
TEST_F(Medium_TruthFromRoles, CopiesTheCorners)
{
    const GroundTruthBatch truth = truth_of_filled_batch();

    EXPECT_EQ(truth.batch, K_IMAGES);
    EXPECT_EQ(truth.max_boxes, K_SLOTS);
    EXPECT_EQ(truth.boxes, K_CORNERS);
}

/** @brief Tests that the classes are widened to 64 bits. */
TEST_F(Medium_TruthFromRoles, WidensTheClasses)
{
    EXPECT_EQ(truth_of_filled_batch().labels, (std::vector<std::int64_t>{4, 0, 2, 0, 0, 1}));
}

/** @brief Tests that the flags are made bytes. */
TEST_F(Medium_TruthFromRoles, MakesTheFlagsBytes)
{
    EXPECT_EQ(truth_of_filled_batch().valid, (std::vector<std::uint8_t>{1, 0, 1, 0, 0, 1}));
}

/** @brief Tests that the roles are found by name, in whatever order the batch holds them. */
TEST_F(Medium_TruthFromRoles, FindsTheRolesInAnotherOrder)
{
    Data::Batch batch = filled_batch();
    std::swap(batch.targets[0], batch.targets[2]);

    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));
    EXPECT_EQ(truth.boxes, K_CORNERS);
    EXPECT_EQ(truth.valid, (std::vector<std::uint8_t>{1, 0, 1, 0, 0, 1}));
}

/** @brief Tests that a second batch of the same size is written into the same storage. */
TEST_F(Medium_TruthFromRoles, KeepsTheStorageOfATruthBetweenBatchesOfOneSize)
{
    Data::Batch batch = filled_batch();

    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));
    const float* const boxes = truth.boxes.data();
    const std::int64_t* const labels = truth.labels.data();
    const std::uint8_t* const valid = truth.valid.data();

    batch.targets[1].values.data_as<std::int32_t>()[0] = 3;
    SA_EXPECT_OK(truth_from_roles(batch, truth));

    EXPECT_EQ(truth.boxes.data(), boxes);
    EXPECT_EQ(truth.labels.data(), labels);
    EXPECT_EQ(truth.valid.data(), valid);
    EXPECT_EQ(truth.labels[0], 3);
}

/** @brief Tests that a batch without one of the three roles is refused with the role. */
TEST_P(Medium_TruthFromRolesMissing, RefusesABatchThatLacksARole)
{
    Data::Batch batch = filled_batch();
    const std::size_t missing = GetParam();
    const std::string role = batch.targets[missing].role;
    batch.targets.erase(batch.targets.begin() + static_cast<std::ptrdiff_t>(missing));

    GroundTruthBatch truth;
    SA_EXPECT_FAILURE(truth_from_roles(batch, truth), StatusCode::InvalidDescription,
                      "no role '" + role + "'", "it carries");
}

INSTANTIATE_TEST_SUITE_P(Roles, Medium_TruthFromRolesMissing, ::testing::Values(0u, 1u, 2u));

/** @brief Tests that a batch with no role at all is refused, naming the first role it needs. */
TEST_F(Medium_TruthFromRoles, RefusesABatchWithNoRole)
{
    GroundTruthBatch truth;

    SA_EXPECT_FAILURE(truth_from_roles(allocate({}), truth), StatusCode::InvalidDescription,
                      "no role 'boxes'", "(none)");
}

/** @brief Tests that a role the truth cannot read is refused with what it holds. */
TEST_P(Medium_TruthFromRolesBadRole, RefusesARoleTheTruthCannotRead)
{
    const BadRole& row = GetParam();
    std::vector<Data::TargetDescription> described = roles();
    row.change(described);
    GroundTruthBatch truth;

    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      row.needles[0], row.needles[1], row.needles[2]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_TruthFromRolesBadRole,
    ::testing::Values(
        BadRole{"ClassesOfFloat32",
                [](std::vector<Data::TargetDescription>& roles) { roles[1].dtype = DType::Float32; },
                {"'classes'", "int32", "float32 [2, 3]"}},
        BadRole{"ValidOfInt32",
                [](std::vector<Data::TargetDescription>& roles) { roles[2].dtype = DType::Int32; },
                {"'valid'", "float32", "int32 [2, 3]"}},
        BadRole{"BoxesOfRankThree",
                [](std::vector<Data::TargetDescription>& roles)
                { roles[0].sample_shape = Shape{K_SLOTS * K_TRUTH_BOX_VALUES}; },
                {"'boxes'", "rank 3", "[2, 12]"}},
        BadRole{"BoxesOfWidthThree",
                [](std::vector<Data::TargetDescription>& roles)
                { roles[0].sample_shape = Shape{K_SLOTS, 3}; },
                {"'boxes'", "[2, 3, 3]", "4 values"}},
        BadRole{"ClassesOfAnotherSlotCount",
                [](std::vector<Data::TargetDescription>& roles)
                { roles[1].sample_shape = Shape{K_SLOTS + 1}; },
                {"'classes'", "[2, 4]", "[2, 3, 4]"}},
        BadRole{"ValidOfAnotherSlotCount",
                [](std::vector<Data::TargetDescription>& roles)
                { roles[2].sample_shape = Shape{K_SLOTS - 1}; },
                {"'valid'", "[2, 2]", "[2, 3, 4]"}}),
    [](const ::testing::TestParamInfo<BadRole>& info) { return std::string(info.param.name); });

/** @brief Tests that a flag other than 0 and 1 is refused, naming the flag and its slot. */
TEST_P(Medium_TruthFromRolesFlag, RefusesAFlagThatIsNeitherZeroNorOne)
{
    Data::Batch batch = filled_batch();
    batch.targets[2].values.data_as<float>()[4] = GetParam().flag;
    GroundTruthBatch truth;

    SA_EXPECT_FAILURE(truth_from_roles(batch, truth), StatusCode::InvalidDescription, "'valid'",
                      "slot 1 of image 1");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_TruthFromRolesFlag,
    ::testing::Values(BadFlag{"Half", 0.5f},
                      BadFlag{"NotANumber", std::numeric_limits<float>::quiet_NaN()}),
    [](const ::testing::TestParamInfo<BadFlag>& info) { return std::string(info.param.name); });

/** @brief Tests that a refused batch leaves the boxes, the labels and the flags as they were. */
TEST_F(Medium_TruthFromRoles, ARefusedBatchLeavesTheWholeTruthUnchanged)
{
    Data::Batch batch = filled_batch();
    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));
    const GroundTruthBatch before = truth;

    batch.targets[1].values.data_as<std::int32_t>()[0] = 1;
    batch.targets[2].values.data_as<float>()[4] = 0.5f;
    EXPECT_FALSE(truth_from_roles(batch, truth).is_ok());

    EXPECT_EQ(truth.boxes, before.boxes);
    EXPECT_EQ(truth.labels, before.labels);
    EXPECT_EQ(truth.valid, before.valid);
}
