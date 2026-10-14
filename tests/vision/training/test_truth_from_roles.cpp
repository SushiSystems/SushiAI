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
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
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
    class Unit_TruthFromRoles : public Test::EngineTest
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
    };
} // namespace

/** @brief Tests that the description is boxes, classes and valid over M slots. */
TEST(Unit_TruthRoles, DescribesBoxesClassesAndValidOverTheSlots)
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
TEST(Unit_TruthRoles, DescribesAClassCountOfZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Data::TargetDescription> roles,
                            describe_truth_roles(1, 0));
    EXPECT_EQ(roles[1].classes, 0);
}

/** @brief Tests that no slot and a negative class count are refused with their values. */
TEST(Unit_TruthRoles, RefusesNoSlotAndANegativeClassCount)
{
    SA_EXPECT_FAILURE(describe_truth_roles(0, K_CLASSES), StatusCode::InvalidDescription,
                      "object slot", "got 0");
    SA_EXPECT_FAILURE(describe_truth_roles(K_SLOTS, -1), StatusCode::InvalidDescription,
                      "classes", "got -1");
}

/**
 * @brief Tests that the corners are copied, the classes widened and the flags made bytes.
 *
 * Every value is a copy or an exact conversion, so the tolerance is zero.
 */
TEST_F(Unit_TruthFromRoles, CopiesTheCornersWidensTheClassesAndMakesTheFlagsBytes)
{
    const Data::Batch batch = filled_batch();

    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));

    EXPECT_EQ(truth.batch, K_IMAGES);
    EXPECT_EQ(truth.max_boxes, K_SLOTS);
    EXPECT_EQ(truth.boxes, K_CORNERS);
    EXPECT_EQ(truth.labels, (std::vector<std::int64_t>{4, 0, 2, 0, 0, 1}));
    EXPECT_EQ(truth.valid, (std::vector<std::uint8_t>{1, 0, 1, 0, 0, 1}));
}

/** @brief Tests that the roles are found by name, in whatever order the batch holds them. */
TEST_F(Unit_TruthFromRoles, FindsTheRolesInAnotherOrder)
{
    Data::Batch batch = filled_batch();
    std::swap(batch.targets[0], batch.targets[2]);

    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));
    EXPECT_EQ(truth.boxes, K_CORNERS);
    EXPECT_EQ(truth.valid, (std::vector<std::uint8_t>{1, 0, 1, 0, 0, 1}));
}

/** @brief Tests that a second batch of the same size is written into the same storage. */
TEST_F(Unit_TruthFromRoles, KeepsTheStorageOfATruthBetweenBatchesOfOneSize)
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
TEST_F(Unit_TruthFromRoles, RefusesABatchThatLacksARole)
{
    for (std::size_t missing = 0; missing < 3; ++missing)
    {
        Data::Batch batch = filled_batch();
        const std::string role = batch.targets[missing].role;
        batch.targets.erase(batch.targets.begin() + static_cast<std::ptrdiff_t>(missing));

        GroundTruthBatch truth;
        SA_EXPECT_FAILURE(truth_from_roles(batch, truth), StatusCode::InvalidDescription,
                          "no role '" + role + "'", "it carries");
    }

    GroundTruthBatch truth;
    SA_EXPECT_FAILURE(truth_from_roles(allocate({}), truth), StatusCode::InvalidDescription,
                      "no role 'boxes'", "(none)");
}

/** @brief Tests that a role of another element type or rank is refused with what it holds. */
TEST_F(Unit_TruthFromRoles, RefusesARoleOfAnotherElementTypeOrRank)
{
    std::vector<Data::TargetDescription> described = roles();
    described[1].dtype = DType::Float32;
    GroundTruthBatch truth;
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'classes'", "int32", "float32 [2, 3]");

    described = roles();
    described[2].dtype = DType::Int32;
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'valid'", "float32", "int32 [2, 3]");

    described = roles();
    described[0].sample_shape = Shape{K_SLOTS * K_TRUTH_BOX_VALUES};
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'boxes'", "rank 3", "[2, 12]");
}

/** @brief Tests that a box of three values and roles of other slot counts are refused. */
TEST_F(Unit_TruthFromRoles, RefusesABoxOfAnotherWidthAndRolesOfOtherSlots)
{
    std::vector<Data::TargetDescription> described = roles();
    described[0].sample_shape = Shape{K_SLOTS, 3};
    GroundTruthBatch truth;
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'boxes'", "[2, 3, 3]", "4 values");

    described = roles();
    described[1].sample_shape = Shape{K_SLOTS + 1};
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'classes'", "[2, 4]", "[2, 3, 4]");

    described = roles();
    described[2].sample_shape = Shape{K_SLOTS - 1};
    SA_EXPECT_FAILURE(truth_from_roles(allocate(described), truth), StatusCode::Incompatible,
                      "'valid'", "[2, 2]", "[2, 3, 4]");
}

/** @brief Tests that a flag other than 0 and 1 is refused and the truth is left as it was. */
TEST_F(Unit_TruthFromRoles, RefusesAFlagThatIsNeitherZeroNorOneAndLeavesTheTruth)
{
    Data::Batch batch = filled_batch();
    GroundTruthBatch truth;
    SA_EXPECT_OK(truth_from_roles(batch, truth));

    batch.targets[1].values.data_as<std::int32_t>()[0] = 1;
    batch.targets[2].values.data_as<float>()[4] = 0.5f;
    SA_EXPECT_FAILURE(truth_from_roles(batch, truth), StatusCode::InvalidDescription, "'valid'",
                      "0.5", "slot 1 of image 1");
    EXPECT_EQ(truth.labels[0], 4);

    batch.targets[2].values.data_as<float>()[4] = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_FAILURE(truth_from_roles(batch, truth), StatusCode::InvalidDescription, "'valid'",
                      "slot 1 of image 1");
}
