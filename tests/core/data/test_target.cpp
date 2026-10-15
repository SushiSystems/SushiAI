/**************************************************************************/
/* test_target.cpp                                                        */
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
 * @file test_target.cpp
 * @brief Tests the lookups by role, the batches allocate_batch and prepare_batch build, the input a
 *        dataset states, and the labels role of the synthetic classification set.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Data::BatchTarget;
using Data::TargetDescription;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the batches are allocated on. */
    class Medium_DataTarget : public SaTest::EngineTest
    {

    };

    /** @brief Returns the two descriptions the lookups and the allocation are tested with. */
    std::vector<TargetDescription> two_roles()
    {
        return {
            TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32, 5},
            TargetDescription{"boxes", Shape{3, 4}, DType::Float32, 0},
        };
    }

    /** @brief Returns a float32 input of @p sample per row. */
    Data::InputDescription float_input(const Shape& sample)
    {
        return Data::InputDescription{sample, DType::Float32};
    }

    /** @brief Returns the input of a token batch: seven int32 ids per row. */
    Data::InputDescription token_input()
    {
        return Data::InputDescription{Shape{7}, DType::Int32};
    }
} // namespace

/** @brief Tests that a description is found by its role and an unknown role yields null. */
TEST(Small_DataTargetLookup, ADescriptionIsFoundByItsRole)
{
    const std::vector<TargetDescription> targets = two_roles();

    EXPECT_EQ(Data::find_target(targets, Data::K_ROLE_LABELS), &targets[0]);
    EXPECT_EQ(Data::find_target(targets, "boxes"), &targets[1]);
    EXPECT_EQ(Data::find_target(targets, "mask"), nullptr);
    EXPECT_EQ(Data::find_target(std::vector<TargetDescription>{}, "boxes"), nullptr);
}

/** @brief Tests that a batch target is found by its role and an unknown role yields null. */
TEST(Small_DataTargetLookup, ABatchTargetIsFoundByItsRole)
{
    std::vector<BatchTarget> targets(2);
    targets[0].role = std::string(Data::K_ROLE_LABELS);
    targets[1].role = "boxes";

    EXPECT_EQ(Data::find_target(targets, Data::K_ROLE_LABELS), &targets[0]);
    EXPECT_EQ(Data::find_target(targets, "boxes"), &targets[1]);
    EXPECT_EQ(Data::find_target(targets, "mask"), nullptr);
}

/** @brief Tests that allocate_batch yields one tensor per description, in their order. */
TEST_F(Medium_DataTarget, AllocateBatchYieldsOneTensorPerDescriptionInOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            Data::allocate_batch(*engine, 8, float_input(Shape{6}),
                                                 two_roles()));

    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], 8);
    EXPECT_EQ(batch.inputs.shape[1], 6);
    EXPECT_EQ(batch.inputs.dtype, DType::Float32);

    ASSERT_EQ(batch.targets.size(), 2u);

    EXPECT_EQ(batch.targets[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(batch.targets[0].values.dtype, DType::Int32);
    ASSERT_EQ(batch.targets[0].values.rank, 1);
    EXPECT_EQ(batch.targets[0].values.shape[0], 8);

    EXPECT_EQ(batch.targets[1].role, "boxes");
    EXPECT_EQ(batch.targets[1].values.dtype, DType::Float32);
    ASSERT_EQ(batch.targets[1].values.rank, 3);
    EXPECT_EQ(batch.targets[1].values.shape[0], 8);
    EXPECT_EQ(batch.targets[1].values.shape[1], 3);
    EXPECT_EQ(batch.targets[1].values.shape[2], 4);
}

/** @brief Tests that allocate_batch with no description yields the inputs alone. */
TEST_F(Medium_DataTarget, AllocateBatchWithNoDescriptionYieldsTheInputsAlone)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            Data::allocate_batch(*engine, 2, float_input(Shape{4, 4, 1}), {}));

    EXPECT_EQ(batch.inputs.rank, 4);
    EXPECT_TRUE(batch.targets.empty());
}

/** @brief Tests that allocate_batch gives the input the element type its description names. */
TEST_F(Medium_DataTarget, AllocateBatchGivesTheInputItsStatedElementType)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            Data::allocate_batch(*engine, 3, token_input(), two_roles()));

    EXPECT_EQ(batch.inputs.dtype, DType::Int32);
    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], 3);
    EXPECT_EQ(batch.inputs.shape[1], 7);
    EXPECT_EQ(batch.targets[1].values.dtype, DType::Float32);
}

/** @brief Tests that prepare_batch keeps the tensors of a batch that holds the layout alone. */
TEST_F(Medium_DataTarget, PrepareBatchKeepsABatchThatHoldsTheLayoutAlone)
{
    Data::Batch batch;
    SA_EXPECT_OK(Data::prepare_batch(*engine, 3, token_input(), two_roles(), batch));
    const void* const first = batch.inputs.data();

    SA_EXPECT_OK(Data::prepare_batch(*engine, 3, token_input(), two_roles(), batch));

    EXPECT_EQ(batch.inputs.data(), first);
    EXPECT_EQ(batch.inputs.dtype, DType::Int32);
}

/** @brief Tests that prepare_batch replaces an input of the same shape and another type. */
TEST_F(Medium_DataTarget, PrepareBatchReplacesAnInputOfAnotherElementType)
{
    Data::Batch batch;
    SA_EXPECT_OK(Data::prepare_batch(*engine, 3, float_input(Shape{7}), two_roles(), batch));
    ASSERT_EQ(batch.inputs.dtype, DType::Float32);

    SA_EXPECT_OK(Data::prepare_batch(*engine, 3, token_input(), two_roles(), batch));

    EXPECT_EQ(batch.inputs.dtype, DType::Int32);
    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[1], 7);
}

/** @brief Tests that a dataset states float32 unless it overrides, and the type it was given. */
TEST_F(Medium_DataTarget, ADatasetStatesTheElementTypeOfItsInput)
{
    Data::SyntheticOptions options;
    options.batches = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> synthetic,
                            Data::SyntheticClassification::make(*engine, options));
    const Data::MaterialisedDataset tokens({}, 3, token_input(), {}, "tokens");

    EXPECT_EQ(synthetic->sample_dtype(), DType::Float32);
    EXPECT_EQ(Data::describe_input(*synthetic).dtype, DType::Float32);
    EXPECT_EQ(Data::describe_input(*synthetic).sample_shape, Shape{options.features});
    EXPECT_EQ(Data::describe_input(tokens).dtype, DType::Int32);
    EXPECT_EQ(Data::describe_input(tokens).sample_shape, Shape{7});
}

/** @brief Tests that the synthetic set publishes labels as one int32 per sample, in rotation. */
TEST_F(Medium_DataTarget, SyntheticClassificationPublishesItsLabelsInRotation)
{
    Data::SyntheticOptions options;
    options.batch_size = 4;
    options.batches = 2;
    options.features = 5;
    options.classes = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, options));

    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets().front().role, Data::K_ROLE_LABELS);
    EXPECT_EQ(dataset->targets().front().sample_shape, Shape{});
    EXPECT_EQ(dataset->targets().front().dtype, DType::Int32);
    EXPECT_EQ(dataset->targets().front().classes, 3);
    EXPECT_EQ(dataset->sample_shape(), Shape{5});

    const std::vector<std::vector<int32_t>> expected = {{0, 1, 2, 0}, {1, 2, 0, 1}};
    for (std::size_t b = 0; b < expected.size(); ++b)
    {
        Data::Batch dataset_batch;
        ASSERT_TRUE(SaTest::status_is_ok(dataset->read(b, dataset_batch)));
        const BatchTarget* labels =
            Data::find_target(dataset_batch.targets, Data::K_ROLE_LABELS);
        ASSERT_NE(labels, nullptr) << "batch " << b;
        ASSERT_EQ(labels->values.dtype, DType::Int32);
        ASSERT_EQ(labels->values.num_elements, 4);

        const int32_t* values = labels->values.data_as<int32_t>();
        EXPECT_EQ(std::vector<int32_t>(values, values + 4), expected[b]) << "batch " << b;
    }
}
