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
 * @file test_target.cpp
 * @brief Tests the lookups by role, the batches allocate_batch and prepare_batch build, the input a
 *        dataset states, and the labels role of the synthetic classification set.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
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

    /** @brief Holds the element type and the extents of one tensor. */
    struct TensorLayout
    {
        /** @brief Names the element type. */
        DType dtype = DType::Float32;

        /** @brief Holds the extents, outermost first. */
        std::vector<int64_t> shape;
    };

    /** @brief Returns whether two layouts hold the same type and extents. */
    bool operator==(const TensorLayout& left, const TensorLayout& right)
    {
        return left.dtype == right.dtype && left.shape == right.shape;
    }

    /** @brief Returns the type and the extents of @p tensor. */
    TensorLayout layout_of(const SushiBLAS::Tensor& tensor)
    {
        TensorLayout layout;
        layout.dtype = tensor.dtype;
        for (int dim = 0; dim < tensor.rank; ++dim)
            layout.shape.push_back(tensor.shape[dim]);

        return layout;
    }

    /** @brief Returns the role and the layout of every target of @p batch, in order. */
    std::vector<std::pair<std::string, TensorLayout>> target_layouts_of(const Data::Batch& batch)
    {
        std::vector<std::pair<std::string, TensorLayout>> layouts;
        for (const BatchTarget& target : batch.targets)
            layouts.emplace_back(target.role, layout_of(target.values));

        return layouts;
    }

    /** @brief Names one allocate_batch call and the layout it must give. */
    struct AllocateCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the samples per batch. */
        int64_t rows;

        /** @brief Describes the input. */
        Data::InputDescription input;

        /** @brief Describes the roles; empty for none. */
        std::vector<TargetDescription> roles;

        /** @brief Holds the layout the inputs must have. */
        TensorLayout inputs;

        /** @brief Holds the role and layout of each target, in the order of @ref roles. */
        std::vector<std::pair<std::string, TensorLayout>> targets;
    };

    /** @brief Returns the rows of the allocate_batch table. */
    std::vector<AllocateCase> allocate_cases()
    {
        return {
            AllocateCase{"OneTensorPerDescriptionInOrder", 8, float_input(Shape{6}), two_roles(),
                         TensorLayout{DType::Float32, {8, 6}},
                         {{std::string(Data::K_ROLE_LABELS), TensorLayout{DType::Int32, {8}}},
                          {"boxes", TensorLayout{DType::Float32, {8, 3, 4}}}}},
            AllocateCase{"NoDescriptionYieldsTheInputsAlone", 2, float_input(Shape{4, 4, 1}), {},
                         TensorLayout{DType::Float32, {2, 4, 4, 1}}, {}},
            AllocateCase{"TheInputKeepsItsStatedElementType", 3, token_input(), two_roles(),
                         TensorLayout{DType::Int32, {3, 7}},
                         {{std::string(Data::K_ROLE_LABELS), TensorLayout{DType::Int32, {3}}},
                          {"boxes", TensorLayout{DType::Float32, {3, 3, 4}}}}},
        };
    }

    /** @brief Runs each allocate_batch row on the engine. */
    class Medium_DataTargetAllocation : public SaTest::EngineTest,
                                        public ::testing::WithParamInterface<AllocateCase>
    {

    };

    /** @brief Fills a batch the way a row of the prepare_batch table starts; keeps what it needs. */
    using Starter = void (*)(SushiBLAS::Engine& engine, Data::Batch& batch,
                             std::unique_ptr<Data::Dataset>& owner);

    /** @brief Names the starting batch of one prepare_batch call and whether it is kept. */
    struct PrepareCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Fills the batch prepare_batch is then called on. */
        Starter start;

        /** @brief States whether prepare_batch leaves the input tensor where it was. */
        bool is_kept;
    };

    /** @brief Starts from a batch that already holds the token layout. */
    void start_from_the_layout(SushiBLAS::Engine& engine, Data::Batch& batch,
                               std::unique_ptr<Data::Dataset>&)
    {
        SA_EXPECT_OK(Data::prepare_batch(engine, 3, token_input(), two_roles(), batch));
    }

    /** @brief Starts from a batch of the same shape and another element type. */
    void start_from_another_element_type(SushiBLAS::Engine& engine, Data::Batch& batch,
                                         std::unique_ptr<Data::Dataset>&)
    {
        SA_EXPECT_OK(Data::prepare_batch(engine, 3, float_input(Shape{7}), two_roles(), batch));
    }

    /** @brief Starts from a batch that a dataset read handed out, so @p owner shares it. */
    void start_from_a_dataset_read(SushiBLAS::Engine& engine, Data::Batch& batch,
                                   std::unique_ptr<Data::Dataset>& owner)
    {
        std::vector<Data::Batch> held(1);
        SA_EXPECT_OK(Data::prepare_batch(engine, 3, token_input(), two_roles(), held[0]));
        owner = std::make_unique<Data::MaterialisedDataset>(std::move(held), 3, token_input(),
                                                            two_roles(), "held");

        SA_EXPECT_OK(owner->read(0, batch));
    }

    /** @brief Runs each prepare_batch row on the engine. */
    class Medium_DataTargetPreparation : public SaTest::EngineTest,
                                         public ::testing::WithParamInterface<PrepareCase>
    {

    };

    /** @brief Names the labels a batch of the synthetic set must publish. */
    struct LabelsCase
    {
        /** @brief Counts the batch in the set. */
        std::size_t batch;

        /** @brief Holds the class index of every sample of the batch. */
        std::vector<int32_t> labels;
    };

    /** @brief Runs each label row on the engine. */
    class Medium_DataTargetLabels : public SaTest::EngineTest,
                                    public ::testing::WithParamInterface<LabelsCase>
    {

    };

    /** @brief Checks that a list of two @p Target values is searched by role, for either kind. */
    template <typename Target>
    void expect_found_by_role()
    {
        std::vector<Target> targets(2);
        targets[0].role = std::string(Data::K_ROLE_LABELS);
        targets[1].role = "boxes";

        EXPECT_EQ(Data::find_target(targets, Data::K_ROLE_LABELS), &targets[0]);
        EXPECT_EQ(Data::find_target(targets, "boxes"), &targets[1]);
        EXPECT_EQ(Data::find_target(targets, "mask"), nullptr);
        EXPECT_EQ(Data::find_target(std::vector<Target>{}, "boxes"), nullptr);
    }
} // namespace

/** @brief Tests that a description and a batch target are each found by their role. */
TEST(Small_DataTargetLookup, ATargetOfEitherKindIsFoundByItsRole)
{
    expect_found_by_role<TargetDescription>();
    expect_found_by_role<BatchTarget>();
}

/** @brief Tests that allocate_batch gives the inputs and each target the layout the row states. */
TEST_P(Medium_DataTargetAllocation, AllocateBatchGivesTheLayoutOfItsDescriptions)
{
    const AllocateCase& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            Data::allocate_batch(*engine, row.rows, row.input, row.roles));

    EXPECT_TRUE(layout_of(batch.inputs) == row.inputs);
    EXPECT_TRUE(target_layouts_of(batch) == row.targets);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_DataTargetAllocation,
                         ::testing::ValuesIn(allocate_cases()),
                         [](const ::testing::TestParamInfo<AllocateCase>& info)
                         {
                             return std::string(info.param.name);
                         });

/** @brief Tests that prepare_batch keeps an input of the layout and replaces any other. */
TEST_P(Medium_DataTargetPreparation, PrepareBatchKeepsOrReplacesTheInput)
{
    const PrepareCase& row = GetParam();
    Data::Batch batch;
    std::unique_ptr<Data::Dataset> owner;
    row.start(*engine, batch, owner);
    const void* const before = batch.inputs.data();

    SA_EXPECT_OK(Data::prepare_batch(*engine, 3, token_input(), two_roles(), batch));

    EXPECT_EQ(batch.inputs.data() == before, row.is_kept);
    EXPECT_TRUE(layout_of(batch.inputs) == (TensorLayout{DType::Int32, {3, 7}}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_DataTargetPreparation,
    ::testing::Values(PrepareCase{"ALayoutAlreadyHeldIsKept", &start_from_the_layout, true},
                      PrepareCase{"AnotherElementTypeIsReplaced", &start_from_another_element_type,
                                  false},
                      PrepareCase{"ATensorADatasetSharesIsReplaced", &start_from_a_dataset_read,
                                  false}),
    [](const ::testing::TestParamInfo<PrepareCase>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that describe_input reports the type and the sample shape a dataset overrides. */
TEST_F(Medium_DataTarget, DescribeInputReportsTheTypeAndShapeADatasetOverrides)
{
    const Data::MaterialisedDataset tokens({}, 3, token_input(), {}, "tokens");

    const Data::InputDescription described = Data::describe_input(tokens);

    EXPECT_EQ(described.dtype, DType::Int32);
    EXPECT_EQ(described.sample_shape, Shape{7});
}

/** @brief Tests that the synthetic set labels the samples of a batch in rotation over its classes. */
TEST_P(Medium_DataTargetLabels, SyntheticClassificationPublishesItsLabelsInRotation)
{
    const LabelsCase& row = GetParam();
    Data::SyntheticOptions options;
    options.batch_size = 4;
    options.batches = 2;
    options.features = 5;
    options.classes = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, options));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(row.batch, batch)));

    const BatchTarget* labels = Data::find_target(batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    ASSERT_EQ(labels->values.dtype, DType::Int32);
    ASSERT_EQ(labels->values.num_elements, 4);
    const int32_t* values = labels->values.data_as<int32_t>();
    EXPECT_EQ(std::vector<int32_t>(values, values + 4), row.labels);
}

INSTANTIATE_TEST_SUITE_P(Batches, Medium_DataTargetLabels,
                         ::testing::Values(LabelsCase{0, {0, 1, 2, 0}},
                                           LabelsCase{1, {1, 2, 0, 1}}));
