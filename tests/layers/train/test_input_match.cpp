/**************************************************************************/
/* test_input_match.cpp                                                   */
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
 * @file test_input_match.cpp
 * @brief Tests describe_input_leaf and match_input: what they accept and what a refusal names.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/input_match.hpp>

using namespace SushiAI;
using Data::InMemoryDataset;
using Data::InputDescription;

namespace
{
    /** @brief Returns a leaf of @p shape and @p dtype. */
    Graph::Value leaf_of(const Shape& shape, DType dtype)
    {
        Graph::Value leaf;
        leaf.shape = shape;
        leaf.dtype = dtype;
        return leaf;
    }

    /** @brief Returns an in-memory dataset that publishes @p input, with no batch and no role. */
    InMemoryDataset publishing(const InputDescription& input)
    {
        return InMemoryDataset(0, 4, input, {});
    }

    /** @brief Holds a leaf and the description of it past the batch axis. */
    struct LeafRow
    {
        std::string name;       /**< The row's name. */
        Shape leaf;             /**< The shape of the leaf. */
        DType dtype;            /**< The element type of the leaf. */
        Shape sample_shape;     /**< The shape of one sample the description holds. */
    };

    /** @brief Describes the leaf of each row. */
    class Small_InputMatchLeaf : public ::testing::TestWithParam<LeafRow>
    {

    };

    /** @brief Holds the input a step reads, the input a dataset publishes and the refusal. */
    struct MatchRow
    {
        std::string name;               /**< The row's name. */
        InputDescription required;      /**< The input the step reads. */
        InputDescription published;     /**< The input the dataset publishes. */
        std::string first;              /**< A text the message holds. */
        std::string second;             /**< Another text the message holds. */
    };

    /** @brief Matches the dataset of each row against the input of the step. */
    class Small_InputMatchRefusal : public ::testing::TestWithParam<MatchRow>
    {

    };

    /** @brief Matches a dataset against the input it publishes, for each accepting row. */
    class Small_InputMatchAccepted : public ::testing::TestWithParam<InputDescription>
    {

    };
} // namespace

/** @brief Tests that a leaf is described by its extents past the batch axis and its type. */
TEST_P(Small_InputMatchLeaf, ALeafIsDescribedWithoutItsBatchAxis)
{
    const LeafRow& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const InputDescription described,
                            Train::describe_input_leaf(leaf_of(row.leaf, row.dtype)));

    EXPECT_EQ(described.sample_shape, row.sample_shape);
    EXPECT_EQ(described.dtype, row.dtype);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InputMatchLeaf,
    ::testing::Values(LeafRow{"Tokens", Shape{4, 16}, DType::Int32, Shape{16}},
                      LeafRow{"Images", Shape{2, 8, 8, 3}, DType::Float32, Shape{8, 8, 3}},
                      LeafRow{"Scalars", Shape{4}, DType::Float64, Shape{}}),
    [](const ::testing::TestParamInfo<LeafRow>& info) { return info.param.name; });

/** @brief Tests that a leaf of rank 0, which has no batch axis, is refused. */
TEST(Small_InputMatch, ALeafWithNoBatchAxisIsRefused)
{
    SA_EXPECT_FAILURE(Train::describe_input_leaf(leaf_of(Shape{}, DType::Float32)),
                      StatusCode::InvalidDescription, "no batch extent");
}

/** @brief Tests that a dataset publishing the shape and the type the step reads is accepted. */
TEST_P(Small_InputMatchAccepted, ADatasetThatPublishesTheInputIsAccepted)
{
    const InputDescription& input = GetParam();
    const InMemoryDataset dataset = publishing(input);

    SA_EXPECT_OK(Train::match_input(input, dataset));
}

INSTANTIATE_TEST_SUITE_P(Inputs, Small_InputMatchAccepted,
                         ::testing::Values(InputDescription{Shape{16}, DType::Int32},
                                           InputDescription{Shape{16}, DType::Float32}));

/** @brief Tests that another element type or shape is refused with both descriptions. */
TEST_P(Small_InputMatchRefusal, AnotherInputIsRefusedWithBothDescriptions)
{
    const MatchRow& row = GetParam();
    const InMemoryDataset dataset = publishing(row.published);

    SA_EXPECT_FAILURE(Train::match_input(row.required, dataset), StatusCode::Incompatible,
                      "'in-memory'", row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_InputMatchRefusal,
    ::testing::Values(
        MatchRow{"AnInt32DatasetAgainstAFloat32Leaf", InputDescription{Shape{16}, DType::Float32},
                 InputDescription{Shape{16}, DType::Int32}, "as int32 [16] per sample",
                 "compiled for float32 [16] per sample"},
        MatchRow{"AFloat32DatasetAgainstAnInt32Leaf", InputDescription{Shape{16}, DType::Int32},
                 InputDescription{Shape{16}, DType::Float32}, "as float32 [16] per sample",
                 "compiled for int32 [16] per sample"},
        MatchRow{"AnotherPerSampleShape", InputDescription{Shape{32}, DType::Int32},
                 InputDescription{Shape{16}, DType::Int32}, "as int32 [16] per sample",
                 "compiled for int32 [32] per sample"}),
    [](const ::testing::TestParamInfo<MatchRow>& info) { return info.param.name; });
