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
 * @file test_input_match.cpp
 * @brief Tests describe_input_leaf and match_input: what they accept and what a refusal names.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/input_match.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Data::InputDescription;

namespace
{
    /** @brief Publishes the input it is built with, no role and no batch. */
    class Published final : public Data::Dataset
    {
        public:
            /** @brief Holds @p input as the input the dataset publishes. */
            explicit Published(const InputDescription& input) : input_(input)
            {

            }

            int64_t batch_size() const noexcept override { return 4; }
            Shape sample_shape() const override { return input_.sample_shape; }
            DType sample_dtype() const noexcept override { return input_.dtype; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return 0; }
            std::string_view name() const noexcept override { return "published"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch&) const override
            {
                SA_FATAL_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return Status();
            }

        private:
            /** @brief Holds the input the dataset publishes. */
            InputDescription input_;

            /** @brief Holds no role. */
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Returns a leaf of @p shape and @p dtype. */
    Graph::Value leaf_of(const Shape& shape, DType dtype)
    {
        Graph::Value leaf;
        leaf.shape = shape;
        leaf.dtype = dtype;
        return leaf;
    }
} // namespace

/** @brief Tests that a leaf is described by its extents past the batch axis and its type. */
TEST(Unit_InputMatch, ALeafIsDescribedWithoutItsBatchAxis)
{
    SA_ASSERT_OK_AND_ASSIGN(const InputDescription tokens,
                            Train::describe_input_leaf(leaf_of(Shape{4, 16}, DType::Int32)));
    SA_ASSERT_OK_AND_ASSIGN(
        const InputDescription images,
        Train::describe_input_leaf(leaf_of(Shape{2, 8, 8, 3}, DType::Float32)));
    SA_ASSERT_OK_AND_ASSIGN(const InputDescription scalars,
                            Train::describe_input_leaf(leaf_of(Shape{4}, DType::Float64)));

    EXPECT_EQ(tokens.sample_shape, Shape{16});
    EXPECT_EQ(tokens.dtype, DType::Int32);
    EXPECT_EQ(images.sample_shape, (Shape{8, 8, 3}));
    EXPECT_EQ(images.dtype, DType::Float32);
    EXPECT_EQ(scalars.sample_shape, Shape{});
    EXPECT_EQ(scalars.dtype, DType::Float64);
}

/** @brief Tests that a leaf of rank 0, which has no batch axis, is refused. */
TEST(Unit_InputMatch, ALeafWithNoBatchAxisIsRefused)
{
    SA_EXPECT_FAILURE(Train::describe_input_leaf(leaf_of(Shape{}, DType::Float32)),
                      StatusCode::InvalidDescription, "no batch extent");
}

/** @brief Tests that a dataset publishing the shape and the type the step reads is accepted. */
TEST(Unit_InputMatch, ADatasetThatPublishesTheInputIsAccepted)
{
    const Published tokens(InputDescription{Shape{16}, DType::Int32});
    const Published reals(InputDescription{Shape{16}, DType::Float32});

    SA_EXPECT_OK(Train::match_input(InputDescription{Shape{16}, DType::Int32}, tokens));
    SA_EXPECT_OK(Train::match_input(InputDescription{Shape{16}, DType::Float32}, reals));
}

/** @brief Tests that an int32 dataset against a float32 leaf is refused with both types. */
TEST(Unit_InputMatch, AnotherElementTypeIsRefusedWithBothTypes)
{
    const Published tokens(InputDescription{Shape{16}, DType::Int32});
    const Published reals(InputDescription{Shape{16}, DType::Float32});

    SA_EXPECT_FAILURE(
        Train::match_input(InputDescription{Shape{16}, DType::Float32}, tokens),
        StatusCode::Incompatible, "'published'", "as int32 [16] per sample",
        "compiled for float32 [16] per sample");
    SA_EXPECT_FAILURE(Train::match_input(InputDescription{Shape{16}, DType::Int32}, reals),
                      StatusCode::Incompatible, "as float32 [16] per sample",
                      "compiled for int32 [16] per sample");
}

/** @brief Tests that another per-sample shape is refused with both shapes. */
TEST(Unit_InputMatch, AnotherShapeIsRefusedWithBothShapes)
{
    const Published tokens(InputDescription{Shape{16}, DType::Int32});

    SA_EXPECT_FAILURE(Train::match_input(InputDescription{Shape{32}, DType::Int32}, tokens),
                      StatusCode::Incompatible, "'published'", "as int32 [16] per sample",
                      "compiled for int32 [32] per sample");
}
