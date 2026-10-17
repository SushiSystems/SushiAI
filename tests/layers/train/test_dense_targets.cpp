/**************************************************************************/
/* test_dense_targets.cpp                                                 */
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
 * @file test_dense_targets.cpp
 * @brief Tests the host fills that copy the real values of a batch role into a target leaf.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/dense_targets.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every element of a leaf has before a fill. */
    constexpr double K_STALE = 7.0;

    /** @brief Provides the engine the role and leaf tensors are allocated on. */
    class Medium_DenseTargets : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape, DType dtype,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(*engine, shape, dtype));
                SaTest::write_tensor(tensor, values);
                return tensor;
            }

            /** @brief Returns a leaf of @p shape and @p dtype with K_STALE in every element. */
            [[nodiscard]] Result<SushiBLAS::Tensor> stale_leaf(const Shape& shape, DType dtype)
            {
                return tensor_of(shape, dtype,
                                 std::vector<double>(
                                     static_cast<std::size_t>(shape.num_elements()), K_STALE));
            }
    };

    /** @brief Holds six values that float32 represents exactly. */
    const std::vector<double> K_VALUES = {0.5, -1.25, 3.0, 0.0, 2.75, -8.0};

    /** @brief Holds four labels, each 0 or 1. */
    const std::vector<double> K_LABELS = {1.0, 0.0, 0.0, 1.0};
} // namespace

/** @brief Tests that a float32 and a float64 leaf both receive the values in order. */
TEST_F(Medium_DenseTargets, ValuesAreCopiedInOrderIntoEitherRealType)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor values,
                            tensor_of(Shape{2, 3}, DType::Float32, K_VALUES));

    for (const DType dtype : {DType::Float32, DType::Float64})
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{2, 3}, dtype));
        SA_EXPECT_OK(Train::fill_dense_targets(values, leaf));
        EXPECT_EQ(SaTest::read_tensor(leaf), K_VALUES) << dtype_name(dtype);
    }
}

/** @brief Tests that [N] labels fill an [N, 1] leaf, which holds as many elements. */
TEST_F(Medium_DenseTargets, AColumnLeafTakesOneLabelPerRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            tensor_of(Shape{4}, DType::Float32, K_LABELS));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{4, 1}, DType::Float32));

    SA_EXPECT_OK(Train::fill_binary_targets(labels, leaf));

    EXPECT_EQ(SaTest::read_tensor(leaf), K_LABELS);
}

/** @brief Tests that another count or element type is refused and leaves the leaf as it was. */
TEST_F(Medium_DenseTargets, AnotherCountOrTypeIsRefusedAndTheLeafKept)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor values,
                            tensor_of(Shape{2, 3}, DType::Float32, K_VALUES));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor indices, allocate_tensor(*engine, Shape{4},
                                                                             DType::Int32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{4, 1}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor integer_leaf,
                            allocate_tensor(*engine, Shape{2, 3}, DType::Int32));

    SA_EXPECT_FAILURE(Train::fill_dense_targets(values, leaf), StatusCode::Incompatible,
                      "holds 6 values", "targets hold 4");
    SA_EXPECT_FAILURE(Train::fill_dense_targets(indices, leaf), StatusCode::Incompatible,
                      "int32", "float32");
    SA_EXPECT_FAILURE(Train::fill_dense_targets(values, integer_leaf),
                      StatusCode::Incompatible, "targets are int32");

    EXPECT_EQ(SaTest::read_tensor(leaf), std::vector<double>(4, K_STALE));
}

/** @brief Tests that a label which is neither 0 nor 1 is named and nothing is written. */
TEST_F(Medium_DenseTargets, ALabelThatIsNotZeroOrOneIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            tensor_of(Shape{4}, DType::Float32, {1.0, 0.0, 0.5, 1.0}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{4, 1}, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_binary_targets(labels, leaf), StatusCode::Incompatible,
                      "element 2", "0.5", "0 or 1");

    EXPECT_EQ(SaTest::read_tensor(leaf), std::vector<double>(4, K_STALE));
}
