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
 * @file test_dense_targets.cpp
 * @brief Tests the host fills that copy the real values of a batch role into a target leaf.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/dense_targets.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the value every element of a leaf has before a fill. */
    constexpr double K_STALE = 7.0;

    /** @brief Provides the host engine the role and leaf tensors are allocated on. */
    class HostTensors
    {
        protected:
            /** @brief Returns a tensor of @p shape and @p dtype that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(const Shape& shape, DType dtype,
                                                              const std::vector<double>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor, allocate_tensor(backend.engine(), shape, dtype));
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

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Fills and refuses on the host engine. */
    class Small_DenseTargets : public HostTensors, public ::testing::Test
    {
    };

    /** @brief Fills a leaf of each real element type. */
    class Small_DenseTargetsRealType : public HostTensors,
                                       public ::testing::TestWithParam<DType>
    {
    };

    /** @brief Holds a source, a leaf and the words of the refusal their pairing gets. */
    struct PairRow
    {
        std::string name;       /**< The row's name. */
        Shape source_shape;     /**< The shape of the source values. */
        DType source_dtype;     /**< The element type of the source values. */
        Shape leaf_shape;       /**< The shape of the leaf. */
        DType leaf_dtype;       /**< The element type of the leaf. */
        std::string first;      /**< A text the message holds. */
        std::string second;     /**< Another text the message holds. */
    };

    /** @brief Refuses each pairing of a source with a leaf. */
    class Small_DenseTargetsPair : public HostTensors, public ::testing::TestWithParam<PairRow>
    {
    };

    /** @brief Holds six values that float32 represents exactly. */
    const std::vector<double> K_VALUES = {0.5, -1.25, 3.0, 0.0, 2.75, -8.0};

    /** @brief Holds four labels, each 0 or 1. */
    const std::vector<double> K_LABELS = {1.0, 0.0, 0.0, 1.0};
} // namespace

/** @brief Tests that a float32 and a float64 leaf both receive the values in order. */
TEST_P(Small_DenseTargetsRealType, ValuesAreCopiedInOrderIntoEitherRealType)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor values,
                            tensor_of(Shape{2, 3}, DType::Float32, K_VALUES));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{2, 3}, GetParam()));

    SA_EXPECT_OK(Train::fill_dense_targets(values, leaf));

    EXPECT_EQ(SaTest::read_tensor(leaf), K_VALUES);
}

INSTANTIATE_TEST_SUITE_P(Types, Small_DenseTargetsRealType,
                         ::testing::Values(DType::Float32, DType::Float64),
                         [](const ::testing::TestParamInfo<DType>& info)
                         { return std::string(dtype_name(info.param)); });

/** @brief Tests that [N] labels fill an [N, 1] leaf, which holds as many elements. */
TEST_F(Small_DenseTargets, AColumnLeafTakesOneLabelPerRow)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            tensor_of(Shape{4}, DType::Float32, K_LABELS));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{4, 1}, DType::Float32));

    SA_EXPECT_OK(Train::fill_binary_targets(labels, leaf));

    EXPECT_EQ(SaTest::read_tensor(leaf), K_LABELS);
}

/** @brief Tests that another count or element type is refused and leaves the leaf as it was. */
TEST_P(Small_DenseTargetsPair, AnotherCountOrTypeIsRefusedAndTheLeafKept)
{
    const PairRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(
        const SushiBLAS::Tensor source,
        tensor_of(row.source_shape, row.source_dtype,
                  std::vector<double>(static_cast<std::size_t>(row.source_shape.num_elements()),
                                      0.0)));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(row.leaf_shape, row.leaf_dtype));

    SA_EXPECT_FAILURE(Train::fill_dense_targets(source, leaf), StatusCode::Incompatible,
                      row.first, row.second);

    EXPECT_EQ(SaTest::read_tensor(leaf),
              std::vector<double>(static_cast<std::size_t>(row.leaf_shape.num_elements()),
                                  K_STALE));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DenseTargetsPair,
    ::testing::Values(PairRow{"AnotherCount", Shape{2, 3}, DType::Float32, Shape{4, 1},
                              DType::Float32, "holds 6 values", "targets hold 4"},
                      PairRow{"AnIntegerSource", Shape{4}, DType::Int32, Shape{4, 1},
                              DType::Float32, "int32", "float32"},
                      PairRow{"AnIntegerLeaf", Shape{2, 3}, DType::Float32, Shape{2, 3},
                              DType::Int32, "targets are int32", ""}),
    [](const ::testing::TestParamInfo<PairRow>& info) { return info.param.name; });

/** @brief Tests that a label which is neither 0 nor 1 is named and nothing is written. */
TEST_F(Small_DenseTargets, ALabelThatIsNotZeroOrOneIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            tensor_of(Shape{4}, DType::Float32, {1.0, 0.0, 0.5, 1.0}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor leaf, stale_leaf(Shape{4, 1}, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_binary_targets(labels, leaf), StatusCode::Incompatible,
                      "element 2", "0.5", "0 or 1");

    EXPECT_EQ(SaTest::read_tensor(leaf), std::vector<double>(4, K_STALE));
}
