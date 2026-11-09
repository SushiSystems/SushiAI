/**************************************************************************/
/* test_one_hot.cpp                                                       */
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
 * @file test_one_hot.cpp
 * @brief Tests fill_one_hot: the rows it writes, both element types, and what it refuses.
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
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/one_hot.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the host engine the label and row tensors are allocated on. */
    class Small_OneHot : public ::testing::Test
    {
        protected:
            /** @brief Returns an int32 tensor of one extent that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> labels_of(const std::vector<int32_t>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor labels,
                              allocate_tensor(backend.engine(),
                                              Shape{static_cast<int64_t>(values.size())},
                                              DType::Int32));

                int32_t* data = labels.data_as<int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    data[i] = values[i];

                return labels;
            }

            /** @brief Returns a [rows, classes] tensor of @p dtype with 7 in every element. */
            [[nodiscard]] Result<SushiBLAS::Tensor> stale_rows(int64_t rows, int64_t classes,
                                                               DType dtype)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor one_hot,
                              allocate_tensor(backend.engine(), Shape{rows, classes}, dtype));
                SaTest::write_tensor(
                    one_hot, std::vector<double>(static_cast<std::size_t>(rows * classes), 7.0));

                return one_hot;
            }

            /** @brief The host backend the tensors are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the element type of the leaf a fill writes. */
    struct DTypeRow
    {
        std::string name;   /**< The row's name. */
        DType dtype;        /**< The element type of the leaf. */
    };

    /** @brief Fills a leaf of the element type of each row. */
    class Small_OneHotDType : public Small_OneHot,
                              public ::testing::WithParamInterface<DTypeRow>
    {

    };

    /** @brief Holds labels outside three classes and the words of the refusal. */
    struct OutOfRangeRow
    {
        std::string name;               /**< The row's name. */
        std::vector<int32_t> labels;    /**< The labels, one of them out of range. */
        std::string first;              /**< The row the message names. */
        std::string second;             /**< The label the message names. */
    };

    /** @brief Refuses the labels of each row and writes nothing. */
    class Small_OneHotOutOfRange : public Small_OneHot,
                                   public ::testing::WithParamInterface<OutOfRangeRow>
    {

    };

    /** @brief Holds the rows the labels 2, 0, 1, 2 select among three classes. */
    const std::vector<double> K_EXPECTED_ROWS = {
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
} // namespace

/**
 * @brief Tests that every row holds 1 in its label's column and 0 elsewhere, in either type.
 *        The values are 0 and 1, which both types hold exactly, so the comparison is equality.
 */
TEST_P(Small_OneHotDType, EveryRowHoldsOneInItsLabelsColumn)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({2, 0, 1, 2}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, GetParam().dtype));

    SA_EXPECT_OK(Train::fill_one_hot(labels, one_hot));

    EXPECT_EQ(SaTest::read_tensor(one_hot), K_EXPECTED_ROWS);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_OneHotDType,
    ::testing::Values(DTypeRow{"Float32", DType::Float32}, DTypeRow{"Float64", DType::Float64}),
    [](const ::testing::TestParamInfo<DTypeRow>& info) { return info.param.name; });

/** @brief Tests that a second fill replaces the rows of the first. */
TEST_F(Small_OneHot, ASecondFillReplacesTheFirst)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor first, labels_of({0, 0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor second, labels_of({1, 0}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(2, 2, DType::Float32));

    SA_EXPECT_OK(Train::fill_one_hot(first, one_hot));
    SA_EXPECT_OK(Train::fill_one_hot(second, one_hot));

    EXPECT_EQ(SaTest::read_tensor(one_hot), (std::vector<double>{0.0, 1.0, 1.0, 0.0}));
}

/** @brief Tests that a label outside the classes is named and no row is written. */
TEST_P(Small_OneHotOutOfRange, ALabelOutsideTheClassesIsRefusedAndNothingIsWritten)
{
    const OutOfRangeRow& row = GetParam();
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of(row.labels));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(2, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, row.first,
                      row.second, "3 classes");

    EXPECT_EQ(SaTest::read_tensor(one_hot), std::vector<double>(6, 7.0));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_OneHotOutOfRange,
    ::testing::Values(OutOfRangeRow{"AboveTheClasses", {0, 3}, "row 1", "is 3"},
                      OutOfRangeRow{"BelowZero", {-1, 0}, "row 0", "is -1"}),
    [](const ::testing::TestParamInfo<OutOfRangeRow>& info) { return info.param.name; });

/** @brief Tests that a label count other than the row count is refused with both shapes. */
TEST_F(Small_OneHot, AnotherLabelCountIsRefusedWithBothShapes)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({0, 1, 2}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "[3]",
                      "[4, 3]");
}

/** @brief Tests that labels of another element type are refused with both type names. */
TEST_F(Small_OneHot, LabelsOfAnotherElementTypeAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            allocate_tensor(backend.engine(), Shape{4}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "float32",
                      "int32");
}

/** @brief Tests that rows of an element type that is not a real are refused by name. */
TEST_F(Small_OneHot, RowsThatAreNotRealAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({0, 1}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot,
                            allocate_tensor(backend.engine(), Shape{2, 3}, DType::Int32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "int32",
                      "float32", "float64");
}
