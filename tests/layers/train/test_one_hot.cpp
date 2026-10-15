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
 * @file test_one_hot.cpp
 * @brief Tests fill_one_hot: the rows it writes, both element types, and what it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/one_hot.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the label and row tensors are allocated on. */
    class Medium_OneHot : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns an int32 tensor of one extent that holds @p values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> labels_of(const std::vector<int32_t>& values)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor labels,
                              allocate_tensor(*engine,
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
                              allocate_tensor(*engine, Shape{rows, classes}, dtype));
                SaTest::write_tensor(
                    one_hot, std::vector<double>(static_cast<std::size_t>(rows * classes), 7.0));

                return one_hot;
            }
    };

    /** @brief Holds the rows the labels 2, 0, 1, 2 select among three classes. */
    const std::vector<double> K_EXPECTED_ROWS = {
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
} // namespace

/** @brief Tests that every row holds 1 in its label's column and 0 elsewhere. */
TEST_F(Medium_OneHot, EveryRowHoldsOneInItsLabelsColumn)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({2, 0, 1, 2}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float32));

    SA_EXPECT_OK(Train::fill_one_hot(labels, one_hot));

    EXPECT_EQ(SaTest::read_tensor(one_hot), K_EXPECTED_ROWS);
}

/** @brief Tests that a float64 leaf receives the same rows as a float32 one. */
TEST_F(Medium_OneHot, AFloat64LeafReceivesTheSameRows)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({2, 0, 1, 2}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float64));

    SA_EXPECT_OK(Train::fill_one_hot(labels, one_hot));

    EXPECT_EQ(SaTest::read_tensor(one_hot), K_EXPECTED_ROWS);
}

/** @brief Tests that a second fill replaces the rows of the first. */
TEST_F(Medium_OneHot, ASecondFillReplacesTheFirst)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor first, labels_of({0, 0}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor second, labels_of({1, 0}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(2, 2, DType::Float32));

    SA_EXPECT_OK(Train::fill_one_hot(first, one_hot));
    SA_EXPECT_OK(Train::fill_one_hot(second, one_hot));

    EXPECT_EQ(SaTest::read_tensor(one_hot), (std::vector<double>{0.0, 1.0, 1.0, 0.0}));
}

/** @brief Tests that a label outside the classes is named and no row is written. */
TEST_F(Medium_OneHot, ALabelOutsideTheClassesIsRefusedAndNothingIsWritten)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor above, labels_of({0, 3}));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor below, labels_of({-1, 0}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(2, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(above, one_hot), StatusCode::Incompatible, "row 1",
                      "is 3", "3 classes");
    SA_EXPECT_FAILURE(Train::fill_one_hot(below, one_hot), StatusCode::Incompatible, "row 0",
                      "is -1", "3 classes");

    EXPECT_EQ(SaTest::read_tensor(one_hot), std::vector<double>(6, 7.0));
}

/** @brief Tests that a label count other than the row count is refused with both shapes. */
TEST_F(Medium_OneHot, AnotherLabelCountIsRefusedWithBothShapes)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({0, 1, 2}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "[3]",
                      "[4, 3]");
}

/** @brief Tests that labels of another element type are refused with both type names. */
TEST_F(Medium_OneHot, LabelsOfAnotherElementTypeAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels,
                            allocate_tensor(*engine, Shape{4}, DType::Float32));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot, stale_rows(4, 3, DType::Float32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "float32",
                      "int32");
}

/** @brief Tests that rows of an element type that is not a real are refused by name. */
TEST_F(Medium_OneHot, RowsThatAreNotRealAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor labels, labels_of({0, 1}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor one_hot,
                            allocate_tensor(*engine, Shape{2, 3}, DType::Int32));

    SA_EXPECT_FAILURE(Train::fill_one_hot(labels, one_hot), StatusCode::Incompatible, "int32",
                      "float32", "float64");
}
