/**************************************************************************/
/* test_imported_model.cpp                                                */
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
 * @file test_imported_model.cpp
 * @brief Tests the payload size a record of an imported tensor states.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds an element type and the bytes a [4, 3] record of it states. */
    struct PayloadRow
    {
        std::string name;        /**< The row's name. */
        DType dtype;             /**< The element type of the record. */
        std::size_t bytes;       /**< The payload bytes twelve elements of it take. */
    };

    /** @brief Runs the payload size of a record over each element type. */
    class Small_ImportedModel : public ::testing::TestWithParam<PayloadRow>
    {
    };
} // namespace

/** @brief Tests that a record states its payload as its element count times its element size. */
TEST_P(Small_ImportedModel, ATensorHoldsTheBytesItsRecordStates)
{
    IO::TensorRecord record;
    record.name = "0.weight";
    record.dtype = GetParam().dtype;
    record.shape = Shape{4, 3};

    EXPECT_EQ(record.payload_bytes(), GetParam().bytes);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImportedModel,
    ::testing::Values(PayloadRow{"Float32", DType::Float32, 4 * 3 * 4},
                      PayloadRow{"Int64", DType::Int64, 4 * 3 * 8}),
    [](const ::testing::TestParamInfo<PayloadRow>& info) { return info.param.name; });
