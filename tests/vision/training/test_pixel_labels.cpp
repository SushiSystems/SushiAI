/**************************************************************************/
/* test_pixel_labels.cpp                                                  */
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
 * @file test_pixel_labels.cpp
 * @brief Tests the label check with a caller's ignore index and a mask of rank three.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <utility>
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
#include <SushiAI/vision/training/pixel_labels.hpp>

using namespace SushiAI;
using Vision::Training::validate_pixel_label_values;
using Vision::Training::validate_pixel_labels;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the class count, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Provides a host engine and the INT32 masks the cases check. */
    class Small_VisionTrainingPixelLabelsMask : public ::testing::Test
    {
        protected:
            /** @brief Returns an INT32 tensor of @p shape holding @p values. */
            SushiBLAS::Tensor make_ids(const Shape& shape, const std::vector<std::int32_t>& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(backend.engine(), shape, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());

                std::int32_t* const ids = made.value().data_as<std::int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    ids[i] = values[i];

                return std::move(made).value();
            }

            /** @brief The backend whose engine holds the host storage of every mask. */
            SushiAI::HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that an ignore index of -1 passes and that 255 is then a label out of bounds. */
TEST(Small_VisionTrainingPixelLabels, TheCallersIgnoreIndexReplacesTheDefault)
{
    const std::vector<std::int32_t> minus_one{0, -1, 2, 1};
    const std::vector<std::int32_t> byte_void{0, 255, 2, 1};

    SA_EXPECT_OK(validate_pixel_label_values(minus_one.data(), 4, K_CLASSES, -1));
    SA_EXPECT_FAILURE(validate_pixel_label_values(byte_void.data(), 4, K_CLASSES, -1),
                      StatusCode::Incompatible, "holds 255 at pixel 1");
}

/** @brief Tests that a mask of rank three is checked by its element count alone. */
TEST_F(Small_VisionTrainingPixelLabelsMask, AMaskOfRankThreeIsCheckedByItsElementCount)
{
    const SushiBLAS::Tensor mask = make_ids(Shape{2, 2, 2}, {0, 1, 2, 0, 1, 2, 255, 0});

    SA_EXPECT_OK(validate_pixel_labels(mask, 8, K_CLASSES, 255));
    SA_EXPECT_FAILURE(validate_pixel_labels(mask, 4, K_CLASSES, 255),
                      StatusCode::Incompatible, "expects 4");
}
