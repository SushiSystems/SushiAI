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
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>

using namespace SushiAI;
using Vision::Training::validate_pixel_labels;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the class count, C. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Provides an engine and the INT32 masks the cases check. */
    class Medium_VisionTrainingPixelLabels : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns an INT32 tensor of @p shape holding @p values. */
            SushiBLAS::Tensor make_ids(const Shape& shape, const std::vector<std::int32_t>& values)
            {
                Result<SushiBLAS::Tensor> made = allocate_tensor(*engine, shape, DType::Int32);
                EXPECT_TRUE(made.status().is_ok());

                std::int32_t* const ids = made.value().data_as<std::int32_t>();
                for (std::size_t i = 0; i < values.size(); ++i)
                    ids[i] = values[i];

                return std::move(made).value();
            }
    };
} // namespace

/** @brief Tests that an ignore index of -1 passes and that 255 is then a label out of bounds. */
TEST_F(Medium_VisionTrainingPixelLabels, TheCallersIgnoreIndexReplacesTheDefault)
{
    const SushiBLAS::Tensor minus_one = make_ids(Shape{4}, {0, -1, 2, 1});
    const SushiBLAS::Tensor byte_void = make_ids(Shape{4}, {0, 255, 2, 1});

    SA_EXPECT_OK(validate_pixel_labels(minus_one, 4, K_CLASSES, -1));
    SA_EXPECT_FAILURE(validate_pixel_labels(byte_void, 4, K_CLASSES, -1),
                      StatusCode::Incompatible, "holds 255 at pixel 1");
}

/** @brief Tests that a mask of rank three is checked by its element count alone. */
TEST_F(Medium_VisionTrainingPixelLabels, AMaskOfRankThreeIsCheckedByItsElementCount)
{
    const SushiBLAS::Tensor mask = make_ids(Shape{2, 2, 2}, {0, 1, 2, 0, 1, 2, 255, 0});

    SA_EXPECT_OK(validate_pixel_labels(mask, 8, K_CLASSES, 255));
    SA_EXPECT_FAILURE(validate_pixel_labels(mask, 4, K_CLASSES, 255),
                      StatusCode::Incompatible, "expects 4");
}
