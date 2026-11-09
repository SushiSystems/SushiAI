/**************************************************************************/
/* test_batch_inputs.cpp                                                  */
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
 * @file test_batch_inputs.cpp
 * @brief Tests the input writer on a host engine: bytes divided by 255 and the images it refuses.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every image of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /**
     * @brief Holds the absolute tolerance of a byte divided by 255 in float: 32 times the float
     *        unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_SCALE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Returns two gray samples of K_SIZE, the first of level 51 and the second of 255. */
    std::vector<FittedSample> two_samples()
    {
        std::vector<FittedSample> samples(2);
        samples[0].image = Support::uniform_image(K_SIZE, 1, 51);
        samples[1].image = Support::uniform_image(K_SIZE, 1, 255);
        return samples;
    }

    /** @brief Returns the samples as the span the writer takes. */
    SushiRuntime::span<const FittedSample> span_of(const std::vector<FittedSample>& samples)
    {
        return SushiRuntime::span<const FittedSample>(samples.data(), samples.size());
    }

    /** @brief Provides a host engine to allocate the input tensor of a batch from. */
    class Small_BatchInputs : public ::testing::Test
    {
        protected:
            /** @brief Allocates the inputs of a batch of some gray images of K_SIZE. */
            Data::Batch allocate(std::int64_t images)
            {
                const Data::InputDescription input{Shape{K_SIZE.height, K_SIZE.width, 1},
                                                   DType::Float32};
                Result<Data::Batch> batch =
                    Data::allocate_batch(backend.engine(), images, input, {});
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Holds the host engine the tensors are allocated from. */
            HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that the images are written row by row as bytes divided by 255. */
TEST_F(Small_BatchInputs, InputsAreBytesDividedBy255)
{
    const std::vector<FittedSample> samples = two_samples();
    Data::Batch batch = allocate(2);

    SA_EXPECT_OK(write_batch_inputs(span_of(samples), batch.inputs));

    const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
    ASSERT_EQ(inputs.size(), 4u);
    EXPECT_NEAR(inputs[0], 0.2, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[1], 0.2, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[2], 1.0, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[3], 1.0, K_SCALE_TOLERANCE);
}

/** @brief Tests that an image of another size is refused with its position in the batch. */
TEST_F(Small_BatchInputs, AnImageOfAnotherSizeIsRefused)
{
    std::vector<FittedSample> samples = two_samples();
    samples[1].image = Support::uniform_image(ImageSize{2, 2}, 1, 0);
    Data::Batch batch = allocate(2);

    SA_EXPECT_FAILURE(write_batch_inputs(span_of(samples), batch.inputs),
                      StatusCode::Incompatible, "write_batch_inputs", "image 1", "2 by 2");
}

/** @brief Tests that a tensor of another image count is refused. */
TEST_F(Small_BatchInputs, ATensorOfAnotherImageCountIsRefused)
{
    const std::vector<FittedSample> one_sample(1, two_samples()[0]);
    Data::Batch batch = allocate(2);

    SA_EXPECT_FAILURE(write_batch_inputs(span_of(one_sample), batch.inputs),
                      StatusCode::Incompatible, "write_batch_inputs", "holds 2 images");
}
