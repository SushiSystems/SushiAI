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
#include <SushiAI/config/input_normalise.hpp>
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

    /** @brief Holds the size of the image that holds each byte level once. */
    constexpr ImageSize K_LEVELS_SIZE{1, 256};

    /** @brief Returns one gray sample of K_LEVELS_SIZE whose pixel n is the level n. */
    std::vector<FittedSample> every_level_sample()
    {
        std::vector<FittedSample> samples(1);
        samples[0].image = Vision::Imaging::Image(K_LEVELS_SIZE, 1);
        for (std::size_t level = 0; level < 256; ++level)
            samples[0].image.data()[level] = static_cast<std::uint8_t>(level);
        return samples;
    }

    /**
     * @brief Expects value n of @p values to be the float quotient of n by 255, bit for bit.
     *
     * The oracle is the division the writer made before a normalisation could be stated.
     */
    void expect_each_level_divided_by_255(const std::vector<double>& values)
    {
        ASSERT_EQ(values.size(), 256u);
        for (std::size_t level = 0; level < 256; ++level)
            EXPECT_EQ(values[level], static_cast<double>(static_cast<float>(level) / 255.0f))
                << "level " << level;
    }

    /** @brief Provides a host engine to allocate the input tensor of a batch from. */
    class Small_BatchInputs : public ::testing::Test
    {
        protected:
            /** @brief Allocates the inputs of a batch of images of @p size and @p channels. */
            Data::Batch allocate(std::int64_t images, ImageSize size = K_SIZE,
                                 std::int64_t channels = 1)
            {
                const Data::InputDescription input{Shape{size.height, size.width, channels},
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

/**
 * @brief Tests that the default normalisation gives every byte level its quotient by 255, bit
 *        for bit; the oracle is the division the writer used before the normalisation existed.
 */
TEST_F(Small_BatchInputs, TheDefaultNormalisationIsTheByteDividedBy255ForEveryLevel)
{
    const std::vector<FittedSample> samples = every_level_sample();
    Data::Batch batch = allocate(1, K_LEVELS_SIZE);

    SA_EXPECT_OK(write_batch_inputs(span_of(samples), batch.inputs, Config::InputNormalise()));

    expect_each_level_divided_by_255(SaTest::read_tensor(batch.inputs));
}

/**
 * @brief Tests that one pixel of the bytes 2, 4 and 8 is written by the formula of the object, in
 *        the blue-first order. Every number is a power of two, so each value is exact.
 */
TEST_F(Small_BatchInputs, ABlueFirstPixelTakesTheScaleTheMeanAndTheDeviation)
{
    constexpr ImageSize pixel{1, 1};
    Config::InputNormalise normalise;
    normalise.channels = Config::ChannelOrder::Bgr;
    normalise.scale = 0.5;
    normalise.mean = {1.0, 0.5, 0.25};
    normalise.deviation = {2.0, 0.5, 0.25};
    std::vector<FittedSample> samples(1);
    samples[0].image = Vision::Imaging::Image(pixel, 3);
    samples[0].image.data()[0] = 2;
    samples[0].image.data()[1] = 4;
    samples[0].image.data()[2] = 8;
    Data::Batch batch = allocate(1, pixel, 3);

    SA_EXPECT_OK(write_batch_inputs(span_of(samples), batch.inputs, normalise));

    // Blue-first reads 8, 4, 2; halved 4, 2, 1; less the mean 3, 1.5, 0.75; over 1.5, 3, 3.
    const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
    ASSERT_EQ(inputs.size(), 3u);
    EXPECT_EQ(inputs[0], 1.5);
    EXPECT_EQ(inputs[1], 3.0);
    EXPECT_EQ(inputs[2], 3.0);
}

/** @brief Tests that the decoded order keeps each byte on its channel, with the same formula. */
TEST_F(Small_BatchInputs, ARedFirstPixelKeepsEachByteOnItsChannel)
{
    constexpr ImageSize pixel{1, 1};
    Config::InputNormalise normalise;
    normalise.scale = 0.5;
    normalise.mean = {1.0, 0.5, 0.25};
    normalise.deviation = {2.0, 0.5, 0.25};
    std::vector<FittedSample> samples(1);
    samples[0].image = Vision::Imaging::Image(pixel, 3);
    samples[0].image.data()[0] = 2;
    samples[0].image.data()[1] = 4;
    samples[0].image.data()[2] = 8;
    Data::Batch batch = allocate(1, pixel, 3);

    SA_EXPECT_OK(write_batch_inputs(span_of(samples), batch.inputs, normalise));

    // Half of 2, 4, 8 is 1, 2, 4; less the mean 0, 1.5, 3.75; over the deviation 0, 3, 15.
    const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
    ASSERT_EQ(inputs.size(), 3u);
    EXPECT_EQ(inputs[0], 0.0);
    EXPECT_EQ(inputs[1], 3.0);
    EXPECT_EQ(inputs[2], 15.0);
}

/** @brief Tests that a gray input refuses a normalisation that states colours. */
TEST_F(Small_BatchInputs, ANormalisationOfColoursIsRefusedForAGrayInput)
{
    const std::vector<FittedSample> samples = two_samples();
    Config::InputNormalise normalise;
    normalise.channels = Config::ChannelOrder::Bgr;
    Data::Batch batch = allocate(2);

    SA_EXPECT_FAILURE(write_batch_inputs(span_of(samples), batch.inputs, normalise),
                      StatusCode::Incompatible, "write_batch_inputs", "1 channels", "stated for 3");
}

/** @brief Tests that a scale alone is allowed on a gray input and replaces the divisor. */
TEST_F(Small_BatchInputs, AScaleAloneAppliesToAGrayInput)
{
    const std::vector<FittedSample> samples = two_samples();
    Config::InputNormalise normalise;
    normalise.scale = 0.5;
    Data::Batch batch = allocate(2);

    SA_EXPECT_OK(write_batch_inputs(span_of(samples), batch.inputs, normalise));

    // The levels 51 and 255 halved are 25.5 and 127.5, both exact in float.
    const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
    ASSERT_EQ(inputs.size(), 4u);
    EXPECT_EQ(inputs[0], 25.5);
    EXPECT_EQ(inputs[2], 127.5);
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
