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
 * @file test_batch_inputs.cpp
 * @brief Tests the input writer: bytes divided by 255 and the images it refuses.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::PixelBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the size of every image of these tests. */
    constexpr ImageSize K_SIZE{1, 2};

    /** @brief Provides an engine, a two-file list and the samples of one batch. */
    class Unit_BatchInputs : public SaTest::EngineTest
    {
        protected:
            /** @brief Names the two files and makes one gray sample of K_SIZE per file. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                list.files = {"first.png", "crowded.png"};

                samples.resize(2);
                samples[0].image = Support::uniform_image(K_SIZE, 1, 51);
                samples[0].file = 0;
                samples[1].image = Support::uniform_image(K_SIZE, 1, 255);
                samples[1].file = 1;
            }

            /** @brief Returns the samples as the span a collator takes. */
            SushiRuntime::span<const FittedSample> span() const
            {
                return SushiRuntime::span<const FittedSample>(samples.data(), samples.size());
            }

            /** @brief Allocates a batch of two gray images of K_SIZE over some roles. */
            Data::Batch allocate(const std::vector<Data::TargetDescription>& targets)
            {
                const Data::InputDescription input{Shape{K_SIZE.height, K_SIZE.width, 1},
                                                   DType::Float32};
                Result<Data::Batch> batch = Data::allocate_batch(*engine, 2, input, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Holds the files the samples name. */
            SampleList list;

            /** @brief Holds the fitted samples of the batch. */
            std::vector<FittedSample> samples;
    };

    /** @brief Holds the absolute tolerance of a byte divided by 255 in float. */
    constexpr float K_SCALE_TOLERANCE = 1e-6f;

} // namespace

/** @brief Tests that the images are written row by row as bytes divided by 255. */
TEST_F(Unit_BatchInputs, InputsAreBytesDividedBy255)
{
    Data::Batch batch = allocate({});
    SA_EXPECT_OK(write_batch_inputs(span(), batch.inputs));

    const float* inputs = batch.inputs.data_as<float>();
    EXPECT_NEAR(inputs[0], 0.2f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[1], 0.2f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[2], 1.0f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[3], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that an image of another size is refused with its position in the batch. */
TEST_F(Unit_BatchInputs, AnImageOfAnotherSizeIsRefused)
{
    samples[1].image = Support::uniform_image(ImageSize{2, 2}, 1, 0);

    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(write_batch_inputs(span(), batch.inputs), StatusCode::Incompatible,
                      "write_batch_inputs", "image 1", "2 by 2");
}

/** @brief Tests that a tensor of another image count is refused. */
TEST_F(Unit_BatchInputs, ATensorOfAnotherImageCountIsRefused)
{
    samples.pop_back();

    Data::Batch batch = allocate({});
    SA_EXPECT_FAILURE(write_batch_inputs(span(), batch.inputs), StatusCode::Incompatible,
                      "write_batch_inputs", "holds 2 images");
}
