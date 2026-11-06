/**************************************************************************/
/* test_in_memory_sample_source.cpp                                       */
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
 * @file test_in_memory_sample_source.cpp
 * @brief Tests that InMemorySampleSource keeps the promise of the SampleSource interface.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/in_memory_sample_source.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Builds a one-channel image of @p height by @p width with every byte @p value. */
    Image filled(std::int32_t height, std::int32_t width, std::uint8_t value)
    {
        Image image(ImageSize{height, width}, 1);
        image.fill(value);
        return image;
    }

    /** @brief Builds the labels of one object of class @p object_class. */
    FileLabels one_object(std::int32_t object_class)
    {
        return FileLabels({object_class}, {NormalizedBox{0.5f, 0.5f, 0.25f, 0.25f}});
    }
} // namespace

/** @brief Tests that an empty source holds no image. */
TEST(Small_InMemorySampleSource, AnEmptySourceHoldsNoImage)
{
    const InMemorySampleSource source;

    EXPECT_EQ(source.size(), 0u);
}

/** @brief Tests that add numbers the images in order and size counts them. */
TEST(Small_InMemorySampleSource, AddNumbersTheImagesInOrder)
{
    InMemorySampleSource source;

    EXPECT_EQ(source.add(filled(2, 3, 7)), 0u);
    EXPECT_EQ(source.add(filled(4, 4, 9)), 1u);

    EXPECT_EQ(source.size(), 2u);
}

/** @brief Tests that load through the interface hands back the image and labels of the index. */
TEST(Small_InMemorySampleSource, LoadHandsBackTheImageAndLabelsOfTheIndex)
{
    InMemorySampleSource source;
    static_cast<void>(source.add(filled(2, 3, 7), one_object(1)));
    static_cast<void>(source.add(filled(4, 5, 9), one_object(3)));
    const SampleSource& port = source;
    Image image;
    FileLabels labels;

    SA_EXPECT_OK(port.load(1, image, labels));

    EXPECT_EQ(image.size(), (ImageSize{4, 5}));
    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(image.data()[0], 9);
    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 3);
}

/** @brief Tests that a scripted failure is returned for its index only. */
TEST(Small_InMemorySampleSource, AScriptedFailureIsReturnedForItsIndexOnly)
{
    InMemorySampleSource source;
    static_cast<void>(source.add(filled(2, 2, 1)));
    static_cast<void>(source.add(filled(2, 2, 2)));
    source.fail_load_of(0, Status::from_message(StatusCode::FileNotFound, []()
    {
        return std::string("a.png is missing");
    }));
    Image image;
    FileLabels labels;

    SA_EXPECT_FAILURE(source.load(0, image, labels), StatusCode::FileNotFound, "a.png");
    SA_EXPECT_OK(source.load(1, image, labels));
    EXPECT_EQ(image.data()[0], 2);
}

/** @brief Tests that a success removes a scripted failure. */
TEST(Small_InMemorySampleSource, ASuccessRemovesAScriptedFailure)
{
    InMemorySampleSource source;
    static_cast<void>(source.add(filled(2, 2, 1)));
    source.fail_load_of(0, Status::from_message(StatusCode::InvalidFile, []()
    {
        return std::string("broken");
    }));
    source.fail_load_of(0, Status());
    Image image;
    FileLabels labels;

    SA_EXPECT_OK(source.load(0, image, labels));
}

/** @brief Tests that an index outside the source stops the process, as the interface states. */
TEST(Small_InMemorySampleSource, AnIndexOutsideTheSourceStopsTheProcess)
{
    InMemorySampleSource source;
    static_cast<void>(source.add(filled(2, 2, 1)));
    Image image;
    FileLabels labels;

    SA_EXPECT_FATAL(static_cast<void>(source.load(1, image, labels)),
                    "InMemorySampleSource::load", "index 1");
}
