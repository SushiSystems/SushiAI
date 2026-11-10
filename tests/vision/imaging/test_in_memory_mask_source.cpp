/**************************************************************************/
/* test_in_memory_mask_source.cpp                                         */
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
 * @file test_in_memory_mask_source.cpp
 * @brief Tests that InMemoryMaskSource keeps the promise of the MaskSource interface.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/in_memory_mask_source.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Builds a one-channel mask of @p height by @p width with every class @p value. */
    Image mask_of(std::int32_t height, std::int32_t width, std::uint8_t value)
    {
        Image mask(ImageSize{height, width}, 1);
        mask.fill(value);
        return mask;
    }
} // namespace

/** @brief Tests that add numbers the masks in order. */
TEST(Small_InMemoryMaskSource, AddNumbersTheMasksInOrder)
{
    InMemoryMaskSource source;

    EXPECT_EQ(source.count(), 0u);
    EXPECT_EQ(source.add(mask_of(2, 2, 1)), 0u);
    EXPECT_EQ(source.add(mask_of(3, 3, 2)), 1u);
    EXPECT_EQ(source.count(), 2u);
}

/** @brief Tests that load through the interface hands back the mask of the index. */
TEST(Small_InMemoryMaskSource, LoadHandsBackTheMaskOfTheIndex)
{
    InMemoryMaskSource source;
    static_cast<void>(source.add(mask_of(2, 2, 1)));
    static_cast<void>(source.add(mask_of(3, 4, 2)));
    const MaskSource& port = source;
    Image mask;

    SA_EXPECT_OK(port.load(1, mask));

    EXPECT_EQ(mask.size(), (ImageSize{3, 4}));
    EXPECT_EQ(mask.channels(), 1);
    EXPECT_EQ(mask.data()[0], 2);
}

/** @brief Tests that a scripted failure is returned for its index only. */
TEST(Small_InMemoryMaskSource, AScriptedFailureIsReturnedForItsIndexOnly)
{
    InMemoryMaskSource source;
    static_cast<void>(source.add(mask_of(2, 2, 1)));
    static_cast<void>(source.add(mask_of(2, 2, 2)));
    source.fail_load_of(1, Status::from_message(StatusCode::InvalidFile, []()
    {
        return std::string("mask 1 is truncated");
    }));
    Image mask;

    SA_EXPECT_OK(source.load(0, mask));
    SA_EXPECT_FAILURE(source.load(1, mask), StatusCode::InvalidFile, "truncated");
}

/** @brief Tests that an index outside the source stops the process. */
TEST(Medium_InMemoryMaskSourceFatal, AnIndexOutsideTheSourceStopsTheProcess)
{
    InMemoryMaskSource source;
    static_cast<void>(source.add(mask_of(2, 2, 1)));
    Image mask;

    SA_EXPECT_FATAL(static_cast<void>(source.load(5, mask)), "InMemoryMaskSource::load",
                    "index 5");
}
