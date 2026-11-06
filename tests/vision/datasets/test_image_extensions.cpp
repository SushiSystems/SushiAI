/**************************************************************************/
/* test_image_extensions.cpp                                              */
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
 * @file test_image_extensions.cpp
 * @brief Tests that an image extension is judged by name, and only decoded formats pass.
 * @author Mustafa Garip
 */

#include <filesystem>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/image_extensions.hpp>

using SushiAI::Vision::Datasets::has_image_extension;

/** @brief Tests that image formats the codecs do not decode are refused. */
TEST(Small_DatasetsImageExtensions, FormatsTheCodecsDoNotDecodeAreRefused)
{
    EXPECT_FALSE(has_image_extension("a.gif"));
    EXPECT_FALSE(has_image_extension("a.bmp"));
    EXPECT_FALSE(has_image_extension("a.tiff"));
    EXPECT_FALSE(has_image_extension("a.webp"));
}

/** @brief Tests that the path is judged by its name, so an absent file is still an image. */
TEST(Small_DatasetsImageExtensions, TheFileIsNotOpened)
{
    const std::filesystem::path absent =
        std::filesystem::path("sushiai_no_such_directory") / "no_such_image.png";

    ASSERT_FALSE(std::filesystem::exists(absent));
    EXPECT_TRUE(has_image_extension(absent));
}
