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
#include <cctype>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/vision/datasets/image_extensions.hpp>

using SushiAI::Vision::Datasets::has_image_extension;

namespace
{
    /** @brief Runs has_image_extension over the name of each format the codecs do not decode. */
    class Small_DatasetsImageExtensionsUndecoded : public ::testing::TestWithParam<std::string>
    {
    };

    /** @brief Runs has_image_extension over each name an image extension is taken from. */
    class Small_DatasetsImageExtensionsTaken : public ::testing::TestWithParam<std::string>
    {
    };

    /** @brief Runs has_image_extension over each name that holds no whole image extension. */
    class Small_DatasetsImageExtensionsNoExtension : public ::testing::TestWithParam<std::string>
    {
    };

    /** @brief Names a test after a path, with every character that is no letter or digit as _. */
    std::string path_row_name(const ::testing::TestParamInfo<std::string>& info)
    {
        std::string name = info.param;
        for (char& letter : name)
        {
            if (!std::isalnum(static_cast<unsigned char>(letter)))
                letter = '_';
        }

        return name;
    }

    /** @brief Names a test after the extension of its row. */
    std::string extension_row_name(const ::testing::TestParamInfo<std::string>& info)
    {
        return info.param.substr(info.param.find('.') + 1);
    }
} // namespace

/** @brief Tests that an image format the codecs do not decode is refused. */
TEST_P(Small_DatasetsImageExtensionsUndecoded, FormatsTheCodecsDoNotDecodeAreRefused)
{
    EXPECT_FALSE(has_image_extension(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsImageExtensionsUndecoded,
                         ::testing::Values("a.gif", "a.bmp", "a.tiff", "a.webp"),
                         extension_row_name);

/** @brief Tests that an image extension is matched without regard to case, at any depth. */
TEST_P(Small_DatasetsImageExtensionsTaken, ImageExtensionsIgnoreCase)
{
    EXPECT_TRUE(has_image_extension(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsImageExtensionsTaken,
                         ::testing::Values("a.jpg", "a.JPEG", "dir/a.PNG", "a.saim"),
                         path_row_name);

/** @brief Tests that a name with another, a double or no extension is refused. */
TEST_P(Small_DatasetsImageExtensionsNoExtension, ANameWithoutAnImageExtensionIsRefused)
{
    EXPECT_FALSE(has_image_extension(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DatasetsImageExtensionsNoExtension,
                         ::testing::Values("a.txt", "a.jpg.bak", "jpg"), path_row_name);

/** @brief Tests that the path is judged by its name alone, so no file need exist. */
TEST(Small_DatasetsImageExtensions, TheFileIsNotOpened)
{
    const std::filesystem::path absent =
        std::filesystem::path("sushiai_no_such_directory") / "no_such_image.png";

    EXPECT_TRUE(has_image_extension(absent));
}
