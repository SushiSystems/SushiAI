/**************************************************************************/
/* test_decode_options.cpp                                                */
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
 * @file test_decode_options.cpp
 * @brief Tests the defaults and the storage of the options every image decoder takes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/vision/codecs/decode_options.hpp>

using namespace SushiAI::Vision::Codecs;

/** @brief Tests that default options keep the channels and the colours the file stores. */
TEST(Small_CodecsDecodeOptions, TheDefaultsKeepWhatTheFileStores)
{
    const DecodeOptions options;

    EXPECT_EQ(options.layout, ChannelLayout::Stored);
    EXPECT_EQ(options.palette, PaletteRead::Colors);
}

/** @brief Tests that both enumerations are stored in one unsigned byte. */
TEST(Small_CodecsDecodeOptions, BothEnumerationsFitInOneByte)
{
    EXPECT_TRUE((std::is_same_v<std::underlying_type_t<ChannelLayout>, std::uint8_t>));
    EXPECT_TRUE((std::is_same_v<std::underlying_type_t<PaletteRead>, std::uint8_t>));
}
