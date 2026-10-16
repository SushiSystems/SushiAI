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
