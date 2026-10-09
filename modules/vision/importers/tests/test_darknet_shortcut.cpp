/**************************************************************************/
/* test_darknet_shortcut.cpp                                              */
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
 * @file test_darknet_shortcut.cpp
 * @brief Tests the add row a [shortcut] section translates to, and its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <vision_support/darknet_forward.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;

namespace
{
    using Test::sources_of_row;
    using Test::translate_darknet_text;
    using Vision::Importers::DarknetModel;

    /** @brief Holds the first section of an 8 by 8 network with 3 input channels. */
    constexpr std::string_view K_NETWORK = "[net]\nwidth=8\nheight=8\nchannels=3\n";

    /** @brief Holds layers 0 and 1: two 1 by 1 linear convolutions of 4 filters, lines 5 to 16. */
    constexpr std::string_view K_TWO_CONVOLUTIONS =
        "[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n"
        "[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n";

    /** @brief Holds one 18-filter 1 by 1 convolution and the [yolo] head that reads it. */
    constexpr std::string_view K_HEAD_TAIL =
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=0\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n";

} // namespace

TEST(Unit_DarknetShortcut, NegativeFromAddsTheLayerBeforeAndTheNamedLayer)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\n").append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_darknet_text(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_TRUE(model.spec.layers[2].attributes.keys().empty());
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1, 0}));
}

TEST(Unit_DarknetShortcut, AbsoluteFromAddsTheSameTwoRows)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=0\nactivation=linear\n")
                                 .append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate_darknet_text(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1, 0}));
}

TEST(Unit_DarknetShortcut, FromAboveTheShortcutIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=2\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "line 18",
                      "only layers 0 to 1");
}

TEST(Unit_DarknetShortcut, DifferentChannelCountsAreRefusedNamingItsLine)
{
    const std::string text =
        std::string(K_NETWORK)
            .append("[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[convolutional]\nfilters=5\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[shortcut]\nfrom=0\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "line 18",
                      "share their shape");
}

TEST(Unit_DarknetShortcut, NonLinearActivationIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nactivation=leaky\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::Unsupported, "line 19", "'leaky'");
}

TEST(Unit_DarknetShortcut, WeightsTypeIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nweights_type=per_feature\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::Unsupported, "line 19",
                      "weights_type");
}

TEST(Unit_DarknetShortcut, MissingFromIsRefusedNamingTheSection)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nactivation=linear\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription, "no 'from'");
}

TEST(Unit_DarknetShortcut, MoreThanOneFromIsRefused)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-1,-2\n");

    SA_EXPECT_FAILURE(translate_darknet_text(text), StatusCode::InvalidDescription,
                      "names one layer");
}
