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
#include <common/test_status.hpp>

using namespace SushiAI;

namespace
{
    using Vision::Importers::DarknetConfig;
    using Vision::Importers::DarknetModel;
    using Vision::Importers::DarknetTranslation;
    using Vision::Importers::parse_darknet_config;
    using Vision::Importers::translate_darknet_model;

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

    /** @brief Returns the model of @p text, translated in the separate normalisation. */
    Result<DarknetModel> translate(std::string_view text)
    {
        SA_TRY_ASSIGN(const DarknetConfig config, parse_darknet_config(text, "test.cfg"));
        return translate_darknet_model(config, DarknetTranslation{});
    }

    /** @brief Returns the sources row @p row reads, as the row indices they name. */
    std::vector<std::int64_t> sources_of(const DarknetModel& model, std::size_t row)
    {
        std::vector<std::int64_t> sources;
        for (const Config::SourceSpec& source : model.spec.layers.at(row).from)
            sources.push_back(source.node);

        return sources;
    }

} // namespace

TEST(Unit_DarknetShortcut, NegativeFromAddsTheLayerBeforeAndTheNamedLayer)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\n").append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_TRUE(model.spec.layers[2].attributes.keys().empty());
    EXPECT_EQ(sources_of(model, 2), (std::vector<std::int64_t>{1, 0}));
}

TEST(Unit_DarknetShortcut, AbsoluteFromAddsTheSameTwoRows)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=0\nactivation=linear\n")
                                 .append(K_HEAD_TAIL);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetModel model, translate(text));

    ASSERT_EQ(model.spec.layers.size(), 4u);
    EXPECT_EQ(model.spec.layers[2].type, "add");
    EXPECT_EQ(sources_of(model, 2), (std::vector<std::int64_t>{1, 0}));
}

TEST(Unit_DarknetShortcut, FromAboveTheShortcutIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=2\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::InvalidDescription, "line 18",
                      "only layers 0 to 1");
}

TEST(Unit_DarknetShortcut, DifferentChannelCountsAreRefusedNamingItsLine)
{
    const std::string text =
        std::string(K_NETWORK)
            .append("[convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[convolutional]\nfilters=5\nsize=1\nstride=1\npad=0\nactivation=linear\n")
            .append("[shortcut]\nfrom=0\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::InvalidDescription, "line 18",
                      "share their shape");
}

TEST(Unit_DarknetShortcut, NonLinearActivationIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nactivation=leaky\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::Unsupported, "line 19", "'leaky'");
}

TEST(Unit_DarknetShortcut, WeightsTypeIsRefusedNamingItsLine)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-2\nweights_type=per_feature\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::Unsupported, "line 19", "weights_type");
}

TEST(Unit_DarknetShortcut, MissingFromIsRefusedNamingTheSection)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nactivation=linear\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::InvalidDescription, "no 'from'");
}

TEST(Unit_DarknetShortcut, MoreThanOneFromIsRefused)
{
    const std::string text = std::string(K_NETWORK).append(K_TWO_CONVOLUTIONS)
                                 .append("[shortcut]\nfrom=-1,-2\n");

    SA_EXPECT_FAILURE(translate(text), StatusCode::InvalidDescription, "names one layer");
}
