/**************************************************************************/
/* test_darknet_route_groups.cpp                                          */
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
 * @file test_darknet_route_groups.cpp
 * @brief Tests the slice rows, the join and the refusals of a grouped [route] translation.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

namespace
{
    using Test::sources_of_row;
    using Test::translate_darknet_text;
    using Vision::Importers::DarknetModel;

    /** @brief Holds an 8 by 8 network of 3 channels and one 64-filter 1 by 1 convolution. */
    constexpr std::string_view K_WIDE_NETWORK =
        "[net]\nwidth=8\nheight=8\nchannels=3\n"
        "[convolutional]\nfilters=64\nsize=1\nstride=1\npad=0\nactivation=linear\n";

    /** @brief Holds one 18-filter 1 by 1 convolution and the [yolo] head that reads it. */
    constexpr std::string_view K_HEAD_TAIL =
        "[convolutional]\nfilters=18\nsize=1\nstride=1\npad=0\nactivation=linear\n"
        "[yolo]\nmask=0,1,2\nanchors=4,8, 8,16, 16,24\nclasses=1\nnum=3\n";

    /** @brief Returns the type of row @p row of @p model. */
    const std::string& type_of(const DarknetModel& model, std::int64_t row)
    {
        return model.spec.layers.at(static_cast<std::size_t>(row)).type;
    }

    /** @brief Returns the integer attribute @p key of row @p row of @p model. */
    std::int64_t int_attribute(const DarknetModel& model, std::int64_t row, std::string_view key)
    {
        const Config::Value& attributes =
            model.spec.layers.at(static_cast<std::size_t>(row)).attributes;
        return attributes.at(key, "test").as_int("test");
    }

} // namespace

/**
 * @brief Tests that a grouped route of one source is one slice row of the second half.
 *
 * The 64 channels of layer 1 split into halves of 32; group 1 is channels 32 to 63. Layer 1 is
 * the row 0 output, so the slice is row 1, and the head convolution reads it.
 */
TEST(Unit_DarknetRouteGroups, OneSourceGroupIsOneSliceRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=2\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 3u);
    EXPECT_EQ(type_of(model, 0), "conv2d");
    EXPECT_EQ(type_of(model, 1), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 1), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(int_attribute(model, 1, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 1, "count"), 32);
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1}));

    ASSERT_EQ(model.convolutions.size(), 2u);
    EXPECT_EQ(model.convolutions[1].channels, 32);
}

/**
 * @brief Tests that each source of a grouped route is sliced before one concat joins them.
 *
 * Layer 2 lists -1 and -2, so its first part is layer 1 (row 1) and its second is layer 0
 * (row 0). Each contributes its group 1 of 32 channels, in list order: rows 2 and 3, then the
 * concat of row 4 over both. The head convolution reads 64 channels.
 */
TEST(Unit_DarknetRouteGroups, EverySourceIsSlicedBeforeTheJoin)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[convolutional]\nfilters=64\nsize=1\nstride=1\npad=0\n"
                              "activation=linear\n")
                                   .append("[route]\nlayers=-1,-2\ngroups=2\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 6u);
    EXPECT_EQ(type_of(model, 2), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 2), (std::vector<std::int64_t>{1}));
    EXPECT_EQ(int_attribute(model, 2, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 2, "count"), 32);

    EXPECT_EQ(type_of(model, 3), "slice_channels");
    EXPECT_EQ(sources_of_row(model, 3), (std::vector<std::int64_t>{0}));
    EXPECT_EQ(int_attribute(model, 3, "offset"), 32);
    EXPECT_EQ(int_attribute(model, 3, "count"), 32);

    EXPECT_EQ(type_of(model, 4), "concat");
    EXPECT_EQ(sources_of_row(model, 4), (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(sources_of_row(model, 5), (std::vector<std::int64_t>{4}));

    ASSERT_EQ(model.convolutions.size(), 3u);
    EXPECT_EQ(model.convolutions[2].channels, 64);
}

/**
 * @brief Tests that an ungrouped route of one source adds no row and keeps every channel.
 */
TEST(Unit_DarknetRouteGroups, UngroupedSourceAddsNoRow)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetModel model,
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\n")
                                   .append(K_HEAD_TAIL)));

    ASSERT_EQ(model.spec.layers.size(), 2u);
    EXPECT_EQ(type_of(model, 1), "conv2d");
    EXPECT_EQ(sources_of_row(model, 1), (std::vector<std::int64_t>{0}));

    ASSERT_EQ(model.convolutions.size(), 2u);
    EXPECT_EQ(model.convolutions[1].channels, 64);
}

/**
 * @brief Tests that a group count that does not divide the channels, or a group index out of
 *        range, is refused naming its line.
 *
 * The [route] section starts on line 11 and its layers line is line 12, which names the channel
 * count. A group index is refused on the section line.
 */
TEST(Unit_DarknetRouteGroups, UnequalSplitsAndGroupsOutOfRangeAreRefused)
{
    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=3\ngroup_id=0\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 12", "groups 3", "64 channels");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=2\ngroup_id=2\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "group_id 2");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroups=0\ngroup_id=0\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "groups 0");

    SA_EXPECT_FAILURE(
        translate_darknet_text(std::string(K_WIDE_NETWORK)
                                   .append("[route]\nlayers=-1\ngroup_id=1\n")
                                   .append(K_HEAD_TAIL)),
        StatusCode::InvalidDescription, "line 11", "group_id 1");
}
