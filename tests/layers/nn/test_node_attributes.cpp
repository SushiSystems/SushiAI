/**************************************************************************/
/* test_node_attributes.cpp                                               */
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
 * @file test_node_attributes.cpp
 * @brief Tests the status-returning readers a node factory reads a model-file row with.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_attributes.hpp>

using namespace SushiAI;
namespace NNDetail = SushiAI::NN::Detail;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Names the row in every diagnostic of this file. */
    constexpr std::string_view K_CONTEXT = "row 2 'conv'";

    /** @brief Bounds the error of a real a row states and a double holds as read. */
    constexpr double K_REAL_TOLERANCE = 0.0;

    /** @brief Parses @p text as the attributes of one row. */
    Config::Value row(std::string_view text)
    {
        return require_value(Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that list_names joins the names with a comma and a space, and none as empty. */
TEST(Small_NodeAttributes, ListNamesJoinsTheNamesWithCommas)
{
    constexpr std::string_view K_NAMES[] = {"filters", "size", "stride"};

    EXPECT_EQ(NNDetail::list_names(NNDetail::KeySpan(K_NAMES)), "filters, size, stride");
    EXPECT_EQ(NNDetail::list_names(NNDetail::KeySpan()), "");
}

/** @brief Tests that read_feature_map takes one rank-3 shape and refuses another count or rank. */
TEST(Small_NodeAttributes, ReadFeatureMapTakesOneRankThreeShape)
{
    const std::vector<Shape> one{Shape{4, 4, 3}};
    const std::vector<Shape> two{Shape{4, 4, 3}, Shape{4, 4, 3}};
    const std::vector<Shape> flat{Shape{12}};

    SA_ASSERT_OK_AND_ASSIGN(const Shape map,
                            NNDetail::read_feature_map(NN::ShapeSpan(one.data(), one.size()),
                                                     K_CONTEXT));
    EXPECT_EQ(map, (Shape{4, 4, 3}));
    SA_EXPECT_FAILURE(NNDetail::read_feature_map(NN::ShapeSpan(two.data(), two.size()), K_CONTEXT),
                      StatusCode::InvalidDescription, K_CONTEXT, "reads one input", "given 2");
    SA_EXPECT_FAILURE(NNDetail::read_feature_map(NN::ShapeSpan(flat.data(), flat.size()),
                                               K_CONTEXT),
                      StatusCode::InvalidDescription, "[12]", "rank-3");
}

/** @brief Tests that an unknown key is refused by name, with the keys the row accepts. */
TEST(Small_NodeAttributes, RejectUnknownKeysNamesTheKeyAndTheAcceptedOnes)
{
    constexpr std::string_view K_ALLOWED[] = {"filters", "size"};

    SA_EXPECT_OK(NNDetail::reject_unknown_keys(row(R"({"filters": 8})"),
                                             NNDetail::KeySpan(K_ALLOWED), K_CONTEXT));
    SA_EXPECT_FAILURE(NNDetail::reject_unknown_keys(row(R"({"filters": 8, "pad": 1})"),
                                                  NNDetail::KeySpan(K_ALLOWED), K_CONTEXT),
                      StatusCode::InvalidDescription, K_CONTEXT, "'pad'", "filters, size");
}

/** @brief Tests that a positive whole number is read, and an absent, real or zero one refused. */
TEST(Small_NodeAttributes, ReadPositiveIntTakesOnlyAWholeNumberAboveZero)
{
    SA_ASSERT_OK_AND_ASSIGN(const int64_t filters,
                            NNDetail::read_positive_int(row(R"({"filters": 8})"), "filters",
                                                      K_CONTEXT));
    EXPECT_EQ(filters, 8);

    SA_EXPECT_FAILURE(NNDetail::read_positive_int(row("{}"), "filters", K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "absent");
    SA_EXPECT_FAILURE(NNDetail::read_positive_int(row(R"({"filters": 1.5})"), "filters", K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "whole number");
    SA_EXPECT_FAILURE(NNDetail::read_positive_int(row(R"({"filters": 0})"), "filters", K_CONTEXT),
                      StatusCode::InvalidDescription, "'filters'", "positive");
}

/** @brief Tests that the defaulting readers fall back on an absent key only. */
TEST(Small_NodeAttributes, TheDefaultingReadersFallBackOnAnAbsentKeyOnly)
{
    SA_ASSERT_OK_AND_ASSIGN(const int64_t stride,
                            NNDetail::read_positive_int_or(row("{}"), "stride", K_CONTEXT, 1));
    EXPECT_EQ(stride, 1);
    SA_EXPECT_FAILURE(NNDetail::read_positive_int_or(row(R"({"stride": -2})"), "stride", K_CONTEXT,
                                                   1),
                      StatusCode::InvalidDescription, "'stride'", "positive");

    SA_ASSERT_OK_AND_ASSIGN(const bool absent,
                            NNDetail::read_bool_or(row("{}"), "bias", K_CONTEXT, true));
    EXPECT_TRUE(absent);
    SA_ASSERT_OK_AND_ASSIGN(const bool stated,
                            NNDetail::read_bool_or(row(R"({"bias": false})"), "bias", K_CONTEXT,
                                                 true));
    EXPECT_FALSE(stated);
    SA_EXPECT_FAILURE(NNDetail::read_bool_or(row(R"({"bias": 1})"), "bias", K_CONTEXT, true),
                      StatusCode::InvalidDescription, "'bias'", "true or false");
}

/** @brief Tests that read_real takes a whole number as a real and refuses an absent key. */
TEST(Small_NodeAttributes, ReadRealTakesAnyNumber)
{
    SA_ASSERT_OK_AND_ASSIGN(const double whole,
                            NNDetail::read_real(row(R"({"scale": 2})"), "scale", K_CONTEXT));
    EXPECT_NEAR(whole, 2.0, K_REAL_TOLERANCE);
    SA_ASSERT_OK_AND_ASSIGN(const double real,
                            NNDetail::read_real(row(R"({"scale": 0.25})"), "scale", K_CONTEXT));
    EXPECT_NEAR(real, 0.25, K_REAL_TOLERANCE);

    SA_EXPECT_FAILURE(NNDetail::read_real(row("{}"), "scale", K_CONTEXT),
                      StatusCode::InvalidDescription, "'scale'", "absent");
    SA_EXPECT_FAILURE(NNDetail::read_real(row(R"({"scale": "big"})"), "scale", K_CONTEXT),
                      StatusCode::InvalidDescription, "'scale'", "must be a number");
}

/** @brief Tests that the norm readers default to the layers' own values and refuse bad ones. */
TEST(Small_NodeAttributes, TheNormReadersDefaultToTheLayersValues)
{
    SA_ASSERT_OK_AND_ASSIGN(const double layer_epsilon,
                            NNDetail::read_layer_norm_epsilon(row("{}"), K_CONTEXT));
    EXPECT_NEAR(layer_epsilon, NN::K_LAYER_NORM_EPSILON, K_REAL_TOLERANCE);
    SA_EXPECT_FAILURE(NNDetail::read_layer_norm_epsilon(row(R"({"ln_epsilon": 0})"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'ln_epsilon'", "positive");

    SA_ASSERT_OK_AND_ASSIGN(const NN::NormOptions defaults,
                            NNDetail::read_norm_options(row("{}"), K_CONTEXT));
    EXPECT_NEAR(defaults.epsilon, NN::NormOptions{}.epsilon, K_REAL_TOLERANCE);
    EXPECT_NEAR(defaults.momentum, NN::NormOptions{}.momentum, K_REAL_TOLERANCE);

    SA_ASSERT_OK_AND_ASSIGN(
        const NN::NormOptions stated,
        NNDetail::read_norm_options(row(R"({"bn_epsilon": 0.001, "bn_momentum": 0.03})"),
                                  K_CONTEXT));
    EXPECT_NEAR(stated.epsilon, 0.001, K_REAL_TOLERANCE);
    EXPECT_NEAR(stated.momentum, 0.03, K_REAL_TOLERANCE);
    SA_EXPECT_FAILURE(NNDetail::read_norm_options(row(R"({"bn_momentum": 1.5})"), K_CONTEXT),
                      StatusCode::InvalidDescription, "'bn_momentum'", "[0, 1]");
}

/** @brief Tests that wrap_block presents a block as a one-port node, or returns its refusal. */
TEST(Small_NodeAttributes, WrapBlockPresentsTheBlockAsAOnePortNode)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<NN::Node> node,
                            NNDetail::wrap_block(NN::Linear{6, 8}, Shape{6}));
    ASSERT_EQ(node->ports().size(), 1u);
    EXPECT_EQ(node->ports()[0].shape, (Shape{8}));
    EXPECT_EQ(node->input_count(), 1u);

    EXPECT_FALSE(NNDetail::wrap_block(NN::Linear{6, 8}, Shape{5}).has_value());
}
