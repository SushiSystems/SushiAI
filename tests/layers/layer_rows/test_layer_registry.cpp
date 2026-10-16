/**************************************************************************/
/* test_layer_registry.cpp                                                */
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
 * @file test_layer_registry.cpp
 * @brief Tests the registry make_layer_registry() builds and the type list of types_of().
 * @author Mustafa Garip
 */

#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

/** @brief Tests that each call builds a registry of its own holding the layer rows. */
TEST(Small_LayerRowsRegistry, EachCallBuildsAFreshRegistryOfTheLayerRows)
{
    const NN::NodeRegistry first = SaTest::make_layer_registry();
    const NN::NodeRegistry second = SaTest::make_layer_registry();

    EXPECT_NE(first.find_layer("linear"), nullptr);
    EXPECT_EQ(first.find_layer("no_such_layer"), nullptr);
    EXPECT_EQ(first.layers().size(), second.layers().size());
    EXPECT_NE(first.layers().data(), second.layers().data());
}

/** @brief Tests that types_of() lists the type of every row in insertion order. */
TEST(Small_LayerRowsRegistry, TypesOfListsEveryRowInOrder)
{
    const std::vector<NN::LayerEntry> rows{NN::LayerEntry{"first", nullptr},
                                           NN::LayerEntry{"second", nullptr},
                                           NN::LayerEntry{"third", nullptr}};

    EXPECT_EQ(SaTest::types_of(SushiRuntime::span<const NN::LayerEntry>(rows.data(), rows.size())),
              (std::vector<std::string_view>{"first", "second", "third"}));
    EXPECT_TRUE(SaTest::types_of(SushiRuntime::span<const NN::LayerEntry>()).empty());
}

/** @brief Tests that the listed types of the built registry match its rows one for one. */
TEST(Small_LayerRowsRegistry, TheListedTypesMatchTheRegistryRows)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();
    const std::vector<std::string_view> types = SaTest::types_of(registry.layers());

    ASSERT_EQ(types.size(), registry.layers().size());
    for (const std::string_view type : types)
        EXPECT_NE(registry.find_layer(type), nullptr) << type;
}
