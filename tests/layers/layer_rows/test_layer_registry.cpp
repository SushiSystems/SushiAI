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
 * @file test_layer_registry.cpp
 * @brief Tests the registry make_layer_registry() builds and the type list of types_of().
 * @author Mustafa Garip
 */

#include <algorithm>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

/** @brief Tests that the built registry holds the "linear" row and no row that is not a layer. */
TEST(Small_LayerRowsRegistry, TheRegistryHoldsTheLayerRowsByType)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();

    EXPECT_NE(registry.find_layer("linear"), nullptr);
    EXPECT_EQ(registry.find_layer("no_such_layer"), nullptr);
}

/** @brief Tests that each call builds a registry of its own, with storage of its own. */
TEST(Small_LayerRowsRegistry, EachCallBuildsAFreshRegistryOfTheLayerRows)
{
    const NN::NodeRegistry first = SaTest::make_layer_registry();
    const NN::NodeRegistry second = SaTest::make_layer_registry();

    EXPECT_EQ(first.layers().size(), second.layers().size());
    EXPECT_NE(first.layers().data(), second.layers().data());
}

/** @brief Tests that the listed types of the built registry match its rows one for one. */
TEST(Small_LayerRowsRegistry, TheListedTypesMatchTheRegistryRows)
{
    const NN::NodeRegistry registry = SaTest::make_layer_registry();
    const std::vector<std::string_view> types = SaTest::types_of(registry.layers());

    EXPECT_EQ(types.size(), registry.layers().size());
    EXPECT_TRUE(std::all_of(types.begin(), types.end(), [&registry](std::string_view type)
                            { return registry.find_layer(type) != nullptr; }));
}
