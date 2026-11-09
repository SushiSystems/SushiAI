/**************************************************************************/
/* test_objective_registry.cpp                                            */
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
 * @file test_objective_registry.cpp
 * @brief Tests the registry of the objectives a training file may name.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>

using namespace SushiAI;
using Train::ObjectiveContext;
using Train::ObjectiveEntry;
using Train::ObjectiveRegistry;

namespace
{
    using SushiAI::Test::require_value;

    /** @brief Holds the batch the context gives its factory. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the classes the context gives its factory. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Returns a context of K_BATCH samples over K_CLASSES with one published output. */
    ObjectiveContext make_context()
    {
        ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{6};
        given.outputs.push_back(Train::OutputShape{"output", Shape{K_BATCH, K_CLASSES}});
        return given;
    }

    /** @brief Returns the registry of the layers tier. */
    ObjectiveRegistry layers_registry()
    {
        ObjectiveRegistry registry;
        EXPECT_TRUE(Train::register_objectives(registry).is_ok());
        return registry;
    }

    /** @brief Holds a type asked of the registry and whether it holds a row for it. */
    struct FindRow
    {
        std::string name;       /**< The row's name. */
        std::string type;       /**< The type asked for. */
        bool held;              /**< Whether the registry holds the row. */
    };

    /** @brief Searches the registry of the layers tier for the type of each row. */
    class Small_ObjectiveRegistryFind : public ::testing::TestWithParam<FindRow>
    {

    };
} // namespace

/** @brief Tests that the layers tier registers its five objectives and no domain's. */
TEST(Small_ObjectiveRegistry, TheLayersTierAddsItsFiveObjectivesInOrder)
{
    const ObjectiveRegistry registry = layers_registry();

    ASSERT_EQ(registry.rows().size(), 5u);
    EXPECT_EQ(registry.rows()[0].type, "cross_entropy");
    EXPECT_EQ(registry.rows()[1].type, "binary_cross_entropy");
    EXPECT_EQ(registry.rows()[2].type, "multi_label");
    EXPECT_EQ(registry.rows()[3].type, "mean_squared_error");
    EXPECT_EQ(registry.rows()[4].type, "sum");
}

/** @brief Tests that find returns the row of a held type and null for another. */
TEST_P(Small_ObjectiveRegistryFind, FindReturnsAHeldTypeAndNullForAnother)
{
    const FindRow& row = GetParam();
    const ObjectiveRegistry registry = layers_registry();

    EXPECT_EQ(registry.find(row.type) != nullptr, row.held);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ObjectiveRegistryFind,
    ::testing::Values(FindRow{"AHeldType", "cross_entropy", true},
                      FindRow{"ADomainType", "anchor", false}),
    [](const ::testing::TestParamInfo<FindRow>& info) { return info.param.name; });

/** @brief Tests that a type is held once and that the refusal names it. */
TEST(Small_ObjectiveRegistry, ATypeIsHeldOnce)
{
    ObjectiveRegistry registry = layers_registry();

    SA_EXPECT_FAILURE(registry.add(ObjectiveEntry{"cross_entropy", &Train::make_cross_entropy}),
                      StatusCode::InvalidDescription, "'cross_entropy'", "already registered");
    EXPECT_EQ(registry.rows().size(), 5u);
}

/** @brief Tests that an unknown type is refused with every type the registry holds. */
TEST(Small_ObjectiveRegistry, AnUnknownTypeListsWhatTheRegistryHolds)
{
    const ObjectiveRegistry registry = layers_registry();
    const Config::Value empty = require_value(Config::parse_json("{}", "<test>"));

    SA_EXPECT_FAILURE(
        Train::make_objective(registry, "yolo", make_context(), empty, "objective"),
        StatusCode::InvalidDescription, "'yolo'", "'cross_entropy'", "'sum'");
}

/** @brief Tests that the failure of a factory reaches the caller as it was returned. */
TEST(Small_ObjectiveRegistry, TheFailureOfAFactoryIsReturned)
{
    const ObjectiveRegistry registry = layers_registry();
    const Config::Value stray =
        require_value(Config::parse_json(R"({"label_smoothing": 0.1})", "<test>"));

    SA_EXPECT_FAILURE(
        Train::make_objective(registry, "cross_entropy", make_context(), stray, "the objective"),
        StatusCode::InvalidDescription, "the objective", "'label_smoothing'");
}

/**
 * @brief Tests that make_objective gives the factory the registry it was called with: the
 *        sum row refuses a context with no registry, and here it reaches its own terms rule.
 */
TEST(Small_ObjectiveRegistry, TheFactoryReceivesTheRegistryItWasCalledWith)
{
    const ObjectiveRegistry registry = layers_registry();
    const Config::Value no_terms = require_value(Config::parse_json(R"({"terms": []})", "<test>"));

    SA_EXPECT_FAILURE(
        Train::make_objective(registry, "sum", make_context(), no_terms, "the objective"),
        StatusCode::InvalidDescription, "at least one term");
}
