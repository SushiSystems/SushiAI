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
 * @file test_objective_registry.cpp
 * @brief Tests the registry of the objectives a training file may name.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Train::ObjectiveContext;
using Train::ObjectiveEntry;
using Train::ObjectiveRegistry;

namespace
{
    /** @brief Holds the batch the stand-in factory expects of its context. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the classes the stand-in factory expects of its context. */
    constexpr std::int64_t K_CLASSES = 3;

    /** @brief Returns a context of K_BATCH samples over K_CLASSES with one published output. */
    ObjectiveContext make_context()
    {
        ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{6};
        given.task = Config::parse_json(R"({"type": "stand_in"})", "<test>");
        given.outputs.push_back(Train::OutputShape{"logits", Shape{K_BATCH, K_CLASSES}});
        return given;
    }

    /**
     * @brief Builds cross-entropy when the context arrives as make_context() made it.
     * @return The objective, or Unsupported naming the first field that arrived changed.
     */
    Result<std::unique_ptr<Train::Objective>> make_stand_in(const ObjectiveContext& given,
                                                            const Config::Value& attributes,
                                                            std::string_view context)
    {
        SA_FAIL_IF(given.task.is_null() || !given.task.has("type"), StatusCode::Unsupported,
                   "{} received no task", context);
        SA_FAIL_IF(given.outputs.size() != 1 || given.outputs.front().name != "logits",
                   StatusCode::Unsupported, "{} received other outputs", context);
        SA_FAIL_IF(given.input_sample != Shape{6}, StatusCode::Unsupported,
                   "{} received another input sample", context);
        SA_FAIL_IF(!attributes.has("weight"), StatusCode::Unsupported,
                   "{} received other attributes", context);
        SA_FAIL_IF(given.registry == nullptr || given.registry->find("stand_in") == nullptr,
                   StatusCode::Unsupported, "{} received no registry that holds its row",
                   context);

        return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<Train::Objective>(
                std::make_unique<Train::CrossEntropyObjective>(given.batch, given.classes));
        });
    }
} // namespace

/** @brief Tests that the layers tier registers its five objectives and no domain's. */
TEST(Unit_ObjectiveRegistry, TheLayersTierAddsItsFiveObjectives)
{
    ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));

    ASSERT_EQ(registry.rows().size(), 5u);
    EXPECT_EQ(registry.rows()[0].type, "cross_entropy");
    EXPECT_EQ(registry.rows()[1].type, "binary_cross_entropy");
    EXPECT_EQ(registry.rows()[2].type, "multi_label");
    EXPECT_EQ(registry.rows()[3].type, "mean_squared_error");
    EXPECT_EQ(registry.rows()[4].type, "sum");
    EXPECT_NE(registry.find("cross_entropy"), nullptr);
    EXPECT_EQ(registry.find("anchor"), nullptr);
}

/** @brief Tests that a type is held once and that the refusal names it. */
TEST(Unit_ObjectiveRegistry, ATypeIsHeldOnce)
{
    ObjectiveRegistry registry;
    SA_EXPECT_OK(registry.add(ObjectiveEntry{"stand_in", &make_stand_in}));

    SA_EXPECT_FAILURE(registry.add(ObjectiveEntry{"stand_in", &make_stand_in}),
                      StatusCode::InvalidDescription, "'stand_in'", "already registered");
    EXPECT_EQ(registry.rows().size(), 1u);
}

/** @brief Tests that the factory receives the context, the registry, the keys and the label. */
TEST(Unit_ObjectiveRegistry, TheFactoryReceivesTheContextAndItsRegistry)
{
    ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));
    SA_EXPECT_OK(registry.add(ObjectiveEntry{"stand_in", &make_stand_in}));
    const Config::Value attributes = Config::parse_json(R"({"weight": 2.0})", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        Train::make_objective(registry, "stand_in", make_context(), attributes, "objective"));

    ASSERT_NE(objective, nullptr);
    EXPECT_EQ(objective->name(), "cross_entropy");
}

/** @brief Tests that an unknown type is refused with every type the registry holds. */
TEST(Unit_ObjectiveRegistry, AnUnknownTypeListsWhatTheRegistryHolds)
{
    ObjectiveRegistry registry;
    SA_EXPECT_OK(Train::register_objectives(registry));
    SA_EXPECT_OK(registry.add(ObjectiveEntry{"stand_in", &make_stand_in}));
    const Config::Value empty = Config::parse_json("{}", "<test>");

    SA_EXPECT_FAILURE(
        Train::make_objective(registry, "yolo", make_context(), empty, "objective"),
        StatusCode::InvalidDescription, "'yolo'", "'cross_entropy'", "'stand_in'");
}

/** @brief Tests that the failure of a factory reaches the caller as it was returned. */
TEST(Unit_ObjectiveRegistry, TheFailureOfAFactoryIsReturned)
{
    ObjectiveRegistry registry;
    SA_EXPECT_OK(registry.add(ObjectiveEntry{"stand_in", &make_stand_in}));
    const Config::Value empty = Config::parse_json("{}", "<test>");

    SA_EXPECT_FAILURE(
        Train::make_objective(registry, "stand_in", make_context(), empty, "the objective"),
        StatusCode::Unsupported, "the objective", "other attributes");
}
