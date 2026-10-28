/**************************************************************************/
/* test_free_objective_factory.cpp                                        */
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
 * @file test_free_objective_factory.cpp
 * @brief Tests the factory that builds an anchor-free objective from the files of a run.
 * @author Mustafa Garip
 */

#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>

#include "free_objective_ports.hpp"

using namespace SushiAI;
using namespace SushiAI::FreeObjectiveTest;
using Vision::Training::FreeLossGains;
using Vision::Training::FreeObjective;

namespace
{
    using SushiAI::Test::require_value;
    /** @brief Returns the objective section @p text states. */
    Config::Value section_from(const char* text)
    {
        return require_value(Config::parse_json(text, "<test>"));
    }
} // namespace

/** @brief Tests that the factory derives the facts and reads the gains of the section. */
TEST(Small_FreeObjectiveFactory, TheFactoryDerivesTheFactsAndReadsTheGains)
{
    const Config::Value section = section_from(R"({"type": "anchor_free", "box_gain": 5.0})");

    SA_ASSERT_OK_AND_ASSIGN(const FreeObjective objective,
                            Vision::Training::build_free_objective(make_context(), section));

    EXPECT_EQ(objective.facts().levels.size(), K_SIDES.size());
    EXPECT_EQ(objective.facts().reg_max, K_REG_MAX);
    EXPECT_EQ(objective.facts().levels[1].stride, 16);
    EXPECT_EQ(objective.gains().box_gain, 5.0f);
    EXPECT_EQ(objective.gains().class_gain, FreeLossGains{}.class_gain);
}

/** @brief Tests that a key that is no gain is refused by name. */
TEST(Small_FreeObjectiveFactory, TheFactoryRefusesAKeyThatIsNoGain)
{
    const Config::Value section = section_from(R"({"type": "anchor_free", "reg_max": 16})");

    SA_EXPECT_FAILURE(Vision::Training::build_free_objective(make_context(), section),
                      StatusCode::InvalidDescription, "reg_max");
}

/** @brief Tests that the keys of the one-to-many branch weights are keys of the section. */
TEST(Small_FreeObjectiveFactory, TheBranchWeightKeysAreKeysOfTheSection)
{
    const Config::Value section = section_from(
        R"({"type": "anchor_free", "one_to_many_weight_start": 0.6,
            "one_to_many_weight_end": 0.2})");

    SA_ASSERT_OK_AND_ASSIGN(const FreeObjective objective,
                            Vision::Training::build_free_objective(make_context(), section));

    EXPECT_EQ(objective.name(), Vision::Training::K_FREE_OBJECTIVE_TYPE);
}

/** @brief Tests that a type other than the anchor-free one is refused. */
TEST(Small_FreeObjectiveFactory, ASectionOfAnotherTypeIsRefused)
{
    const Config::Value section = section_from(R"({"type": "anchor"})");

    const Result<FreeObjective> built =
        Vision::Training::build_free_objective(make_context(), section);

    ASSERT_FALSE(built.has_value());
    EXPECT_EQ(built.status().code(), StatusCode::InvalidDescription);
}

/** @brief Tests that a model without a task section is refused naming the section. */
TEST(Small_FreeObjectiveFactory, AModelWithoutATaskIsRefused)
{
    Train::ObjectiveContext given = make_context();
    given.task = Config::Value();

    SA_EXPECT_FAILURE(Vision::Training::build_free_objective(
                          given, section_from(R"({"type": "anchor_free"})")),
                      StatusCode::InvalidDescription, "'task'");
}

/** @brief Tests that a data class count the score ports do not hold is refused by both counts. */
TEST(Small_FreeObjectiveFactory, ADataClassCountOtherThanThePortsIsRefused)
{
    Train::ObjectiveContext given = make_context();
    given.classes = K_CLASSES + 3;

    SA_EXPECT_FAILURE(Vision::Training::build_free_objective(
                          given, section_from(R"({"type": "anchor_free"})")),
                      StatusCode::Incompatible, "5 classes", "hold 2");
}

/** @brief Tests that data that states no class count is taken as the ports have it. */
TEST(Small_FreeObjectiveFactory, DataThatStatesNoClassCountIsAccepted)
{
    Train::ObjectiveContext given = make_context();
    given.classes = 0;

    SA_ASSERT_OK_AND_ASSIGN(const FreeObjective objective,
                            Vision::Training::build_free_objective(
                                given, section_from(R"({"type": "anchor_free"})")));

    EXPECT_EQ(objective.facts().classes, K_CLASSES);
}

/** @brief Tests that the registry entry point builds an objective of the anchor-free type. */
TEST(Small_FreeObjectiveFactory, TheRegistryEntryBuildsTheObjective)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Objective> objective,
        Vision::Training::make_free_objective(make_context(),
                                              section_from(R"({"type": "anchor_free"})"),
                                              "the objective block"));

    ASSERT_NE(objective, nullptr);
    EXPECT_EQ(objective->name(), Vision::Training::K_FREE_OBJECTIVE_TYPE);
}

/** @brief Tests that the registry entry point prefixes a refusal with its context. */
TEST(Small_FreeObjectiveFactory, TheRegistryEntryPrefixesARefusalWithItsContext)
{
    Train::ObjectiveContext given = make_context();
    given.task = Config::Value();

    SA_EXPECT_FAILURE(Vision::Training::make_free_objective(
                          given, section_from(R"({"type": "anchor_free"})"),
                          "the objective block"),
                      StatusCode::InvalidDescription, "the objective block", "'task'");
}
