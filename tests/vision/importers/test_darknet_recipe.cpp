/**************************************************************************/
/* test_darknet_recipe.cpp                                                */
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
 * @file test_darknet_recipe.cpp
 * @brief Tests the training settings of a .cfg written as the sections of a training file.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_recipe.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>

using namespace SushiAI;
using Vision::Importers::DarknetConfig;
using Vision::Importers::DarknetRecipe;
using Vision::Importers::DarknetTraining;

namespace
{
    /** @brief Holds the tolerance of every double a test reads back, absolute. */
    constexpr double K_DOUBLE_TOLERANCE = 1e-12;

    /** @brief Holds the tolerance of every float a test reads back, absolute. */
    constexpr float K_FLOAT_TOLERANCE = 1e-6f;

    /** @brief Holds a [net] section with a solver, a schedule and three keys without a place. */
    constexpr std::string_view K_SOLVER =
        "[net]\nbatch=8\nsubdivisions=2\nmomentum=0.9\ndecay=0.0005\nlearning_rate=0.001\n"
        "burn_in=100\nmax_batches=2000\npolicy=steps\nsteps=1600,1800\nscales=.1,.1\n";

    /** @brief Returns a [yolo] section that ignores above @p ignore_thresh, on five lines. */
    std::string head_section(std::string_view ignore_thresh)
    {
        return "[yolo]\nmask=0,1,2\nclasses=1\njitter=.3\nignore_thresh="
             + std::string(ignore_thresh) + "\n";
    }

    /** @brief Parses @p text as a .cfg and writes its recipe. */
    Result<DarknetRecipe> write_recipe_of(const std::string& text)
    {
        SA_TRY_ASSIGN(const DarknetConfig config,
                      Vision::Importers::parse_darknet_config(text, "test.cfg"));
        SA_TRY_ASSIGN(const DarknetTraining training,
                      Vision::Importers::read_darknet_training(config));
        return Vision::Importers::write_darknet_recipe(config, training);
    }

    /** @brief Returns whether a note of @p notes contains every one of @p needles. */
    bool has_note(const std::vector<std::string>& notes,
                  std::initializer_list<std::string_view> needles)
    {
        return std::any_of(notes.begin(), notes.end(), [&](const std::string& note)
        {
            return std::all_of(needles.begin(), needles.end(), [&](std::string_view needle)
            {
                return note.find(needle) != std::string::npos;
            });
        });
    }
} // namespace

/** @brief Tests that the solver becomes "sgd" with its momentum and its decay. */
TEST(Unit_DarknetRecipe, WritesTheSolverAsSGD)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetRecipe recipe,
                            write_recipe_of(std::string(K_SOLVER) + head_section(".7")));

    const Config::Value& optimizer = recipe.training.at("optimizer", "the recipe");
    EXPECT_EQ(optimizer.keys().size(), 3u);
    EXPECT_EQ(optimizer.at("type", "optimizer").as_string("type"), "sgd");
    EXPECT_NEAR(optimizer.at("momentum", "optimizer").as_real("momentum"), 0.9,
                K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(optimizer.at("weight_decay", "optimizer").as_real("weight_decay"), 0.0005,
                K_DOUBLE_TOLERANCE);
    EXPECT_FALSE(optimizer.has("learning_rate"));
}

/** @brief Tests that the schedule section reads back as the rate, burn-in, steps and scales. */
TEST(Unit_DarknetRecipe, WritesTheRateAsAStepSchedule)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetRecipe recipe,
                            write_recipe_of(std::string(K_SOLVER) + head_section(".7")));

    SA_ASSERT_OK_AND_ASSIGN(const Optim::StepSchedule schedule,
                            Optim::read_step_schedule_section(
                                recipe.training.at("schedule", "the recipe")));
    EXPECT_NEAR(schedule.base_rate, 0.001, K_DOUBLE_TOLERANCE);
    EXPECT_EQ(schedule.warmup_iterations, 100);
    EXPECT_EQ(schedule.steps, (std::vector<std::int64_t>{1600, 1800}));
    ASSERT_EQ(schedule.scales.size(), 2u);
    EXPECT_NEAR(schedule.scales[0], 0.1, K_DOUBLE_TOLERANCE);
    EXPECT_NEAR(schedule.scales[1], 0.1, K_DOUBLE_TOLERANCE);
}

/** @brief Tests that the objective section is "anchor" and reads back as the head's options. */
TEST(Unit_DarknetRecipe, WritesTheSharedHeadOptionsAsTheAnchorObjective)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetRecipe recipe,
        write_recipe_of(std::string(K_SOLVER) + head_section(".7") + head_section(".7")));

    const Config::Value& objective = recipe.training.at("objective", "the recipe");
    EXPECT_EQ(objective.at("type", "objective").as_string("type"), "anchor");

    SA_ASSERT_OK_AND_ASSIGN(const Vision::Detect::Anchor::HeadTrainingOptions options,
                            Vision::Detect::Anchor::read_head_training_section(objective));
    EXPECT_NEAR(options.thresholds.ignore_threshold, 0.7f, K_FLOAT_TOLERANCE);
    EXPECT_EQ(options.box_gradient, Vision::Detect::BoxGradient::DarknetCompatible);
}

/** @brief Tests that the recipe holds the three sections and no other key. */
TEST(Unit_DarknetRecipe, HoldsTheObjectiveTheOptimizerAndTheSchedule)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetRecipe recipe,
                            write_recipe_of(std::string(K_SOLVER) + head_section(".7")));

    EXPECT_EQ(recipe.training.keys(),
              (std::vector<std::string>{"objective", "optimizer", "schedule"}));
}

/** @brief Tests that a key no training file holds becomes a note with its line and value. */
TEST(Unit_DarknetRecipe, NotesEveryKeyWithoutAPlace)
{
    SA_ASSERT_OK_AND_ASSIGN(const DarknetRecipe recipe,
                            write_recipe_of(std::string(K_SOLVER) + head_section(".7")));

    EXPECT_EQ(recipe.notes.size(), 4u);
    EXPECT_TRUE(has_note(recipe.notes, {"line 2", "batch=8", "[net]", "batch_size"}));
    EXPECT_TRUE(has_note(recipe.notes, {"line 3", "subdivisions=2", "[net]"}));
    EXPECT_TRUE(has_note(recipe.notes, {"line 8", "max_batches=2000", "epochs"}));
    EXPECT_TRUE(has_note(recipe.notes, {"line 15", "jitter=.3", "[yolo]"}));
}

/** @brief Tests that a .cfg whose keys all have a place leaves no note. */
TEST(Unit_DarknetRecipe, LeavesNoNoteWhenEveryKeyHasAPlace)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetRecipe recipe,
        write_recipe_of("[net]\nlearning_rate=0.01\n[yolo]\nmask=0\nclasses=1\n"));

    EXPECT_TRUE(recipe.notes.empty());
}

/** @brief Tests that heads with different options are refused with both heads and the key. */
TEST(Unit_DarknetRecipe, HeadsThatDifferAreRefused)
{
    SA_EXPECT_FAILURE(
        write_recipe_of(std::string(K_SOLVER) + head_section(".7") + head_section(".5")),
        StatusCode::InvalidDescription, "head 0 (line 12)", "head 1 (line 17)",
        "'ignore_threshold'", "one set");
}

/** @brief Tests that settings without a head are refused, since they state no objective. */
TEST(Unit_DarknetRecipe, AFileWithoutAHeadIsRefused)
{
    SA_EXPECT_FAILURE(write_recipe_of(std::string(K_SOLVER)), StatusCode::InvalidDescription,
                      "[yolo]", "no objective");
}
