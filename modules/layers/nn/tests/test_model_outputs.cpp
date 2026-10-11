/**************************************************************************/
/* test_model_outputs.cpp                                                 */
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
 * @file test_model_outputs.cpp
 * @brief Tests ModelOutputs: publishing named values and reading them back.
 * @author Mustafa Garip
 */

#include <string>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <common/test_status.hpp>

using SushiAI::StatusCode;
using SushiAI::Graph::ValueID;
using SushiAI::NN::ModelOutputs;

static_assert(std::is_final<ModelOutputs>::value, "ModelOutputs is not a base class");
static_assert(std::is_copy_constructible<ModelOutputs>::value, "a set of ids copies");
static_assert(std::is_nothrow_move_constructible<ModelOutputs>::value, "a move never throws");

namespace
{
    /** @brief Returns a set holding "boxes" as value 4 and "scores" as value 9, in that order. */
    ModelOutputs boxes_and_scores()
    {
        ModelOutputs outputs;
        SA_EXPECT_OK(outputs.add("boxes", ValueID{4}));
        SA_EXPECT_OK(outputs.add("scores", ValueID{9}));
        return outputs;
    }

    /** @brief Adds an output that carries Graph::NO_VALUE, which stops the process. */
    void add_without_value()
    {
        ModelOutputs outputs;
        const SushiAI::Status ignored = outputs.add("boxes", SushiAI::Graph::K_NO_VALUE);
        static_cast<void>(ignored);
    }

} // namespace

/** @brief Tests that a new set holds no output. */
TEST(Unit_ModelOutputs, StartsEmpty)
{
    const ModelOutputs outputs;

    EXPECT_TRUE(outputs.all().empty());
}

/** @brief Tests that all() returns the outputs in the order they were added. */
TEST(Unit_ModelOutputs, AllKeepsTheOrderOfAdd)
{
    ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("scores", ValueID{9}));
    SA_EXPECT_OK(outputs.add("boxes", ValueID{4}));
    SA_EXPECT_OK(outputs.add("features", ValueID{0}));

    ASSERT_EQ(outputs.all().size(), 3u);
    EXPECT_EQ(outputs.all()[0], ModelOutputs::Entry("scores", ValueID{9}));
    EXPECT_EQ(outputs.all()[1], ModelOutputs::Entry("boxes", ValueID{4}));
    EXPECT_EQ(outputs.all()[2], ModelOutputs::Entry("features", ValueID{0}));
}

/** @brief Tests that add refuses a name the set already holds and keeps the first value. */
TEST(Unit_ModelOutputs, AddRejectsADuplicateName)
{
    ModelOutputs outputs = boxes_and_scores();

    SA_EXPECT_FAILURE(outputs.add("boxes", ValueID{7}), StatusCode::InvalidDescription,
                      "'boxes'", "boxes, scores");

    ASSERT_EQ(outputs.all().size(), 2u);
    SA_ASSERT_OK_AND_ASSIGN(const ValueID boxes, outputs.at("boxes"));
    EXPECT_EQ(boxes, ValueID{4});
}

/** @brief Tests that add refuses an empty name and leaves the set unchanged. */
TEST(Unit_ModelOutputs, AddRejectsAnEmptyName)
{
    ModelOutputs outputs;

    SA_EXPECT_FAILURE(outputs.add("", ValueID{1}), StatusCode::InvalidDescription, "needs a name");

    EXPECT_TRUE(outputs.all().empty());
}

/** @brief Tests that add stops the process when the value is Graph::NO_VALUE. */
TEST(Unit_ModelOutputsDeathTest, AddStopsOnNoValue)
{
    SA_EXPECT_FATAL(add_without_value(), "'boxes'", "no value");
}

/** @brief Tests that at returns the value published under each name. */
TEST(Unit_ModelOutputs, AtReturnsTheNamedValue)
{
    const ModelOutputs outputs = boxes_and_scores();

    SA_ASSERT_OK_AND_ASSIGN(const ValueID boxes, outputs.at("boxes"));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID scores, outputs.at("scores"));
    EXPECT_EQ(boxes, ValueID{4});
    EXPECT_EQ(scores, ValueID{9});
}

/** @brief Tests that at matches a whole name and not a prefix of one. */
TEST(Unit_ModelOutputs, AtDoesNotMatchAPrefix)
{
    const ModelOutputs outputs = boxes_and_scores();

    SA_EXPECT_FAILURE(outputs.at("box"), StatusCode::InvalidDescription, "'box'");
    SA_EXPECT_FAILURE(outputs.at("boxes."), StatusCode::InvalidDescription, "'boxes.'");
}

/** @brief Tests that at on a missing name fails and lists every name the set holds. */
TEST(Unit_ModelOutputs, AtOnAMissingNameListsTheNames)
{
    const ModelOutputs outputs = boxes_and_scores();

    SA_EXPECT_FAILURE(outputs.at("masks"), StatusCode::InvalidDescription, "'masks'",
                      "boxes, scores");
}

/** @brief Tests that at on an empty set fails and says the set holds none. */
TEST(Unit_ModelOutputs, AtOnAnEmptySetSaysNone)
{
    const ModelOutputs outputs;

    SA_EXPECT_FAILURE(outputs.at("boxes"), StatusCode::InvalidDescription, "'boxes'", "none");
}

/** @brief Tests that only returns the value of a set that holds exactly one. */
TEST(Unit_ModelOutputs, OnlyReturnsTheSingleValue)
{
    ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("logits", ValueID{12}));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID logits, outputs.only());
    EXPECT_EQ(logits, ValueID{12});
}

/** @brief Tests that only fails on a set that holds no output. */
TEST(Unit_ModelOutputs, OnlyFailsOnZero)
{
    const ModelOutputs outputs;

    SA_EXPECT_FAILURE(outputs.only(), StatusCode::InvalidDescription, "exactly one", "has 0",
                      "none");
}

/** @brief Tests that only fails on a set that holds two outputs and names both. */
TEST(Unit_ModelOutputs, OnlyFailsOnTwo)
{
    const ModelOutputs outputs = boxes_and_scores();

    SA_EXPECT_FAILURE(outputs.only(), StatusCode::InvalidDescription, "exactly one", "has 2",
                      "boxes, scores");
}
