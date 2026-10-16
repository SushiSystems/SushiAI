/**************************************************************************/
/* test_training_graph.cpp                                                */
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
 * @file test_training_graph.cpp
 * @brief Tests the values a StepSpec, a ParameterBinding and a TrainingGraph start with.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
using Graph::K_NO_VALUE;

/** @brief Tests that a default StepSpec names none of the four values a loop reads. */
TEST(Small_TrainingGraph, ADefaultSpecNamesNoValue)
{
    const Train::StepSpec spec{};

    EXPECT_EQ(spec.inputs, K_NO_VALUE);
    EXPECT_EQ(spec.targets, K_NO_VALUE);
    EXPECT_EQ(spec.logits, K_NO_VALUE);
    EXPECT_EQ(spec.loss, K_NO_VALUE);
}

/** @brief Tests that a default binding is an unnamed weight with no value and no gradient. */
TEST(Small_TrainingGraph, ADefaultBindingIsAWeightWithNoValues)
{
    const Train::ParameterBinding binding{};

    EXPECT_TRUE(binding.name.empty());
    EXPECT_EQ(binding.value, K_NO_VALUE);
    EXPECT_EQ(binding.gradient, K_NO_VALUE);
    EXPECT_EQ(binding.role, NN::ParameterRole::Weight);
}

/** @brief Tests that a default training graph is empty and asks for no loss scale. */
TEST(Small_TrainingGraph, ADefaultGraphIsEmptyAndUnscaled)
{
    const Train::TrainingGraph built;

    EXPECT_EQ(built.graph.node_count(), 0u);
    EXPECT_EQ(built.graph.value_count(), 0u);
    EXPECT_TRUE(built.parameters.empty());
    EXPECT_TRUE(built.pinned.empty());
    EXPECT_EQ(built.loss_scale, K_NO_VALUE);
    EXPECT_EQ(built.spec.loss, K_NO_VALUE);
    EXPECT_FALSE(built.leaves.has(Data::K_ROLE_LABELS));
}
