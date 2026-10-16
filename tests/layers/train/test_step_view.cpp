/**************************************************************************/
/* test_step_view.cpp                                                     */
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
 * @file test_step_view.cpp
 * @brief Tests what a StepOutput holds by default and what a StepView carries.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Bounds the difference between the loss a view is given and the one it returns. */
    constexpr double K_LOSS_TOLERANCE = 0.0;
} // namespace

/** @brief Tests that a default StepOutput names nothing and points at no tensor. */
TEST(Small_StepView, ADefaultOutputIsUnbound)
{
    const Train::StepOutput output{};

    EXPECT_TRUE(output.name.empty());
    EXPECT_EQ(output.tensor, nullptr);
}

/** @brief Tests that a view refers to its batch and keeps the loss and the outputs in order. */
TEST(Small_StepView, AViewCarriesTheBatchTheLossAndTheOutputsInOrder)
{
    const SushiBLAS::Tensor small{};
    const SushiBLAS::Tensor large{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> outputs{Train::StepOutput{"p3", &small},
                                                 Train::StepOutput{"p4", &large}};

    const Train::StepView view{
        batch, 0.75,
        SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};

    EXPECT_EQ(&view.batch, &batch);
    EXPECT_NEAR(view.loss, 0.75, K_LOSS_TOLERANCE);
    ASSERT_EQ(view.outputs.size(), 2u);
    EXPECT_EQ(view.outputs[0].name, "p3");
    EXPECT_EQ(view.outputs[0].tensor, &small);
    EXPECT_EQ(view.outputs[1].name, "p4");
    EXPECT_EQ(view.outputs[1].tensor, &large);
}
