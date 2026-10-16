/**************************************************************************/
/* test_parameter_groups.cpp                                              */
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
 * @file test_parameter_groups.cpp
 * @brief Tests that weight decay is decided by a parameter's role, at compile time.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/train/parameter_groups.hpp>

using namespace SushiAI;
using NN::ParameterRole;

namespace
{
    /** @brief Lists every parameter role, the weight first. */
    constexpr ParameterRole K_ROLES[] = {ParameterRole::Weight, ParameterRole::Bias,
                                         ParameterRole::Scale, ParameterRole::Embedding};

    /** @brief Counts the roles of K_ROLES that take weight decay, in a constant expression. */
    constexpr std::size_t decaying_role_count()
    {
        std::size_t count = 0;
        for (const ParameterRole role : K_ROLES)
            count += Train::takes_weight_decay(role) ? 1u : 0u;
        return count;
    }
} // namespace

/** @brief Tests that the weight is the one role of four that takes decay, known at compile time. */
TEST(Small_ParameterGroups, TheWeightIsTheOnlyRoleThatDecays)
{
    static_assert(decaying_role_count() == 1u, "exactly one role takes weight decay");
    static_assert(Train::takes_weight_decay(ParameterRole::Weight), "a weight takes decay");
    static_assert(!Train::takes_weight_decay(ParameterRole::Embedding),
                  "an embedding table takes no decay");

    EXPECT_EQ(decaying_role_count(), 1u);
}
