/**************************************************************************/
/* test_rule.cpp                                                          */
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
 * @file test_rule.cpp
 * @brief Tests that concat_rules joins two rule tables at compile time, in order.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <type_traits>
#include <gtest/gtest.h>
#include <SushiAI/autograd/rule.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/ops/ops.hpp>

using SushiAI::Autograd::BackwardRule;
using SushiAI::Autograd::concat_rules;
using SushiAI::Autograd::GradientTape;
using SushiAI::Autograd::RuleEntry;
namespace Ops = SushiAI::Ops;

namespace
{
    /** @brief Stands in for a first backward rule; does nothing. */
    void first_rule(const GradientTape&)
    {

    }

    /** @brief Stands in for a second backward rule; does nothing. */
    void second_rule(const GradientTape&)
    {

    }

    /** @brief Holds a left table of two rows. */
    constexpr std::array<RuleEntry, 2> K_LEFT = {{
        {Ops::K_ADD.id, &first_rule},
        {Ops::K_SUB.id, &first_rule},
    }};

    /** @brief Holds a right table of one row. */
    constexpr std::array<RuleEntry, 1> K_RIGHT = {{
        {Ops::K_MUL.id, &second_rule},
    }};

    /** @brief Holds the joined table, built at compile time. */
    constexpr std::array<RuleEntry, 3> K_JOINED = concat_rules(K_LEFT, K_RIGHT);
} // namespace

/** @brief Tests that a rule is a function of a read-only tape. */
TEST(Small_AutogradRule, ARuleIsAFunctionOfAReadOnlyTape)
{
    static_assert(std::is_same<BackwardRule, void (*)(const GradientTape&)>::value,
                  "a backward rule reads the tape through a const reference");

    const RuleEntry entry{Ops::K_ADD.id, &first_rule};
    EXPECT_EQ(entry.op, Ops::K_ADD.id);
    EXPECT_EQ(entry.rule, &first_rule);
}

/** @brief Tests that the joined table holds the left rows, then the right rows, in order. */
TEST(Small_AutogradRule, TheJoinedTableHoldsTheLeftRowsThenTheRight)
{
    static_assert(K_JOINED.size() == 3, "the joined size is the sum of both sizes");
    static_assert(K_JOINED[2].op == Ops::K_MUL.id, "the join runs at compile time");

    EXPECT_EQ(K_JOINED[0].op, Ops::K_ADD.id);
    EXPECT_EQ(K_JOINED[1].op, Ops::K_SUB.id);
    EXPECT_EQ(K_JOINED[2].op, Ops::K_MUL.id);
    EXPECT_EQ(K_JOINED[0].rule, &first_rule);
    EXPECT_EQ(K_JOINED[1].rule, &first_rule);
    EXPECT_EQ(K_JOINED[2].rule, &second_rule);
}

/** @brief Tests that joining an empty table leaves the other table's rows unchanged. */
TEST(Small_AutogradRule, JoiningAnEmptyTableKeepsTheOtherOne)
{
    constexpr std::array<RuleEntry, 0> empty{};
    constexpr std::array<RuleEntry, 1> right_only = concat_rules(empty, K_RIGHT);
    constexpr std::array<RuleEntry, 2> left_only = concat_rules(K_LEFT, empty);

    EXPECT_EQ(right_only[0].op, Ops::K_MUL.id);
    EXPECT_EQ(right_only[0].rule, &second_rule);
    EXPECT_EQ(left_only[0].op, Ops::K_ADD.id);
    EXPECT_EQ(left_only[1].op, Ops::K_SUB.id);
}
