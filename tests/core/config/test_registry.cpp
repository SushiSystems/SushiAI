/**************************************************************************/
/* test_registry.cpp                                                      */
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
 * @file test_registry.cpp
 * @brief Tests that find_entry finds a row by its type and refuses an unknown type.
 * @author Mustafa Garip
 */

#include <array>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

namespace
{
    /** @brief Holds one row of the test table: a discriminator and a value. */
    struct Row
    {
        std::string_view type; /**< The discriminator a file names. */
        int value;             /**< What the row carries. */
    };

    /** @brief Holds the table every case looks up. */
    constexpr std::array<Row, 3> K_ROWS = {{
        {"sgd", 1},
        {"adamw", 2},
        {"lamb", 3},
    }};
} // namespace

/** @brief Tests that a known type returns the table's own row, not a copy. */
TEST(Small_ConfigRegistry, AKnownTypeReturnsTheTablesOwnRow)
{
    const Row& found = SushiAI::Config::find_entry(K_ROWS, "adamw", "optimizer");

    EXPECT_EQ(&found, &K_ROWS[1]);
    EXPECT_EQ(found.value, 2);
}

/** @brief Tests that a type is matched whole, so a prefix of a row's type is unknown. */
TEST(Small_ConfigRegistry, APrefixOfATypeIsNotAMatch)
{
    SA_EXPECT_ERROR((void)SushiAI::Config::find_entry(K_ROWS, "adam", "optimizer"),
                    "unknown optimizer 'adam'");
}

/** @brief Tests that an unknown type throws naming the category and every row, in table order. */
TEST(Small_ConfigRegistry, AnUnknownTypeThrowsListingEveryRow)
{
    SA_EXPECT_ERROR((void)SushiAI::Config::find_entry(K_ROWS, "rmsprop", "optimizer"),
                    "unknown optimizer 'rmsprop'",
                    "this build has: 'sgd', 'adamw', 'lamb'");
}
