/**************************************************************************/
/* test_registries.cpp                                                    */
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
 * @file test_registries.cpp
 * @brief Tests that compose_registries fills the registries of an app with the library's rows.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include "registries.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace Apps = SushiAI::Apps;

namespace
{
    /** @brief Returns how many rows @p registries holds across every table it has. */
    std::size_t row_count(const Apps::Registries& registries)
    {
        return registries.nodes.nodes().size() + registries.nodes.layers().size() +
               registries.nodes.linked().size() + registries.datasets.rows().size() +
               registries.objectives.rows().size() + registries.runs.rows().size() +
               registries.inferences.rows().size() + registries.importers.rows().size();
    }
} // namespace

/** @brief Tests that a model, data and objective block of the library each find their row. */
TEST(Small_Registries, TheLibraryRowsAreComposed)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries registries, Apps::compose_registries());

    EXPECT_NE(registries.nodes.find_layer("linear"), nullptr);
    EXPECT_NE(registries.nodes.find_node("batchnorm2d"), nullptr);
    EXPECT_NE(registries.datasets.find("rings"), nullptr);
    EXPECT_NE(registries.objectives.find("cross_entropy"), nullptr);
    EXPECT_NE(registries.runs.fallback(), nullptr);
}

/** @brief Tests that two compositions are independent and hold the same rows. */
TEST(Small_Registries, EachCompositionStartsFromEmptyRegistries)
{
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries first, Apps::compose_registries());
    SA_ASSERT_OK_AND_ASSIGN(const Apps::Registries second, Apps::compose_registries());

    EXPECT_GT(row_count(first), 0u);
    EXPECT_EQ(row_count(second), row_count(first));
}
