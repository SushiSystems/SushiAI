/**************************************************************************/
/* test_trainer_run.cpp                                                   */
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
 * @file test_trainer_run.cpp
 * @brief Tests that register_runs sets the fallback run once and adds no row.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include "trainer_run.hpp"
#include "training_run.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::StatusCode;
namespace Apps = SushiAI::Apps;

/** @brief Tests that registering sets a fallback and leaves the table of rows empty. */
TEST(Small_TrainerRun, RegisteringSetsTheFallbackAndAddsNoRow)
{
    Apps::RunRegistry registry;

    SA_EXPECT_OK(Apps::register_runs(registry));

    EXPECT_NE(registry.fallback(), nullptr);
    EXPECT_TRUE(registry.rows().empty());
}

/** @brief Tests that a second registration is refused and keeps the first fallback. */
TEST(Small_TrainerRun, ASecondRegistrationIsRefused)
{
    Apps::RunRegistry registry;
    SA_EXPECT_OK(Apps::register_runs(registry));
    const Apps::RunFunction first = registry.fallback();

    SA_EXPECT_FAILURE(Apps::register_runs(registry), StatusCode::InvalidDescription);
    EXPECT_EQ(registry.fallback(), first);
}
