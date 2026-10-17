/**************************************************************************/
/* test_sushiai_log_project.cpp                                           */
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
 * @file test_sushiai_log_project.cpp
 * @brief Tests sushiai_log_project.hpp: the name and the level variable of SushiAI's log project.
 * @author Mustafa Garip
 */

#include <SushiAI/log/sushiai_log_project.hpp>

#include <gtest/gtest.h>

/** @brief Tests that SushiAI logs under the project sushiai, whose level SA_LOG_LEVEL sets. */
TEST(Small_SushiAILogProject, NamesSushiAIAndItsLevelVariable)
{
    static_assert(SushiAI::K_SUSHIAI_LOG_PROJECT.name() == "sushiai", "project name");
    static_assert(SushiAI::K_SUSHIAI_LOG_PROJECT.level_variable() == "SA_LOG_LEVEL",
                  "level variable");
    SUCCEED();
}
