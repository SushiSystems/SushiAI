/**************************************************************************/
/* test_programs_log_category.cpp                                         */
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
 * @file test_programs_log_category.cpp
 * @brief Tests programs_log_category.hpp: the project and the module of the programs' records.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>

#include "programs_log_category.hpp"

/** @brief Tests that the programs log under sushiai.programs. */
TEST(Small_ProgramsLogCategory, NamesSushiAIAndThePrograms)
{
    static_assert(SushiAI::Apps::K_PROGRAMS_LOG_CATEGORY.project().name() == "sushiai",
                  "project name");
    static_assert(SushiAI::Apps::K_PROGRAMS_LOG_CATEGORY.module() == "programs", "module name");
    SUCCEED();
}
