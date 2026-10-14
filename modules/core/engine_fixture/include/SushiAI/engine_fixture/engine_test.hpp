/**************************************************************************/
/* engine_test.hpp                                                        */
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
 * @file engine_test.hpp
 * @brief Declares EngineTest, the fixture that gives each test an Engine on the shared Runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Provides a fresh Engine per test, on the shared Runtime. */
        class EngineTest : public ::testing::Test
        {
            protected:
                /** @brief Builds the test's Engine on the shared runtime. */
                void SetUp() override
                {
                    engine = std::make_unique<SushiBLAS::Engine>(TestEnvironment::runtime());
                }

                /** @brief Destroys the test's Engine. */
                void TearDown() override
                {
                    engine.reset();
                }

                /** @brief Flushes anything recorded and waits for the device. */
                void flush()
                {
                    engine->execute().value();
                    TestEnvironment::runtime().wait().value();
                }

                /** @brief Holds the Engine the test records on. */
                std::unique_ptr<SushiBLAS::Engine> engine;
        };

    } // namespace Test
} // namespace SushiAI
