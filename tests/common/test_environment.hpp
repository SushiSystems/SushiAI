/**************************************************************************/
/* test_environment.hpp                                                   */
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
 * @file test_environment.hpp
 * @brief Declares the GTest environment that owns the one RuntimeContext a test binary shares.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_environment.hpp".
 */

#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Owns the process-global RuntimeContext for the whole binary. */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Builds the runtime context before the first test. */
                void SetUp() override
                {
                    context_ = std::make_unique<SushiRuntime::Execution::RuntimeContext>();
                }

                /** @brief Tears the runtime context down after the last test. */
                void TearDown() override
                {
                    context_.reset();
                }

                /** @brief Returns the shared context, valid for the whole run. */
                static SushiRuntime::Execution::RuntimeContext& context()
                {
                    return *context_;
                }

            private:
                /** @brief Holds the context between SetUp() and TearDown(). */
                static inline std::unique_ptr<SushiRuntime::Execution::RuntimeContext> context_;
        };

    } // namespace Test
} // namespace SushiAI
