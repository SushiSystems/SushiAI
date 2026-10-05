/****************************************************************/
/* test_environment.hpp                                         */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_environment.hpp
 * @brief Declares the GTest environment that owns the one RuntimeContext a test binary shares.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_environment.hpp".
 */

#pragma once

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
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
