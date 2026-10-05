/****************************************************************/
/* test_environment.hpp                                         */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file test_environment.hpp
 * @brief The one RuntimeContext the whole test binary shares.
 *
 * A RuntimeContext starts a worker pool, probes the topology and builds SYCL
 * queues. Doing that per test would dominate the suite's runtime and, worse,
 * would repeatedly tear down and restart the thread pool while other work is
 * settling. One process-global context in a GTest Environment, and a fresh
 * Engine per test so no test inherits another's task graph or RNG offset -
 * exactly the split SushiBLAS's own suite uses.
 */

#include <memory>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/SushiRuntime.h>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @class TestEnvironment
         * @brief Owns the process-global RuntimeContext for the whole binary.
         */
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

                /**
                 * @brief The shared context.
                 * @return A reference valid for the whole run.
                 */
                static SushiRuntime::Execution::RuntimeContext& context()
                {
                    return *context_;
                }

            private:
                static inline std::unique_ptr<SushiRuntime::Execution::RuntimeContext> context_;
        };

    } // namespace Test
} // namespace SushiAI
