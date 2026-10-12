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
 * @brief Declares the GTest environment that owns the one Runtime a test binary shares.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_environment.hpp".
 */

#pragma once

#include <optional>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Owns the process-global Runtime for the whole binary. */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Creates the runtime before the first test. */
                void SetUp() override
                {
                    Result<SushiRuntime::Runtime> created = Detail::create_runtime();
                    ASSERT_TRUE(created.has_value()) << created.status().message();
                    runtime_.emplace(std::move(created).value());
                }

                /** @brief Destroys the runtime after the last test. */
                void TearDown() override
                {
                    runtime_.reset();
                }

                /** @brief Returns the shared runtime, valid for the whole run. */
                static SushiRuntime::Runtime& runtime()
                {
                    return *runtime_;
                }

            private:
                /** @brief Holds the runtime between SetUp() and TearDown(). */
                static inline std::optional<SushiRuntime::Runtime> runtime_;

        };

    } // namespace Test
} // namespace SushiAI
