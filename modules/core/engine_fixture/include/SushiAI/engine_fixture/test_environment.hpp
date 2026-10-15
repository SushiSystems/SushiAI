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
 * @brief Declares the GTest environment that owns the one Runtime a test program shares.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
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
                /** @brief Destroys the runtime after the last test, when one was created. */
                void TearDown() override
                {
                    runtime_.reset();
                }

                /**
                 * @brief Returns the shared runtime, creating it on the first call.
                 *
                 * A test that never calls it opens no device and starts no worker. A runtime that
                 * cannot be created ends the process with its message.
                 */
                static SushiRuntime::Runtime& runtime()
                {
                    if (!runtime_)
                    {
                        Result<SushiRuntime::Runtime> created = Detail::create_runtime();
                        if (!created.has_value())
                        {
                            const std::string_view message = created.status().message();
                            std::fprintf(stderr, "cannot create the runtime: %.*s\n",
                                         static_cast<int>(message.size()), message.data());
                            std::abort();
                        }
                        runtime_.emplace(std::move(created).value());
                    }
                    return *runtime_;
                }

            private:
                /** @brief Holds the runtime from the first runtime() call to TearDown(). */
                static inline std::optional<SushiRuntime::Runtime> runtime_;

        };

    } // namespace Test
} // namespace SushiAI
