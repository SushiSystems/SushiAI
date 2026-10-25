/**************************************************************************/
/* test_run_progress.cpp                                                  */
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

/**
 * @file test_run_progress.cpp
 * @brief Tests the guarded delivery of a progress report to a run callback.
 * @author Mustafa Garip
 */

#include <functional>
#include <stdexcept>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/run_progress.hpp>

using namespace SushiAI;
using Vision::Training::notify_guarded;
using Vision::Training::report_run_progress;

namespace
{
    /** @brief Holds a progress of the test's own, as an engine would declare one. */
    struct Progress
    {
        int update = 0; /**< Update the progress describes. */
    };
} // namespace

/** @brief Tests that an empty callback succeeds without a call. */
TEST(Small_RunProgress, AnEmptyCallbackSucceeds)
{
    const std::function<void(const Progress&)> callback;

    SA_EXPECT_OK(report_run_progress(callback, Progress{1}));
    SA_EXPECT_OK(notify_guarded(std::function<void()>()));
}

/** @brief Tests that the callback receives the progress it is handed, once. */
TEST(Small_RunProgress, TheCallbackReceivesTheProgressOnce)
{
    int calls = 0;
    int seen = 0;
    const std::function<void(const Progress&)> callback = [&](const Progress& progress)
    {
        ++calls;
        seen = progress.update;
    };

    SA_EXPECT_OK(report_run_progress(callback, Progress{7}));
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(seen, 7);
}

/** @brief Tests that a throw of the callback comes back as a failure instead of unwinding. */
TEST(Small_RunProgress, AThrowOfTheCallbackBecomesAFailure)
{
    const std::function<void(const Progress&)> callback = [](const Progress&)
    {
        throw std::runtime_error("callback refused update");
    };

    const Status status = report_run_progress(callback, Progress{1});
    EXPECT_FALSE(status.is_ok());
    EXPECT_NE(status.message().find("callback refused update"), std::string_view::npos);
}
