/**************************************************************************/
/* test_in_memory_fatal_sink.cpp                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_in_memory_fatal_sink.cpp
 * @brief Tests that InMemoryFatalSink keeps fatal reports and restores the handler it replaced.
 * @author Mustafa Garip
 */

#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/in_memory_fatal_sink.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Counts the reports a test handler received. */
    int g_seen = 0;

    /** @brief Counts the report and returns. */
    void count_report(const char*, int, std::string_view) noexcept
    {
        ++g_seen;
    }
} // namespace

/** @brief Tests that a report delivered to the installed handler is kept with its origin. */
TEST(Small_InMemoryFatalSink, KeepsAReportWithItsFileLineAndMessage)
{
    const InMemoryFatalSink sink;

    Detail::report_fatal("some_file.cpp", 42, "the broken precondition");

    ASSERT_EQ(sink.reports().size(), 1u);
    EXPECT_EQ(sink.reports()[0].file, "some_file.cpp");
    EXPECT_EQ(sink.reports()[0].line, 42);
    EXPECT_EQ(sink.reports()[0].message, "the broken precondition");
    EXPECT_EQ(sink.dropped(), 0u);
}

/** @brief Tests that reports are kept in the order they arrived. */
TEST(Small_InMemoryFatalSink, KeepsReportsInArrivalOrder)
{
    const InMemoryFatalSink sink;

    Detail::report_fatal("a.cpp", 1, "first");
    Detail::report_fatal("b.cpp", 2, "second");

    ASSERT_EQ(sink.reports().size(), 2u);
    EXPECT_EQ(sink.reports()[0].message, "first");
    EXPECT_EQ(sink.reports()[1].message, "second");
}

/** @brief Tests that the destructor puts back the handler that was installed before. */
TEST(Small_InMemoryFatalSink, TheDestructorRestoresThePreviousHandler)
{
    g_seen = 0;
    const FatalHandler before = set_fatal_handler(&count_report);

    {
        const InMemoryFatalSink sink;
        Detail::report_fatal("a.cpp", 1, "to the sink");
        EXPECT_EQ(g_seen, 0);
    }

    Detail::report_fatal("a.cpp", 2, "to the counter");
    EXPECT_EQ(g_seen, 1);
    static_cast<void>(set_fatal_handler(before));
}

/** @brief Tests that the newest sink receives the reports and the older one resumes after it. */
TEST(Small_InMemoryFatalSink, TheNewestSinkReceivesTheReportsAndTheOlderOneResumes)
{
    const InMemoryFatalSink outer;

    {
        const InMemoryFatalSink inner;
        Detail::report_fatal("a.cpp", 1, "to the inner");
        EXPECT_EQ(inner.reports().size(), 1u);
        EXPECT_TRUE(outer.reports().empty());
    }

    Detail::report_fatal("a.cpp", 2, "to the outer");
    ASSERT_EQ(outer.reports().size(), 1u);
    EXPECT_EQ(outer.reports()[0].message, "to the outer");
}
