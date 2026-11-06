/**************************************************************************/
/* test_status.cpp                                                        */
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
 * @file test_status.cpp
 * @brief Tests Status, its macros, the fatal family and the assertions of test_status.hpp.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/in_memory_fatal_sink.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::FatalHandler;
using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::Detail::live_failure_count;

static_assert(sizeof(Status) == sizeof(void*), "a Status is one pointer");
static_assert(std::is_nothrow_default_constructible<Status>::value, "success never throws");
static_assert(std::is_nothrow_copy_constructible<Status>::value, "a copy never throws");
static_assert(std::is_nothrow_move_constructible<Status>::value, "a move never throws");
static_assert(std::is_nothrow_copy_assignable<Status>::value, "an assignment never throws");
static_assert(std::is_nothrow_move_assignable<Status>::value, "an assignment never throws");

namespace
{
    /** @brief Returns @p value and counts the call in @p evaluations. */
    int counted(int& evaluations, int value) noexcept
    {
        ++evaluations;
        return value;
    }

    /** @brief Builds a failure of @p code whose message is @p text. */
    Status failure_of(StatusCode code, std::string_view text) noexcept
    {
        return Status::from_message(code, [&]()
        {
            return std::string(text);
        });
    }

    /** @brief Fails with InvalidDescription when @p extent is not positive. */
    Status check_extent(int extent, int& evaluations) noexcept
    {
        SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                   "extent must be positive; got {}", counted(evaluations, extent));
        return Status();
    }

    /** @brief Forwards check_extent's failure, or sets @p reached_end. */
    Status forward_extent(int extent, int& evaluations, bool& reached_end) noexcept
    {
        SA_TRY(check_extent(extent, evaluations));
        reached_end = true;
        return Status();
    }

    /** @brief Ignores a fatal report; a handler to tell apart from the default one. */
    void record_fatal(const char*, int, std::string_view) noexcept
    {
    }

    /** @brief Writes "recorded: <message>" to stderr, where a death test reads it. */
    void announce_fatal(const char*, int, std::string_view message) noexcept
    {
        std::fputs("recorded: ", stderr);
        std::fwrite(message.data(), 1, message.size(), stderr);
        std::fputs("\n", stderr);
    }

    /**
     * @brief Stops the process when @p condition holds, naming @p extent.
     * @return The line of the check, which a fatal report has to print.
     */
    int stop_when(bool condition, int& evaluations, int extent) noexcept
    {
        constexpr int K_CHECK_LINE = __LINE__ + 1;
        SA_FATAL_IF(condition, "extent {} is refused", counted(evaluations, extent));
        return K_CHECK_LINE;
    }

    /** @brief Throws, standing for an argument whose formatting fails. */
    std::string refused_text()
    {
        throw std::runtime_error("formatting threw");
    }

    /** @brief Stops the process through SA_FATAL whose message cannot be formatted. */
    void stop_with_unformattable_message() noexcept
    {
        SA_FATAL("{}", refused_text());
    }

    /** @brief Installs announce_fatal and stops the process. */
    void stop_under_announcer() noexcept
    {
        SushiAI::set_fatal_handler(&announce_fatal);
        SA_FATAL("stop {}", 3);
    }

#ifndef NDEBUG
    /** @brief Asserts that @p value is positive. */
    void assert_positive(int value) noexcept
    {
        SA_ASSERT(value > 0);
    }
#endif

    /** @brief Lists every enumerator of StatusCode. */
    constexpr std::array<StatusCode, 8> K_STATUS_CODES = {
        StatusCode::Ok, StatusCode::InvalidDescription, StatusCode::FileNotFound,
        StatusCode::InvalidFile, StatusCode::Incompatible, StatusCode::Unsupported,
        StatusCode::OutOfMemory, StatusCode::DeviceFailed};

    /** @brief Runs one case per enumerator of StatusCode. */
    class Small_StatusCodeName : public ::testing::TestWithParam<StatusCode>
    {

    };

    /** @brief Names a Small_StatusCodeName case after the name of its code. */
    std::string code_case_name(const ::testing::TestParamInfo<StatusCode>& info)
    {
        return SushiAI::status_code_name(info.param);
    }
} // namespace

INSTANTIATE_TEST_SUITE_P(, Small_StatusCodeName, ::testing::ValuesIn(K_STATUS_CODES),
                         code_case_name);

/** @brief Tests that the name of every code is not null and is lower case with underscores. */
TEST_P(Small_StatusCodeName, TheNameIsNonNullAndLowerCase)
{
    const char* name = SushiAI::status_code_name(GetParam());

    ASSERT_NE(name, nullptr);
    const std::string_view text(name);
    EXPECT_FALSE(text.empty());
    EXPECT_NE(text, "unknown");
    EXPECT_TRUE(std::all_of(text.begin(), text.end(), [](char letter)
    {
        return (letter >= 'a' && letter <= 'z') || letter == '_';
    }));
}

/** @brief Tests that a value that is no enumerator is named "unknown". */
TEST(Small_Status, AValueThatIsNoCodeIsNamedUnknown)
{
    EXPECT_STREQ(SushiAI::status_code_name(static_cast<StatusCode>(200)), "unknown");
}

/** @brief Tests that from_message keeps the code and the text it is given. */
TEST(Small_Status, FromMessageCarriesTheCodeAndTheText)
{
    const Status status = failure_of(StatusCode::InvalidFile, "the header is 3 bytes short");
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.code(), StatusCode::InvalidFile);
    EXPECT_EQ(status.message(), "the header is 3 bytes short");
}

/**
 * @brief Tests that a success Status has the Ok code and no message, and that making, copying
 *        and moving one, or annotating one, allocates no failure block and composes nothing.
 */
TEST(Small_Status, SuccessPathCreatesNoFailureBlock)
{
    const std::size_t before = live_failure_count();
    int composed = 0;

    const Status made;
    const Status copied(made);
    Status source;
    Status moved(std::move(source));
    Status assigned;
    assigned = copied;
    assigned = std::move(moved);
    const Status annotated = made.annotated([&]()
    {
        ++composed;
        return std::string("prefix");
    });

    EXPECT_EQ(made.code(), StatusCode::Ok);
    EXPECT_TRUE(made.message().empty());
    EXPECT_TRUE(annotated.is_ok());
    EXPECT_TRUE(assigned.is_ok());
    EXPECT_EQ(composed, 0);
    EXPECT_EQ(live_failure_count(), before);
}

/** @brief Tests that SA_FAIL_IF formats its arguments only when its condition holds. */
TEST(Small_Status, FailIfFormatsOnlyWhenTheConditionHolds)
{
    int evaluations = 0;

    SA_EXPECT_OK(check_extent(4, evaluations));
    EXPECT_EQ(evaluations, 0);

    SA_EXPECT_FAILURE(check_extent(-2, evaluations), StatusCode::InvalidDescription,
                      "extent must be positive", "got -2");
    EXPECT_EQ(evaluations, 1);
}

/** @brief Tests that copies of a failure share one block, which the last of them frees. */
TEST(Small_Status, CopiesShareOneBlockAndTheLastOneFreesIt)
{
    const std::size_t before = live_failure_count();

    {
        const Status first = failure_of(StatusCode::Incompatible, "two shapes disagree");
        EXPECT_EQ(live_failure_count(), before + 1);

        const Status second(first);
        Status third;
        third = second;
        EXPECT_EQ(live_failure_count(), before + 1);
        EXPECT_EQ(third.code(), StatusCode::Incompatible);
        EXPECT_EQ(third.message().data(), first.message().data());
    }

    EXPECT_EQ(live_failure_count(), before);
}

/** @brief Tests that a moved-from failed Status keeps its failure. */
TEST(Small_Status, MovedFromStatusKeepsItsFailure)
{
    Status source = failure_of(StatusCode::Unsupported, "no kernel for int64");
    const Status constructed(std::move(source));
    Status assigned;
    assigned = std::move(source);

    SA_EXPECT_FAILURE(source, StatusCode::Unsupported, "no kernel for int64");
    SA_EXPECT_FAILURE(constructed, StatusCode::Unsupported, "no kernel for int64");
    SA_EXPECT_FAILURE(assigned, StatusCode::Unsupported, "no kernel for int64");
}

/** @brief Tests that assigning a Status replaces the failure it holds, and a success frees it. */
TEST(Small_Status, AssignmentReplacesTheHeldFailure)
{
    const std::size_t before = live_failure_count();

    Status status = failure_of(StatusCode::InvalidFile, "first");
    status = failure_of(StatusCode::FileNotFound, "second");
    EXPECT_EQ(live_failure_count(), before + 1);
    SA_EXPECT_FAILURE(status, StatusCode::FileNotFound, "second");

    const Status& alias = status;
    status = alias;
    SA_EXPECT_FAILURE(status, StatusCode::FileNotFound, "second");

    status = Status();
    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(live_failure_count(), before);
}

/** @brief Tests that annotated prefixes the message and keeps the code. */
TEST(Small_Status, AnnotatedPrefixesTheMessageAndKeepsTheCode)
{
    const Status inner = failure_of(StatusCode::InvalidDescription, "stride must be positive");
    const Status outer = SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16);

    EXPECT_EQ(outer.code(), StatusCode::InvalidDescription);
    EXPECT_EQ(outer.message(), "Conv2d(3, 16): stride must be positive");
    EXPECT_EQ(inner.message(), "stride must be positive");
}

/** @brief Tests that from_message whose composer throws yields the fixed OutOfMemory failure. */
TEST(Small_Status, ThrowingComposerYieldsTheStaticOutOfMemoryFailure)
{
    const Status status = Status::from_message(StatusCode::InvalidFile, []() -> std::string
    {
        throw std::runtime_error("composer threw");
    });

    SA_EXPECT_FAILURE(status, StatusCode::OutOfMemory, "message could not be built");
}

/** @brief Tests that annotated whose composer throws returns the status unchanged. */
TEST(Small_Status, ThrowingAnnotationKeepsTheOriginalFailure)
{
    const Status inner = failure_of(StatusCode::InvalidFile, "wrong magic");
    const Status outer = inner.annotated([]() -> std::string
    {
        throw std::runtime_error("composer threw");
    });

    EXPECT_EQ(outer.code(), StatusCode::InvalidFile);
    EXPECT_EQ(outer.message(), "wrong magic");
}

/** @brief Tests that SA_TRY returns a failure and lets a success pass. */
TEST(Small_Status, TryForwardsAFailureAndPassesASuccess)
{
    int evaluations = 0;
    bool reached_end = false;

    SA_EXPECT_FAILURE(forward_extent(0, evaluations, reached_end),
                      StatusCode::InvalidDescription, "got 0");
    EXPECT_FALSE(reached_end);

    SA_EXPECT_OK(forward_extent(5, evaluations, reached_end));
    EXPECT_TRUE(reached_end);
}

/** @brief Tests that a report delivered to the installed sink carries file, line and text. */
TEST(Small_Status, ReportFatalDeliversFileLineAndMessageOnceToTheInstalledSink)
{
    const SushiAI::InMemoryFatalSink sink;

    SushiAI::Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused");

    ASSERT_EQ(sink.reports().size(), 1u);
    EXPECT_EQ(sink.reports()[0].file, "layer.cpp");
    EXPECT_EQ(sink.reports()[0].line, 42);
    EXPECT_EQ(sink.reports()[0].message, "a sealed trace was reused");
}

/** @brief Tests that a null handler installs the default one, which is no earlier handler. */
TEST(Small_Status, NullFatalHandlerInstallsTheDefaultOne)
{
    const FatalHandler original = SushiAI::set_fatal_handler(&record_fatal);
    const FatalHandler recorder = SushiAI::set_fatal_handler(nullptr);
    const FatalHandler standard = SushiAI::set_fatal_handler(original);

    EXPECT_EQ(recorder, &record_fatal);
    ASSERT_NE(standard, nullptr);
    EXPECT_NE(standard, &record_fatal);
}

/** @brief Tests that SA_FATAL_IF formats nothing and returns when its condition is false. */
TEST(Small_Status, FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse)
{
    int evaluations = 0;
    EXPECT_GT(stop_when(false, evaluations, 7), 0);
    EXPECT_EQ(evaluations, 0);
}

#ifdef NDEBUG
/** @brief Tests that SA_ASSERT does not evaluate its condition when NDEBUG is defined. */
TEST(Small_Status, AssertDoesNotEvaluateItsConditionInAnNdebugBuild)
{
    int evaluations = 0;
    SA_ASSERT(counted(evaluations, 0) == 1);
    EXPECT_EQ(evaluations, 0);
}
#endif

/** @brief Tests that from_message with the Ok code stops the process. */
TEST(Medium_StatusFatal, FromMessageWithTheOkCodeIsFatal)
{
    SA_EXPECT_FATAL((void)failure_of(StatusCode::Ok, "nothing failed"),
                    "Status::from_message needs a failure code", "got ok");
}

/** @brief Tests that SA_FATAL_IF reports its message at the file and line of the check. */
TEST(Medium_StatusFatal, FatalIfReportsTheMessageAtTheFileAndLineOfTheCheckAndStops)
{
    int evaluations = 0;
    const int check_line = stop_when(false, evaluations, 7);
    const std::string place = "[test_status.cpp:" + std::to_string(check_line) + "]";

    SA_EXPECT_FATAL((void)stop_when(true, evaluations, 7),
                    "[FATAL] [sushiai.failure]", place, "extent 7 is refused");
}

/** @brief Tests that a fatal check whose message cannot be formatted reports the fixed text. */
TEST(Medium_StatusFatal, FatalCheckWhoseMessageCannotBeFormattedReportsTheFixedMessage)
{
    SA_EXPECT_FATAL(stop_with_unformattable_message(),
                    "[FATAL] [sushiai.failure]", "[test_status.cpp:",
                    "a fatal check failed", "its message could not be composed");
}

/** @brief Tests that the installed handler runs before the process stops. */
TEST(Medium_StatusFatal, ReplacedFatalHandlerRunsBeforeTheStop)
{
    SA_EXPECT_FATAL(stop_under_announcer(), "recorded: stop 3");
}

#ifndef NDEBUG
/** @brief Tests that SA_ASSERT returns on a true condition and stops on a false one. */
TEST(Medium_StatusFatal, AssertStopsOnAFalseConditionInADebugBuild)
{
    assert_positive(1);
    SA_EXPECT_FATAL(assert_positive(0), "assertion failed: value > 0");
}
#endif
