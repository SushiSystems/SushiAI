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
 * @file test_status.cpp
 * @brief Tests Status, its macros, the fatal family and the assertions of test_status.hpp.
 * @author Mustafa Garip
 */

#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/status.hpp>
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

    /** @brief Holds what the recording fatal handler was last given. */
    struct FatalRecord
    {
        /** @brief Holds how many reports arrived. */
        int calls = 0;

        /** @brief Holds the file of the last report. */
        std::string file;

        /** @brief Holds the line of the last report. */
        int line = 0;

        /** @brief Holds the message of the last report. */
        std::string message;
    };

    /** @brief Returns the record the recording handler writes. */
    FatalRecord& fatal_record()
    {
        static FatalRecord record;
        return record;
    }

    /** @brief Stores one fatal report in fatal_record(). */
    void record_fatal(const char* file, int line, std::string_view message) noexcept
    {
        FatalRecord& record = fatal_record();
        ++record.calls;
        record.file = file;
        record.line = line;
        record.message = std::string(message);
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

    /** @brief Stops the process through a fatal check whose message composer throws. */
    void stop_with_throwing_composer() noexcept
    {
        SushiAI::Detail::fail_fast_composed(__FILE__, __LINE__, []() -> std::string
        {
            throw std::runtime_error("composer threw");
        });
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

} // namespace

TEST(Small_Status, DefaultIsSuccessWithNoMessage)
{
    const Status status;
    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(status.code(), StatusCode::Ok);
    EXPECT_TRUE(status.message().empty());
    SA_EXPECT_OK(status);
}

TEST(Small_Status, EveryCodeHasALowerCaseName)
{
    using SushiAI::status_code_name;
    EXPECT_STREQ(status_code_name(StatusCode::Ok), "ok");
    EXPECT_STREQ(status_code_name(StatusCode::InvalidDescription), "invalid_description");
    EXPECT_STREQ(status_code_name(StatusCode::FileNotFound), "file_not_found");
    EXPECT_STREQ(status_code_name(StatusCode::InvalidFile), "invalid_file");
    EXPECT_STREQ(status_code_name(StatusCode::Incompatible), "incompatible");
    EXPECT_STREQ(status_code_name(StatusCode::Unsupported), "unsupported");
    EXPECT_STREQ(status_code_name(StatusCode::OutOfMemory), "out_of_memory");
    EXPECT_STREQ(status_code_name(StatusCode::DeviceFailed), "device_failed");
    EXPECT_STREQ(status_code_name(static_cast<StatusCode>(200)), "unknown");
}

TEST(Small_Status, FromMessageCarriesTheCodeAndTheText)
{
    const Status status = failure_of(StatusCode::InvalidFile, "the header is 3 bytes short");
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.code(), StatusCode::InvalidFile);
    EXPECT_EQ(status.message(), "the header is 3 bytes short");
}

// The failure block is a Status's only allocation, so an unchanged block count means none.
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

    EXPECT_TRUE(annotated.is_ok());
    EXPECT_TRUE(assigned.is_ok());
    EXPECT_EQ(composed, 0);
    EXPECT_EQ(live_failure_count(), before);
}

TEST(Small_Status, FailIfFormatsOnlyWhenTheConditionHolds)
{
    const std::size_t before = live_failure_count();
    int evaluations = 0;

    SA_EXPECT_OK(check_extent(4, evaluations));
    EXPECT_EQ(evaluations, 0);
    EXPECT_EQ(live_failure_count(), before);

    SA_EXPECT_FAILURE(check_extent(-2, evaluations), StatusCode::InvalidDescription,
                      "extent must be positive", "got -2");
    EXPECT_EQ(evaluations, 1);
    EXPECT_EQ(live_failure_count(), before);
}

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

TEST(Small_Status, AnnotatedPrefixesTheMessageAndKeepsTheCode)
{
    const Status inner = failure_of(StatusCode::InvalidDescription, "stride must be positive");
    const Status outer = SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16);

    EXPECT_EQ(outer.code(), StatusCode::InvalidDescription);
    EXPECT_EQ(outer.message(), "Conv2d(3, 16): stride must be positive");
    EXPECT_EQ(inner.message(), "stride must be positive");
}

TEST(Small_Status, ThrowingComposerYieldsTheStaticOutOfMemoryFailure)
{
    const std::size_t before = live_failure_count();

    const Status status = Status::from_message(StatusCode::InvalidFile, []() -> std::string
    {
        throw std::runtime_error("composer threw");
    });

    SA_EXPECT_FAILURE(status, StatusCode::OutOfMemory, "message could not be built");
    EXPECT_EQ(live_failure_count(), before);
}

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

TEST(Small_Status, FromMessageWithTheOkCodeIsFatal)
{
    SA_EXPECT_FATAL((void)failure_of(StatusCode::Ok, "nothing failed"),
                    "Status::from_message needs a failure code", "got ok");
}

TEST(Small_Status, StatusNamesRejectsSuccessAWrongCodeAndAMissingNeedle)
{
    using SushiAI::Test::status_names;
    const Status status = failure_of(StatusCode::InvalidFile, "wrong magic 0x1F");

    EXPECT_TRUE(status_names(status, StatusCode::InvalidFile, {"wrong magic", "0x1F"}));
    EXPECT_TRUE(status_names(status, StatusCode::InvalidFile, {}));
    EXPECT_FALSE(status_names(Status(), StatusCode::InvalidFile, {}));
    EXPECT_FALSE(status_names(status, StatusCode::FileNotFound, {"wrong magic"}));
    EXPECT_FALSE(status_names(status, StatusCode::InvalidFile, {"wrong version"}));
    EXPECT_FALSE(SushiAI::Test::status_is_ok(status));
}

TEST(Small_Status, DeathPatternEscapesAndJoinsItsNeedles)
{
    using SushiAI::Test::death_pattern;
    EXPECT_EQ(death_pattern({"plain words"}), "plain words");
    EXPECT_EQ(death_pattern({"a.b", "[5, 3]"}), "a\\.b.*\\[5, 3\\]");
    EXPECT_EQ(death_pattern({"f(x)*2", "{} | ^$ + ? \\"}),
              "f\\(x\\)\\*2.*\\{\\} \\| \\^\\$ \\+ \\? \\\\");
}

TEST(Small_Status, FatalHandlerIsReplacedByARecorderAndRestored)
{
    fatal_record() = FatalRecord();

    const FatalHandler original = SushiAI::set_fatal_handler(&record_fatal);
    ASSERT_NE(original, nullptr);

    SushiAI::Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused");
    const FatalHandler replaced = SushiAI::set_fatal_handler(original);

    EXPECT_EQ(replaced, &record_fatal);
    EXPECT_EQ(fatal_record().calls, 1);
    EXPECT_EQ(fatal_record().file, "layer.cpp");
    EXPECT_EQ(fatal_record().line, 42);
    EXPECT_EQ(fatal_record().message, "a sealed trace was reused");

    EXPECT_EQ(SushiAI::set_fatal_handler(original), original);
}

TEST(Small_Status, NullFatalHandlerInstallsTheDefaultOne)
{
    const FatalHandler original = SushiAI::set_fatal_handler(&record_fatal);
    const FatalHandler recorder = SushiAI::set_fatal_handler(nullptr);
    const FatalHandler standard = SushiAI::set_fatal_handler(original);

    EXPECT_EQ(recorder, &record_fatal);
    ASSERT_NE(standard, nullptr);
    EXPECT_NE(standard, &record_fatal);
}

TEST(Small_Status, FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse)
{
    int evaluations = 0;
    EXPECT_GT(stop_when(false, evaluations, 7), 0);
    EXPECT_EQ(evaluations, 0);
}

TEST(Small_Status, FatalIfPrintsTheMessageThenTheFileAndTheLineAndStops)
{
    int evaluations = 0;
    const int check_line = stop_when(false, evaluations, 7);
    const std::string place = "test_status.cpp:" + std::to_string(check_line) + ")";

    SA_EXPECT_FATAL((void)stop_when(true, evaluations, 7),
                    "sushiai: fatal: extent 7 is refused", place);
}

TEST(Small_Status, FatalCheckWhoseComposerThrowsReportsTheFixedMessage)
{
    SA_EXPECT_FATAL(stop_with_throwing_composer(),
                    "sushiai: fatal:", "its message could not be composed", "test_status.cpp:");
}

TEST(Small_Status, ReplacedFatalHandlerRunsBeforeTheStop)
{
    SA_EXPECT_FATAL(stop_under_announcer(), "recorded: stop 3");
}

TEST(Small_Status, AssertChecksInDebugBuildsOnly)
{
#ifdef NDEBUG
    int evaluations = 0;
    SA_ASSERT(counted(evaluations, 0) == 1);
    EXPECT_EQ(evaluations, 0);
#else
    assert_positive(1);
    SA_EXPECT_FATAL(assert_positive(0), "assertion failed: value > 0");
#endif
}
