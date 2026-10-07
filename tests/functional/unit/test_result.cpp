/**************************************************************************/
/* test_result.cpp                                                        */
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
 * @file test_result.cpp
 * @brief Tests Result, its reference form and SA_TRY_ASSIGN.
 * @author Mustafa Garip
 */

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include "../../common/test_status.hpp"

using SushiAI::Result;
using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::Detail::live_failure_count;

static_assert(!std::is_default_constructible<Result<int>>::value, "a Result needs a state");
static_assert(!std::is_default_constructible<Result<int&>>::value, "a Result needs a state");
static_assert(std::is_nothrow_constructible<Result<int>, int>::value, "holding an int is free");
static_assert(std::is_nothrow_constructible<Result<int>, Status>::value, "a failure never throws");
static_assert(std::is_nothrow_move_constructible<Result<int>>::value, "a move never throws");
static_assert(sizeof(Result<int>) <= 2 * sizeof(void*), "an int Result is stored in place");
static_assert(sizeof(Result<int&>) == 2 * sizeof(void*), "a reference Result is two pointers");
static_assert(!std::is_constructible<Result<const int&>, int&&>::value, "no temporary referent");
static_assert(!std::is_constructible<Result<int&>, int&&>::value, "no temporary referent");

namespace
{
    /** @brief Builds an InvalidDescription failure whose message is @p text. */
    Status description_failure(std::string_view text) noexcept
    {
        return Status::from_message(StatusCode::InvalidDescription, [&]()
        {
            return std::string(text);
        });
    }

    /** @brief Returns @p extent, or a failure when it is not positive. */
    Result<int> checked_extent(int extent) noexcept
    {
        SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                   "extent must be positive; got {}", extent);
        return extent;
    }

    /** @brief Returns twice the checked extent, or checked_extent's failure. */
    Result<int> doubled_extent(int extent) noexcept
    {
        SA_TRY_ASSIGN(const int checked, checked_extent(extent));
        return checked * 2;
    }

    /** @brief Returns an owned copy of the checked extent, or checked_extent's failure. */
    Result<std::unique_ptr<int>> boxed_extent(int extent) noexcept
    {
        SA_TRY(checked_extent(extent));
        return std::make_unique<int>(extent);
    }

    /** @brief Returns the value boxed_extent owned, taken through SA_TRY_ASSIGN. */
    Result<int> unboxed_extent(int extent) noexcept
    {
        SA_TRY_ASSIGN(const std::unique_ptr<int> box, boxed_extent(extent));
        return *box;
    }

    /** @brief Forwards the failure read through status() of a temporary, or sets @p passed. */
    Status forward_temporary_status(int extent, bool& passed) noexcept
    {
        SA_TRY(checked_extent(extent).status());
        passed = true;
        return Status();
    }

    /** @brief Returns the slot at @p index of @p slots, or a failure outside two slots. */
    Result<int&> slot_at(int (&slots)[2], int index) noexcept
    {
        SA_FAIL_IF(index < 0 || index > 1, StatusCode::InvalidDescription,
                   "slot {} is not one of 2", index);
        return slots[index];
    }

    /** @brief Adds one to the slot at @p index, or returns slot_at's failure. */
    Status bump_slot(int (&slots)[2], int index) noexcept
    {
        SA_TRY_ASSIGN(int& slot, slot_at(slots, index));
        ++slot;
        return Status();
    }

} // namespace

TEST(Unit_Result, HoldsAValueWithASuccessStatus)
{
    Result<int> result(7);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.status().is_ok());
    EXPECT_EQ(result.value(), 7);

    result.value() = 9;
    const Result<int>& view = result;
    EXPECT_EQ(view.value(), 9);
}

TEST(Unit_Result, HoldsAFailureAndNoValue)
{
    const Result<int> result(description_failure("rank must be 2"));
    EXPECT_FALSE(result.has_value());
    SA_EXPECT_FAILURE(result, StatusCode::InvalidDescription, "rank must be 2");
    SA_EXPECT_FAILURE(result.status(), StatusCode::InvalidDescription, "rank must be 2");
}

TEST(Unit_Result, ReturnStatementsConvertFromAValueAndFromAStatus)
{
    SA_ASSERT_OK_AND_ASSIGN(const int extent, checked_extent(5));
    EXPECT_EQ(extent, 5);

    SA_EXPECT_FAILURE(checked_extent(-1), StatusCode::InvalidDescription,
                      "extent must be positive", "got -1");
}

// A Result of int stores its value in place, so an unchanged block count means no allocation.
TEST(Unit_Result, SuccessPathCreatesNoFailureBlock)
{
    const std::size_t before = live_failure_count();

    const Result<int> made = checked_extent(3);
    const Result<int> copied(made);
    Result<int> source(4);
    const Result<int> moved(std::move(source));
    SA_ASSERT_OK_AND_ASSIGN(const int doubled, doubled_extent(3));

    EXPECT_EQ(copied.value() + moved.value() + doubled, 13);
    EXPECT_EQ(live_failure_count(), before);
}

TEST(Unit_Result, RvalueValueMovesAMoveOnlyValueOut)
{
    Result<std::unique_ptr<int>> result = boxed_extent(11);
    ASSERT_TRUE(result.has_value());

    const std::unique_ptr<int> taken = std::move(result).value();
    ASSERT_NE(taken, nullptr);
    EXPECT_EQ(*taken, 11);
}

TEST(Unit_Result, MovedFromValueResultStillHoldsAMovedFromValue)
{
    Result<std::unique_ptr<int>> source = boxed_extent(2);
    const Result<std::unique_ptr<int>> target(std::move(source));

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ(*target.value(), 2);
    EXPECT_TRUE(source.has_value());
    EXPECT_TRUE(source.status().is_ok());
    EXPECT_TRUE(source.value() == nullptr);
}

TEST(Unit_Result, MovedFromFailureResultKeepsItsFailure)
{
    Result<int> source(description_failure("layout must be NHWC"));
    const Result<int> target(std::move(source));

    SA_EXPECT_FAILURE(target, StatusCode::InvalidDescription, "layout must be NHWC");
    EXPECT_FALSE(source.has_value());
    SA_EXPECT_FAILURE(source, StatusCode::InvalidDescription, "layout must be NHWC");
}

TEST(Unit_Result, CopyAndAssignmentCarryTheHeldState)
{
    const Result<int> failed(description_failure("no such key"));
    Result<int> result(1);

    result = failed;
    EXPECT_FALSE(result.has_value());
    SA_EXPECT_FAILURE(result, StatusCode::InvalidDescription, "no such key");

    result = Result<int>(8);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.status().is_ok());
    EXPECT_EQ(result.value(), 8);
}

TEST(Unit_Result, TryAssignBindsTheValueOrReturnsTheFailure)
{
    SA_ASSERT_OK_AND_ASSIGN(const int doubled, doubled_extent(21));
    EXPECT_EQ(doubled, 42);

    SA_EXPECT_FAILURE(doubled_extent(0), StatusCode::InvalidDescription, "got 0");
}

TEST(Unit_Result, TryAssignMovesFromATemporaryResult)
{
    SA_ASSERT_OK_AND_ASSIGN(const int value, unboxed_extent(6));
    EXPECT_EQ(value, 6);

    SA_EXPECT_FAILURE(unboxed_extent(-6), StatusCode::InvalidDescription, "got -6");
}

TEST(Unit_Result, ReferenceFormRefersToTheCallersObject)
{
    int slots[2] = {10, 20};

    const Result<int&> found = slot_at(slots, 1);
    ASSERT_TRUE(found.has_value());
    EXPECT_TRUE(found.status().is_ok());
    EXPECT_EQ(&found.value(), &slots[1]);

    found.value() = 25;
    EXPECT_EQ(slots[1], 25);

    const int fixed = 3;
    const Result<const int&> constant(fixed);
    EXPECT_EQ(&constant.value(), &fixed);
}

TEST(Unit_Result, TryReadsAStatusReferenceIntoATemporaryResult)
{
    const std::size_t before = live_failure_count();
    bool passed = false;

    SA_EXPECT_FAILURE(forward_temporary_status(-4, passed), StatusCode::InvalidDescription,
                      "extent must be positive", "got -4");
    EXPECT_FALSE(passed);
    EXPECT_EQ(live_failure_count(), before);

    SA_EXPECT_OK(forward_temporary_status(4, passed));
    EXPECT_TRUE(passed);
    EXPECT_EQ(live_failure_count(), before);
}

TEST(Unit_Result, MovedFromReferenceResultKeepsItsState)
{
    int slots[2] = {10, 20};

    Result<int&> found = slot_at(slots, 0);
    const Result<int&> copied(found);
    const Result<int&> taken(std::move(found));
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(&found.value(), &slots[0]);
    EXPECT_EQ(&copied.value(), &slots[0]);
    EXPECT_EQ(&taken.value(), &slots[0]);
    EXPECT_TRUE(taken.status().is_ok());

    Result<int&> missing = slot_at(slots, 2);
    const Result<int&> forwarded(std::move(missing));
    EXPECT_FALSE(missing.has_value());
    EXPECT_FALSE(forwarded.has_value());
    SA_EXPECT_FAILURE(missing, StatusCode::InvalidDescription, "slot 2 is not one of 2");
    SA_EXPECT_FAILURE(forwarded, StatusCode::InvalidDescription, "slot 2 is not one of 2");

    Result<int&> assigned = slot_at(slots, 1);
    assigned = forwarded;
    SA_EXPECT_FAILURE(assigned, StatusCode::InvalidDescription, "slot 2 is not one of 2");
    assigned = taken;
    ASSERT_TRUE(assigned.has_value());
    EXPECT_TRUE(assigned.status().is_ok());
    EXPECT_EQ(&assigned.value(), &slots[0]);
}

TEST(Unit_Result, ReferenceFormHoldsAFailure)
{
    int slots[2] = {10, 20};

    const Result<int&> missing = slot_at(slots, 2);
    EXPECT_FALSE(missing.has_value());
    SA_EXPECT_FAILURE(missing, StatusCode::InvalidDescription, "slot 2 is not one of 2");
}

TEST(Unit_Result, TryAssignBindsAReferenceFromTheReferenceForm)
{
    int slots[2] = {10, 20};

    SA_EXPECT_OK(bump_slot(slots, 0));
    EXPECT_EQ(slots[0], 11);

    SA_EXPECT_FAILURE(bump_slot(slots, 5), StatusCode::InvalidDescription, "slot 5");
    EXPECT_EQ(slots[1], 20);
}

TEST(Unit_Result, ValueOnAFailureIsFatalAndPrintsTheFailure)
{
    const Result<int> failed(description_failure("rank must be 2"));
    SA_EXPECT_FATAL((void)failed.value(),
                    "Result::value() was called on a failure", "rank must be 2");

    const Result<int&> missing(description_failure("no such slot"));
    SA_EXPECT_FATAL((void)missing.value(),
                    "Result::value() was called on a failure", "no such slot");
}

TEST(Unit_Result, BuildingFromASuccessStatusIsFatal)
{
    SA_EXPECT_FATAL((void)Result<int>(Status()), "a Result was built from a success Status");
    SA_EXPECT_FATAL((void)Result<int&>(Status()), "a Result was built from a success Status");
}
