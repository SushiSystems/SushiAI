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
#include <SushiAI/failure/in_memory_failure_block_counter.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using SushiAI::Result;
using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::InMemoryFailureBlockCounter;

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

/** @brief Tests that a Result built from a value holds it with an ok status. */
TEST(Small_Result, HoldsAValueWithASuccessStatus)
{
    const Result<int> result(7);

    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.status().is_ok());
    EXPECT_EQ(result.value(), 7);
}

/** @brief Tests that a write through value() reaches the Result the const view reads. */
TEST(Small_Result, AWriteThroughValueIsReadBackThroughTheConstView)
{
    Result<int> result(7);

    result.value() = 9;

    const Result<int>& view = result;
    EXPECT_EQ(view.value(), 9);
}

/** @brief Tests that a Result of either form built from a failure holds no value. */
TEST(Small_Result, HoldsAFailureAndNoValue)
{
    const Result<int> result(description_failure("rank must be 2"));
    EXPECT_FALSE(result.has_value());
    SA_EXPECT_FAILURE(result, StatusCode::InvalidDescription, "rank must be 2");
    SA_EXPECT_FAILURE(result.status(), StatusCode::InvalidDescription, "rank must be 2");

    int slots[2] = {10, 20};
    const Result<int&> missing = slot_at(slots, 2);
    EXPECT_FALSE(missing.has_value());
    SA_EXPECT_FAILURE(missing, StatusCode::InvalidDescription, "slot 2 is not one of 2");
}

/** @brief Tests that a function returning Result<int> converts implicitly from an int. */
TEST(Small_Result, AReturnStatementConvertsFromAValue)
{
    SA_ASSERT_OK_AND_ASSIGN(const int extent, checked_extent(5));

    EXPECT_EQ(extent, 5);
}

/** @brief Tests that a function returning Result<int> converts implicitly from a failed Status. */
TEST(Small_Result, AReturnStatementConvertsFromAFailedStatus)
{
    SA_EXPECT_FAILURE(checked_extent(-1), StatusCode::InvalidDescription,
                      "extent must be positive", "got -1");
}

/** @brief Tests that a Result holding a value allocates no failure block. */
TEST(Small_Result, SuccessPathCreatesNoFailureBlock)
{
    const InMemoryFailureBlockCounter counter;

    const Result<int> made = checked_extent(3);
    const Result<int> copied(made);
    Result<int> source(4);
    const Result<int> moved(std::move(source));
    SA_ASSERT_OK_AND_ASSIGN(const int doubled, doubled_extent(3));

    EXPECT_EQ(copied.value() + moved.value() + doubled, 13);
    EXPECT_EQ(counter.live(), 0u);
}

/** @brief Tests that std::move(result).value() moves a move-only value out. */
TEST(Small_Result, RvalueValueMovesAMoveOnlyValueOut)
{
    Result<std::unique_ptr<int>> result = boxed_extent(11);
    ASSERT_TRUE(result.has_value());

    const std::unique_ptr<int> taken = std::move(result).value();
    ASSERT_NE(taken, nullptr);
    EXPECT_EQ(*taken, 11);
}

/** @brief Tests that moving a Result<T> transfers its value to the target. */
TEST(Small_Result, MovingAValueResultTransfersItsValue)
{
    Result<std::unique_ptr<int>> source = boxed_extent(2);
    const Result<std::unique_ptr<int>> target(std::move(source));

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ(*target.value(), 2);
}

/** @brief Tests that a moved-from failed Result keeps its failure. */
TEST(Small_Result, MovedFromFailureResultKeepsItsFailure)
{
    Result<int> source(description_failure("layout must be NHWC"));
    const Result<int> target(std::move(source));

    SA_EXPECT_FAILURE(target, StatusCode::InvalidDescription, "layout must be NHWC");
    EXPECT_FALSE(source.has_value());
    SA_EXPECT_FAILURE(source, StatusCode::InvalidDescription, "layout must be NHWC");
}

/** @brief Tests that copy-assigning a failure over a value leaves the failure. */
TEST(Small_Result, CopyAssigningAFailureOverAValueLeavesTheFailure)
{
    const Result<int> failed(description_failure("no such key"));
    Result<int> result(1);

    result = failed;

    EXPECT_FALSE(result.has_value());
    SA_EXPECT_FAILURE(result, StatusCode::InvalidDescription, "no such key");
}

/** @brief Tests that assigning a value over a failure leaves the value. */
TEST(Small_Result, AssigningAValueOverAFailureLeavesTheValue)
{
    Result<int> result(description_failure("no such key"));

    result = Result<int>(8);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.status().is_ok());
    EXPECT_EQ(result.value(), 8);
}

/** @brief Tests that SA_TRY_ASSIGN binds the value of a success or returns its failure. */
TEST(Small_Result, TryAssignBindsTheValueOrReturnsTheFailure)
{
    SA_ASSERT_OK_AND_ASSIGN(const int doubled, doubled_extent(21));
    EXPECT_EQ(doubled, 42);

    SA_EXPECT_FAILURE(doubled_extent(0), StatusCode::InvalidDescription, "got 0");
    SA_EXPECT_FAILURE(unboxed_extent(-6), StatusCode::InvalidDescription, "got -6");

    int slots[2] = {10, 20};
    SA_EXPECT_FAILURE(bump_slot(slots, 5), StatusCode::InvalidDescription, "slot 5");
    EXPECT_EQ(slots[1], 20);
}

/** @brief Tests that SA_TRY_ASSIGN moves a move-only value out of a temporary Result. */
TEST(Small_Result, TryAssignMovesFromATemporaryResult)
{
    SA_ASSERT_OK_AND_ASSIGN(const int value, unboxed_extent(6));

    EXPECT_EQ(value, 6);
}

/** @brief Tests that Result<T&> refers to the caller's object, and a write reaches the caller. */
TEST(Small_Result, ReferenceFormRefersToTheCallersObject)
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

/** @brief Tests that SA_TRY over a reference into a temporary Result reads its status. */
TEST(Small_Result, TryReadsAStatusReferenceIntoATemporaryResult)
{
    const InMemoryFailureBlockCounter counter;
    bool passed = false;

    SA_EXPECT_FAILURE(forward_temporary_status(-4, passed), StatusCode::InvalidDescription,
                      "extent must be positive", "got -4");
    EXPECT_FALSE(passed);
    EXPECT_EQ(counter.live(), 0u);

    SA_EXPECT_OK(forward_temporary_status(4, passed));
    EXPECT_TRUE(passed);
    EXPECT_EQ(counter.live(), 0u);
}

/** @brief Tests that a moved-from failed Result<T&> keeps its failure. */
TEST(Small_Result, MovedFromFailedReferenceResultKeepsItsFailure)
{
    int slots[2] = {10, 20};

    Result<int&> missing = slot_at(slots, 2);
    const Result<int&> forwarded(std::move(missing));

    EXPECT_FALSE(missing.has_value());
    EXPECT_FALSE(forwarded.has_value());
    SA_EXPECT_FAILURE(missing, StatusCode::InvalidDescription, "slot 2 is not one of 2");
    SA_EXPECT_FAILURE(forwarded, StatusCode::InvalidDescription, "slot 2 is not one of 2");
}

/** @brief Tests that assigning a Result<T&> replaces the held state in both directions. */
TEST(Small_Result, ReferenceResultAssignmentReplacesTheHeldStateInBothDirections)
{
    int slots[2] = {10, 20};
    const Result<int&> found = slot_at(slots, 0);
    const Result<int&> missing = slot_at(slots, 2);

    Result<int&> assigned = slot_at(slots, 1);
    assigned = missing;
    SA_EXPECT_FAILURE(assigned, StatusCode::InvalidDescription, "slot 2 is not one of 2");

    assigned = found;
    ASSERT_TRUE(assigned.has_value());
    EXPECT_TRUE(assigned.status().is_ok());
    EXPECT_EQ(&assigned.value(), &slots[0]);
}

/** @brief Tests that SA_TRY_ASSIGN binds a reference from a Result<T&> and writes through it. */
TEST(Small_Result, TryAssignBindsAReferenceFromTheReferenceForm)
{
    int slots[2] = {10, 20};

    SA_EXPECT_OK(bump_slot(slots, 0));

    EXPECT_EQ(slots[0], 11);
}

/** @brief Tests that value() on a failure of either form stops the process and prints it. */
TEST(Medium_ResultFatal, ValueOnAFailureIsFatalAndPrintsTheFailure)
{
    const Result<int> failed(description_failure("rank must be 2"));
    SA_EXPECT_FATAL((void)failed.value(),
                    "Result::value() was called on a failure", "rank must be 2");

    const Result<int&> missing(description_failure("no such slot"));
    SA_EXPECT_FATAL((void)missing.value(),
                    "Result::value() was called on a failure", "no such slot");
}

/** @brief Tests that a Result of either form built from a success Status stops the process. */
TEST(Medium_ResultFatal, BuildingFromASuccessStatusIsFatal)
{
    SA_EXPECT_FATAL((void)Result<int>(Status()), "a Result was built from a success Status");
    SA_EXPECT_FATAL((void)Result<int&>(Status()), "a Result was built from a success Status");
}
