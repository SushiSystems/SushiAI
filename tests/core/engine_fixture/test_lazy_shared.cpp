/**************************************************************************/
/* test_lazy_shared.cpp                                                   */
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
 * @file test_lazy_shared.cpp
 * @brief Tests that LazyShared creates its value once, on first use, through its factory.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/engine_fixture/lazy_shared.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Counts the calls of a factory and gives each created value its call number. */
    struct CountingFactory
    {
        /** @brief Creates the value, which is the number of calls so far. */
        Result<int> operator()()
        {
            return ++*calls;
        }

        /// @brief The calls made so far; shared by every copy of the factory.
        int* calls = nullptr;
    };

    /** @brief Fails every call, as a runtime that cannot be created would. */
    Result<int> failing_factory()
    {
        SA_FAIL_IF(true, StatusCode::Unsupported, "no device here");
        return 0;
    }

    using SushiAI::Test::LazyShared;
} // namespace

/** @brief Tests that the factory is not called before the first acquire. */
TEST(Small_LazyShared, TheFactoryIsNotCalledBeforeTheFirstAcquire)
{
    int calls = 0;

    const LazyShared<int> shared{CountingFactory{&calls}};

    EXPECT_EQ(calls, 0);
}

/** @brief Tests that two acquires return the same value and call the factory once. */
TEST(Small_LazyShared, EveryAcquireReturnsTheSameValueFromOneCall)
{
    int calls = 0;
    LazyShared<int> shared{CountingFactory{&calls}};

    int& first = shared.acquire().value();
    int& second = shared.acquire().value();

    EXPECT_EQ(&first, &second);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(first, 1);
}

/** @brief Tests that a factory that fails gives its failure and is asked again next time. */
TEST(Small_LazyShared, AFailedFactoryGivesItsFailureAndIsAskedAgain)
{
    int attempts = 0;
    LazyShared<int> shared{[&attempts]()
    {
        ++attempts;
        return failing_factory();
    }};

    SA_EXPECT_FAILURE(shared.acquire(), StatusCode::Unsupported, "no device here");
    SA_EXPECT_FAILURE(shared.acquire(), StatusCode::Unsupported, "no device here");

    EXPECT_EQ(attempts, 2);
}

/** @brief Tests that release destroys the value, so the next acquire creates another. */
TEST(Small_LazyShared, ReleaseMakesTheNextAcquireCreateAgain)
{
    int calls = 0;
    LazyShared<int> shared{CountingFactory{&calls}};
    SA_ASSERT_OK_AND_ASSIGN(int& before, shared.acquire());
    ASSERT_EQ(before, 1);

    shared.release();

    SA_ASSERT_OK_AND_ASSIGN(int& after, shared.acquire());
    EXPECT_EQ(after, 2);
}

/** @brief Tests that a factory set before the first acquire replaces the first. */
TEST(Small_LazyShared, ANewFactoryReplacesTheOneKept)
{
    int calls = 0;
    LazyShared<int> shared{CountingFactory{&calls}};

    shared.use_factory([]()
    {
        return Result<int>(7);
    });

    SA_ASSERT_OK_AND_ASSIGN(int& value, shared.acquire());
    EXPECT_EQ(value, 7);
    EXPECT_EQ(calls, 0);
}
