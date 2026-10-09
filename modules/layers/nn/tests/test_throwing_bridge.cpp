/**************************************************************************/
/* test_throwing_bridge.cpp                                               */
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
 * @file test_throwing_bridge.cpp
 * @brief Tests bridge_throwing: a value passes through and a thrown Error becomes a Status.
 * @author Mustafa Garip
 */

#include <memory>
#include <new>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using NN::Detail::bridge_throwing;

namespace
{
    /** @brief Throws an Error whose message names a layer and a shape. */
    int refuse_a_shape()
    {
        SA_THROW_IF(true, "Linear({}, {}) was given {}", 6, 8, "[5]");
        return 0;
    }

    /** @brief Throws std::bad_alloc, as a refused allocation does. */
    int refuse_an_allocation()
    {
        throw std::bad_alloc();
    }

    /** @brief Throws an exception that is neither an Error nor std::bad_alloc. */
    int throw_a_logic_error()
    {
        throw std::logic_error("an index was out of range");
    }

    /** @brief Bridges throw_a_logic_error, for a death test. */
    void bridge_a_logic_error()
    {
        const Result<int> ignored = bridge_throwing(&throw_a_logic_error);
        static_cast<void>(ignored);
    }
} // namespace

/** @brief Tests that the value of a call that does not throw comes back as a held value. */
TEST(Unit_ThrowingBridge, AValuePassesThrough)
{
    const Result<int> bridged = bridge_throwing([]()
    {
        return 7;
    });

    SA_EXPECT_OK(bridged);
    EXPECT_EQ(bridged.value(), 7);
}

/** @brief Tests that a move-only value comes back without a copy. */
TEST(Unit_ThrowingBridge, AMoveOnlyValuePassesThrough)
{
    Result<std::unique_ptr<int>> bridged = bridge_throwing([]()
    {
        return std::make_unique<int>(7);
    });

    SA_EXPECT_OK(bridged);
    ASSERT_NE(bridged.value(), nullptr);
    EXPECT_EQ(*bridged.value(), 7);
}

/** @brief Tests that a thrown Error becomes InvalidDescription carrying the Error's message. */
TEST(Unit_ThrowingBridge, AnErrorBecomesInvalidDescriptionWithItsMessage)
{
    SA_EXPECT_FAILURE(bridge_throwing(&refuse_a_shape), StatusCode::InvalidDescription,
                      "Linear(6, 8)", "[5]");
}

/** @brief Tests that std::bad_alloc becomes OutOfMemory. */
TEST(Unit_ThrowingBridge, ARefusedAllocationBecomesOutOfMemory)
{
    SA_EXPECT_FAILURE(bridge_throwing(&refuse_an_allocation), StatusCode::OutOfMemory,
                      "allocation was refused");
}

/** @brief Tests that an exception of another type stops the process and quotes its text. */
TEST(Unit_ThrowingBridgeDeathTest, AnotherExceptionIsFatal)
{
    SA_EXPECT_FATAL(bridge_a_logic_error(), "other than Error", "an index was out of range");
}
