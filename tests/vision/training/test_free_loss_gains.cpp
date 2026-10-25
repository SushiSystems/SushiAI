/**************************************************************************/
/* test_free_loss_gains.cpp                                               */
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
 * @file test_free_loss_gains.cpp
 * @brief Tests the reader of the anchor-free detection loss gains.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>

using namespace SushiAI;
using Config::Value;
using Vision::Training::FreeLossGains;
using Vision::Training::read_free_loss_gains;

/** @brief Tests that a section without a gain key keeps 7.5, 0.5, 1.5 and 0. */
TEST(Small_FreeLossGains, AnAbsentKeyKeepsItsDefault)
{
    const Value section = Value::object({"type"}, {Value::string("free")});

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossGains gains, read_free_loss_gains(section));

    EXPECT_EQ(gains.box_gain, 7.5f);
    EXPECT_EQ(gains.class_gain, 0.5f);
    EXPECT_EQ(gains.distribution_gain, 1.5f);
    EXPECT_EQ(gains.l1_gain, 0.0f);
}

/** @brief Tests that each stated gain is read, a whole number included. */
TEST(Small_FreeLossGains, EveryStatedGainIsRead)
{
    const Value section = Value::object(
        {"box_gain", "class_gain", "distribution_gain", "l1_gain"},
        {Value::real(2.0), Value::integer(1), Value::real(0.25), Value::real(1.5)});

    SA_ASSERT_OK_AND_ASSIGN(const FreeLossGains gains, read_free_loss_gains(section));

    EXPECT_EQ(gains.box_gain, 2.0f);
    EXPECT_EQ(gains.class_gain, 1.0f);
    EXPECT_EQ(gains.distribution_gain, 0.25f);
    EXPECT_EQ(gains.l1_gain, 1.5f);
}

/** @brief Tests that a negative gain is refused, naming its key. */
TEST(Small_FreeLossGains, ANegativeGainIsRefusedNamingItsKey)
{
    const Value section = Value::object({"class_gain"}, {Value::real(-0.5)});

    SA_EXPECT_FAILURE(read_free_loss_gains(section), StatusCode::InvalidDescription,
                      "class_gain", "not negative");
}

/** @brief Tests that a gain that is no number is refused, naming its key. */
TEST(Small_FreeLossGains, AGainThatIsNoNumberIsRefusedNamingItsKey)
{
    const Value section = Value::object({"box_gain"}, {Value::string("high")});

    SA_EXPECT_FAILURE(read_free_loss_gains(section), StatusCode::InvalidDescription, "box_gain");
}
