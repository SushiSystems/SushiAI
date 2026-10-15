/**************************************************************************/
/* test_sampler_section.cpp                                               */
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
 * @file test_sampler_section.cpp
 * @brief Tests that make_sampler builds each of the four samplers and refuses a bad section.
 * @author Mustafa Garip
 */

#include <memory>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>
#include <SushiAI/language/generation/sampler_section.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>

using namespace SushiAI;
using namespace SushiAI::Language::Generation;

namespace
{
    /** @brief Names the section in every diagnostic of these tests. */
    constexpr std::string_view K_CONTEXT = "the task section 'sampler'";

    /** @brief Builds the sampler of a section written as JSON. */
    Result<std::unique_ptr<Sampler>> sampler_of(std::string_view text)
    {
        return make_sampler(Config::parse_json(text, "<test>"), K_CONTEXT);
    }
} // namespace

/** @brief Tests that each type builds the sampler of its name. */
TEST(Small_GenerationSamplerSection, EachTypeBuildsItsSampler)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> greedy,
                            sampler_of(R"({"type": "greedy"})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> temperature,
                            sampler_of(R"({"type": "temperature", "temperature": 0.7})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> defaulted,
                            sampler_of(R"({"type": "temperature"})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> top_k,
                            sampler_of(R"({"type": "top_k", "k": 40, "temperature": 2})"));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> nucleus,
                            sampler_of(R"({"type": "nucleus", "p": 0.9})"));

    EXPECT_EQ(greedy->name(), K_GREEDY_SAMPLER);
    EXPECT_EQ(temperature->name(), K_TEMPERATURE_SAMPLER);
    EXPECT_EQ(defaulted->name(), K_TEMPERATURE_SAMPLER);
    EXPECT_EQ(top_k->name(), K_TOP_K_SAMPLER);
    EXPECT_EQ(nucleus->name(), K_NUCLEUS_SAMPLER);
}

/** @brief Tests that an unknown type lists the four and that a required key is asked for. */
TEST(Small_GenerationSamplerSection, AnUnknownTypeAndAnAbsentKeyAreRefused)
{
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "beam"})"), StatusCode::InvalidDescription,
                      "'beam'", "'greedy', 'temperature', 'top_k', 'nucleus'");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "top_k"})"), StatusCode::InvalidDescription,
                      "'sampler'", "has no 'k'");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "nucleus"})"), StatusCode::InvalidDescription,
                      "has no 'p'");
    SA_EXPECT_FAILURE(sampler_of(R"({"temperature": 1.0})"), StatusCode::InvalidDescription,
                      "has no 'type'");
    SA_EXPECT_FAILURE(sampler_of(R"("greedy")"), StatusCode::InvalidDescription, "an object");
}

/** @brief Tests that a key the type does not read, and a value out of range, are refused. */
TEST(Small_GenerationSamplerSection, AKeyTheTypeDoesNotReadIsRefused)
{
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "greedy", "temperature": 0.5})"),
                      StatusCode::InvalidDescription, "'temperature'", "does not have");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "temperature", "k": 3})"),
                      StatusCode::InvalidDescription, "'k'");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "top_k", "k": 0})"),
                      StatusCode::InvalidDescription, "keep 0 tokens");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "top_k", "k": "many"})"),
                      StatusCode::InvalidDescription, "'k'", "whole number");
    SA_EXPECT_FAILURE(sampler_of(R"({"type": "nucleus", "p": 2})"),
                      StatusCode::InvalidDescription, "at most 1");
}
