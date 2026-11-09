/**************************************************************************/
/* test_sampler_section.cpp                                              */
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
 * @file test_sampler_section.cpp
 * @brief Tests that make_sampler builds each of the four samplers and refuses a bad section.
 * @author Mustafa Garip
 */

#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
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
    using SushiAI::Test::require_value;
    /** @brief Names the section in every diagnostic of these tests. */
    constexpr std::string_view K_CONTEXT = "the task section 'sampler'";

    /** @brief Builds the sampler of a section written as JSON. */
    Result<std::unique_ptr<Sampler>> sampler_of(std::string_view text)
    {
        return make_sampler(require_value(Config::parse_json(text, "<test>")), K_CONTEXT);
    }

    /** @brief Names a section and the name of the sampler it builds. */
    struct BuildCase
    {
        std::string_view section; /**< The section, as JSON. */
        std::string_view name;    /**< The name of the sampler class it builds. */
    };

    /** @brief Names a section and what the refusal of it names. */
    struct RefusalCase
    {
        std::string_view section; /**< The section, as JSON. */
        std::string_view first;   /**< A substring the message holds. */
        std::string_view second;  /**< Another substring the message holds. */
    };

    /** @brief Returns one section per type, and a temperature section without its optional key. */
    std::vector<BuildCase> build_cases()
    {
        return {{R"({"type": "greedy"})", K_GREEDY_SAMPLER},
                {R"({"type": "temperature", "temperature": 0.7})", K_TEMPERATURE_SAMPLER},
                {R"({"type": "temperature"})", K_TEMPERATURE_SAMPLER},
                {R"({"type": "top_k", "k": 40, "temperature": 2})", K_TOP_K_SAMPLER},
                {R"({"type": "nucleus", "p": 0.9})", K_NUCLEUS_SAMPLER}};
    }

    /** @brief Returns the sections of an unknown type, an absent key and a section that is no object. */
    std::vector<RefusalCase> absent_cases()
    {
        return {{R"({"type": "beam"})", "'beam'", "'greedy', 'temperature', 'top_k', 'nucleus'"},
                {R"({"type": "top_k"})", "'sampler'", "has no 'k'"},
                {R"({"type": "nucleus"})", "has no 'p'", ""},
                {R"({"temperature": 1.0})", "has no 'type'", ""},
                {R"("greedy")", "an object", ""}};
    }

    /** @brief Returns the sections with a key the type does not read or a value out of range. */
    std::vector<RefusalCase> unread_cases()
    {
        return {{R"({"type": "greedy", "temperature": 0.5})", "'temperature'", "does not have"},
                {R"({"type": "temperature", "k": 3})", "'k'", ""},
                {R"({"type": "top_k", "k": 0})", "keep 0 tokens", ""},
                {R"({"type": "top_k", "k": "many"})", "'k'", "whole number"},
                {R"({"type": "nucleus", "p": 2})", "at most 1", ""}};
    }

    /** @brief Provides the sections each type builds. */
    class Small_GenerationSamplerBuild : public ::testing::TestWithParam<BuildCase>
    {

    };

    /** @brief Provides the sections refused for an absent or unknown part. */
    class Small_GenerationSamplerAbsent : public ::testing::TestWithParam<RefusalCase>
    {

    };

    /** @brief Provides the sections refused for a key or a value that does not fit. */
    class Small_GenerationSamplerUnread : public ::testing::TestWithParam<RefusalCase>
    {

    };
} // namespace

/** @brief Tests that each type builds the sampler of its name. */
TEST_P(Small_GenerationSamplerBuild, EachTypeBuildsItsSampler)
{
    const BuildCase& row = GetParam();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Sampler> sampler, sampler_of(row.section));

    EXPECT_EQ(sampler->name(), row.name);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationSamplerBuild, ::testing::ValuesIn(build_cases()));

/** @brief Tests that an unknown type lists the four and that a required key is asked for. */
TEST_P(Small_GenerationSamplerAbsent, AnUnknownTypeAndAnAbsentKeyAreRefused)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(sampler_of(row.section), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationSamplerAbsent,
                         ::testing::ValuesIn(absent_cases()));

/** @brief Tests that a key the type does not read, and a value out of range, are refused. */
TEST_P(Small_GenerationSamplerUnread, AKeyTheTypeDoesNotReadIsRefused)
{
    const RefusalCase& row = GetParam();

    SA_EXPECT_FAILURE(sampler_of(row.section), StatusCode::InvalidDescription, row.first,
                      row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_GenerationSamplerUnread,
                         ::testing::ValuesIn(unread_cases()));
