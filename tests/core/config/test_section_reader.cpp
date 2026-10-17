/**************************************************************************/
/* test_section_reader.cpp                                                */
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
 * @file test_section_reader.cpp
 * @brief Tests the readers of one object of a document, and the text of each refusal.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
using Config::KeySpan;
using Config::Kind;
using Config::SectionReader;
using Config::Value;

namespace
{
    /** @brief Names the section in a diagnostic. */
    constexpr std::string_view K_WHERE = "schedule";

    /** @brief Lists the keys the section of the cases accepts. */
    constexpr std::string_view K_KEYS[] = {"type", "base_rate", "steps"};

    /** @brief Parses @p text as a section. */
    Value section_from(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }

    /** @brief Returns K_KEYS as the view check_keys() takes. */
    KeySpan keys()
    {
        return KeySpan(K_KEYS, std::size(K_KEYS));
    }
} // namespace

/** @brief Tests the names of a member and of a list entry, and a quoted list. */
TEST(Small_SectionReader, NamesAMemberAnEntryAndAList)
{
    EXPECT_EQ(Config::member_of("schedule", "steps"), "schedule 'steps'");
    EXPECT_EQ(Config::entry_of("schedule 'steps'", 2), "schedule 'steps' entry 2");
    EXPECT_EQ(Config::list_quoted(keys()), "'type', 'base_rate', 'steps'");
    EXPECT_EQ(Config::list_quoted(KeySpan()), "");

    const Value section = section_from("{}");
    EXPECT_EQ(SectionReader(section, K_WHERE).member("type"), "schedule 'type'");
}

/** @brief Tests that an object passes and that anything else is refused with its kind. */
TEST(Small_SectionReader, CheckObjectRefusesAnotherKind)
{
    const Value object = section_from("{}");
    const Value text = section_from(R"("step")");

    SA_EXPECT_OK(SectionReader(object, K_WHERE).check_object());
    SA_EXPECT_FAILURE(SectionReader(text, K_WHERE).check_object(),
                      StatusCode::InvalidDescription,
                      "schedule must be an object, but it is a string");
}

/** @brief Tests that a key outside the list is refused with the key and the list. */
TEST(Small_SectionReader, CheckKeysNamesTheKeyAndListsWhatIsAccepted)
{
    const Value good = section_from(R"({"steps": [], "type": "step"})");
    const Value bad = section_from(R"({"base_rate": 0.1, "gamma": 2})");

    SA_EXPECT_OK(SectionReader(good, K_WHERE).check_keys(keys()));
    SA_EXPECT_FAILURE(SectionReader(bad, K_WHERE).check_keys(keys()),
                      StatusCode::InvalidDescription,
                      "schedule states the key 'gamma', which it does not have; it accepts "
                      "'type', 'base_rate', 'steps'");
}

/** @brief Tests the type key: absent passes, another string and another kind are refused. */
TEST(Small_SectionReader, CheckTypeAcceptsAnAbsentKeyAndRefusesAnotherType)
{
    const Value absent = section_from(R"({"base_rate": 0.1})");
    const Value same = section_from(R"({"type": "step"})");
    const Value other = section_from(R"({"type": "cosine"})");
    const Value number = section_from(R"({"type": 3})");

    SA_EXPECT_OK(SectionReader(absent, K_WHERE).check_type("type", "step", "a step schedule"));
    SA_EXPECT_OK(SectionReader(same, K_WHERE).check_type("type", "step", "a step schedule"));
    SA_EXPECT_FAILURE(
        SectionReader(other, K_WHERE).check_type("type", "step", "a step schedule"),
        StatusCode::InvalidDescription,
        "schedule 'type' is 'cosine'; a step schedule states 'step'");
    SA_EXPECT_FAILURE(
        SectionReader(number, K_WHERE).check_type("type", "step", "a step schedule"),
        StatusCode::InvalidDescription,
        "schedule 'type' must be a string, but it is a whole number");
}

/** @brief Tests that find() returns a member and names an absent key. */
TEST(Small_SectionReader, FindReturnsTheMemberOrNamesTheAbsentKey)
{
    const Value section = section_from(R"({"base_rate": 0.5})");
    const SectionReader reader(section, K_WHERE);

    EXPECT_TRUE(reader.has("base_rate"));
    EXPECT_FALSE(reader.has("steps"));
    SA_ASSERT_OK_AND_ASSIGN(const Value* const member, reader.find("base_rate"));
    EXPECT_EQ(member->kind(), Kind::Real);
    SA_EXPECT_FAILURE(reader.find("steps"), StatusCode::InvalidDescription,
                      "schedule has no 'steps'");
}

/** @brief Tests that find_of() checks the kind and names both the kind asked and found. */
TEST(Small_SectionReader, FindOfChecksTheKind)
{
    const Value section = section_from(R"({"steps": 3})");
    const SectionReader reader(section, K_WHERE);

    SA_EXPECT_OK(reader.find_of("steps", Kind::Int));
    SA_EXPECT_FAILURE(reader.find_of("steps", Kind::Array), StatusCode::InvalidDescription,
                      "schedule 'steps' must be an array, but it is a whole number");
    SA_EXPECT_FAILURE(reader.find_of("type", Kind::String), StatusCode::InvalidDescription,
                      "schedule has no 'type'");
}

/** @brief Tests the number readers: a whole number widens to a real and not the reverse. */
TEST(Small_SectionReader, TheNumberReadersCheckTheKind)
{
    const Value whole = section_from("7");
    const Value real = section_from("0.25");
    const Value text = section_from(R"("seven")");

    // Both literals are exact in a double, so the values are compared exactly.
    SA_ASSERT_OK_AND_ASSIGN(const double widened, Config::read_real(whole, "rate"));
    EXPECT_EQ(widened, 7.0);
    SA_ASSERT_OK_AND_ASSIGN(const double quarter, Config::read_real(real, "rate"));
    EXPECT_EQ(quarter, 0.25);
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t seven, Config::read_whole(whole, "count"));
    EXPECT_EQ(seven, 7);

    SA_EXPECT_FAILURE(Config::read_real(text, "rate"), StatusCode::InvalidDescription,
                      "rate must be a number, but it is a string");
    SA_EXPECT_FAILURE(Config::read_whole(real, "count"), StatusCode::InvalidDescription,
                      "count must be a whole number, but it is a number");
}

/** @brief Tests that a number float32 cannot hold as a finite value is refused. */
TEST(Small_SectionReader, ReadFloat32RefusesWhatFloat32CannotHold)
{
    const Value fits = section_from("0.5");
    const Value wide = section_from("1e39");
    const Value text = section_from(R"("wide")");

    SA_ASSERT_OK_AND_ASSIGN(const float half, Config::read_float32(fits, "scale"));
    EXPECT_EQ(half, 0.5f);
    SA_EXPECT_FAILURE(Config::read_float32(wide, "scale"), StatusCode::InvalidDescription,
                      "scale is", "which float32 does not hold as a finite value");
    SA_EXPECT_FAILURE(Config::read_float32(text, "scale"), StatusCode::InvalidDescription,
                      "scale must be a number, but it is a string");
}

/** @brief Tests the string and the array readers and the kind each refuses. */
TEST(Small_SectionReader, TheStringAndArrayReadersCheckTheKind)
{
    const Value text = section_from(R"("step")");
    const Value list = section_from("[1, 2, 3]");

    SA_ASSERT_OK_AND_ASSIGN(const std::string* const read, Config::read_string(text, "type"));
    EXPECT_EQ(*read, "step");
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Value>* const entries,
                            Config::read_array(list, "steps"));
    EXPECT_EQ(entries->size(), 3u);

    SA_EXPECT_FAILURE(Config::read_string(list, "type"), StatusCode::InvalidDescription,
                      "type must be a string, but it is an array");
    SA_EXPECT_FAILURE(Config::read_array(text, "steps"), StatusCode::InvalidDescription,
                      "steps must be an array, but it is a string");
}
