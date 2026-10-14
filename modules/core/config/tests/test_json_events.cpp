/**************************************************************************/
/* test_json_events.cpp                                                   */
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
 * @file test_json_events.cpp
 * @brief Tests the reader that streams a JSON file as events, and each of its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/json_events.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Holds one file below the temporary directory and removes it when it ends. */
    class ScratchFile final
    {
        public:
            /** @brief Writes @p text to a file called @p name. */
            ScratchFile(const char* name, std::string_view text)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::ofstream out(path_, std::ios::binary);
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
            }

            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ScratchFile(const ScratchFile&) = delete;
            ScratchFile& operator=(const ScratchFile&) = delete;
            ScratchFile(ScratchFile&&) = delete;
            ScratchFile& operator=(ScratchFile&&) = delete;

            /** @brief Returns where the file is. */
            [[nodiscard]] const std::filesystem::path& path() const noexcept
            {
                return path_;
            }

        private:
            /// @brief The path of the file.
            std::filesystem::path path_;
    };

    /** @brief Writes every event it receives as one line of text, and may stop at one. */
    class EventLog final : public Config::JSONEvents
    {
        public:
            /** @brief Stops at the event numbered @p stop_at, counted from one; 0 never. */
            explicit EventLog(std::size_t stop_at = 0) noexcept : stop_at_(stop_at)
            {

            }

            bool begin_object() override { return note("{"); }
            bool end_object() override { return note("}"); }
            bool begin_array() override { return note("["); }
            bool end_array() override { return note("]"); }
            bool key(std::string_view text) override { return note("key " + std::string(text)); }
            bool null() override { return note("null"); }
            bool boolean(bool value) override { return note(value ? "true" : "false"); }
            bool whole(std::int64_t value) override
            {
                return note("whole " + std::to_string(value));
            }
            bool real(double value) override { return note("real " + std::to_string(value)); }
            bool string(std::string_view text) override
            {
                return note("string " + std::string(text));
            }

            /** @brief Returns the events received, in order. */
            [[nodiscard]] const std::vector<std::string>& events() const noexcept
            {
                return events_;
            }

        private:
            /** @brief Keeps one event and returns whether the read goes on. */
            [[nodiscard]] bool note(std::string event)
            {
                events_.push_back(std::move(event));
                return events_.size() != stop_at_;
            }

            /// @brief The events received.
            std::vector<std::string> events_;

            /// @brief The one-based event the log stops at; 0 for none.
            std::size_t stop_at_ = 0;
    };
} // namespace

/** @brief Tests that every kind of event arrives once, in the order of the text. */
TEST(Unit_JsonEvents, EventsArriveInTheOrderOfTheText)
{
    const ScratchFile file("sushiai_test_json_events_order.json",
                           R"({"a": [1, -2, 2.5, "x", true, null], "b": {"c": false}})");

    EventLog log;
    SA_EXPECT_OK(Config::read_json_events(file.path(), log));

    const std::vector<std::string> expected = {
        "{",        "key a",  "[",     "whole 1", "whole -2", "real 2.500000",
        "string x", "true",   "null",  "]",       "key b",    "{",
        "key c",    "false",  "}",     "}",
    };
    EXPECT_EQ(log.events(), expected);
}

/** @brief Tests that a whole number above the largest int64 arrives as a real. */
TEST(Unit_JsonEvents, AWholeNumberAboveInt64IsAReal)
{
    const ScratchFile file("sushiai_test_json_events_wide.json",
                           "[9223372036854775807, 18446744073709551615]");

    EventLog log;
    SA_EXPECT_OK(Config::read_json_events(file.path(), log));

    ASSERT_EQ(log.events().size(), 4u);
    EXPECT_EQ(log.events()[1], "whole 9223372036854775807");
    EXPECT_EQ(log.events()[2].substr(0, 5), "real ");
}

/** @brief Tests that a document longer than the read buffer arrives whole. */
TEST(Unit_JsonEvents, ADocumentLongerThanTheBufferArrivesWhole)
{
    constexpr std::size_t entries = 40000;
    std::string text = "[";
    for (std::size_t entry = 0; entry < entries; ++entry)
        text += (entry == 0 ? "" : ",") + std::to_string(entry);
    text += "]";
    ASSERT_GT(text.size(), 2 * Config::K_JSON_EVENT_BUFFER_BYTES);

    const ScratchFile file("sushiai_test_json_events_long.json", text);

    EventLog log;
    SA_EXPECT_OK(Config::read_json_events(file.path(), log));

    ASSERT_EQ(log.events().size(), entries + 2);
    EXPECT_EQ(log.events()[entries], "whole " + std::to_string(entries - 1));
    EXPECT_EQ(log.events().back(), "]");
}

/** @brief Tests that an event that returns false ends the read with success. */
TEST(Unit_JsonEvents, AHandlerThatStopsEndsTheReadWithSuccess)
{
    const ScratchFile file("sushiai_test_json_events_stop.json", R"({"a": [1, 2, 3]})");

    EventLog log(4);
    SA_EXPECT_OK(Config::read_json_events(file.path(), log));

    const std::vector<std::string> expected = {"{", "key a", "[", "whole 1"};
    EXPECT_EQ(log.events(), expected);
}

/** @brief Tests that text that is no JSON is refused with the path and the byte offset. */
TEST(Unit_JsonEvents, TextThatIsNoJSONIsRefusedWithItsOffset)
{
    const ScratchFile cut("sushiai_test_json_events_cut.json", R"({"a": [1, 2)");
    const ScratchFile trailing("sushiai_test_json_events_trailing.json", "[1] x");
    const ScratchFile empty("sushiai_test_json_events_empty.json", "");

    EventLog log;
    SA_EXPECT_FAILURE(Config::read_json_events(cut.path(), log), StatusCode::InvalidFile,
                      "sushiai_test_json_events_cut.json", "not valid JSON", "at byte 12");
    SA_EXPECT_FAILURE(Config::read_json_events(trailing.path(), log),
                      StatusCode::InvalidFile, "not valid JSON", "at byte 5");
    SA_EXPECT_FAILURE(Config::read_json_events(empty.path(), log), StatusCode::InvalidFile,
                      "not valid JSON", "at byte");
}

/** @brief Tests that a file that is not there is refused by its path. */
TEST(Unit_JsonEvents, AnAbsentFileIsRefusedByItsPath)
{
    const std::filesystem::path absent =
        std::filesystem::temp_directory_path() / "sushiai_test_json_events_absent.json";

    EventLog log;
    SA_EXPECT_FAILURE(Config::read_json_events(absent, log), StatusCode::FileNotFound,
                      "sushiai_test_json_events_absent.json", "could not open");
}
