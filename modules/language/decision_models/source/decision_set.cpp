/**************************************************************************/
/* decision_set.cpp                                                       */
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
 * @file decision_set.cpp
 * @brief Implements the synthetic decision set and the rules of its three types.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/decision_set.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        using Datasets::DecisionRecord;
        using Datasets::DecisionType;

        /** @brief Holds the id the number rule centres on: it weighs 0. */
        constexpr Text::TokenID K_NUMBER_CENTRE = 28;

        /** @brief Holds the divisor of a number weight, so the weights span [-1, 1]. */
        constexpr float K_NUMBER_SCALE = 3.0f;

        /** @brief Counts the positive tokens a yes-or-no question needs to answer yes. */
        constexpr int K_POSITIVES_FOR_YES = 2;

        /** @brief Counts the types of a record; the types take turns along the set. */
        constexpr std::int64_t K_TYPES = 3;

        /** @brief Holds the splitmix64 increment. */
        constexpr std::uint64_t K_STREAM_STEP = 0x9E3779B97F4A7C15ull;

        /** @brief Draws the integers of a seeded stream. */
        class SeededStream
        {
            public:
                /** @brief Seeds the stream. */
                explicit SeededStream(std::uint64_t seed) noexcept : state_(seed)
                {

                }

                /** @brief Returns an integer in [0, @p bound); @p bound is positive. */
                [[nodiscard]] std::size_t below(std::size_t bound) noexcept
                {
                    return static_cast<std::size_t>(next() % static_cast<std::uint64_t>(bound));
                }

            private:
                /** @brief Returns the next 64 bits of the splitmix64 sequence. */
                [[nodiscard]] std::uint64_t next() noexcept
                {
                    state_ += K_STREAM_STEP;
                    std::uint64_t z = state_;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                    return z ^ (z >> 31);
                }

                /// @brief Holds the splitmix64 state.
                std::uint64_t state_;
        };

        /** @brief Returns a token of the range [@p first, @p first + @p count). */
        [[nodiscard]] Text::TokenID draw_token(SeededStream& stream, Text::TokenID first,
                                               std::int32_t count) noexcept
        {
            return first + static_cast<Text::TokenID>(stream.below(static_cast<std::size_t>(count)));
        }

        /** @brief Returns the option ids of a choice: @p right and @p count - 1 others, shuffled. */
        [[nodiscard]] std::vector<Text::TokenID> draw_option_ids(SeededStream& stream,
                                                                 Text::TokenID right,
                                                                 std::size_t count)
        {
            std::vector<Text::TokenID> others;
            for (std::int32_t index = 0; index < K_CHOICE_KEYS; ++index)
            {
                if (K_CHOICE_OPTION_FIRST + index != right)
                    others.push_back(K_CHOICE_OPTION_FIRST + index);
            }

            // A partial shuffle: the first count - 1 of the others are the ones kept.
            for (std::size_t index = 0; index + 1 < count; ++index)
                std::swap(others[index], others[index + stream.below(others.size() - index)]);

            std::vector<Text::TokenID> chosen(others.begin(),
                                              others.begin() + static_cast<std::ptrdiff_t>(count - 1));
            chosen.push_back(right);
            for (std::size_t index = chosen.size() - 1; index > 0; --index)
                std::swap(chosen[index], chosen[stream.below(index + 1)]);

            return chosen;
        }

        /** @brief Returns a choice question of at most @p slots options. */
        [[nodiscard]] DecisionRecord draw_choice(SeededStream& stream, std::int32_t slots)
        {
            DecisionRecord record;
            record.type = DecisionType::Choice;

            const Text::TokenID key = draw_token(stream, K_CHOICE_KEY_FIRST, K_CHOICE_KEYS);
            const Text::TokenID noise = draw_token(stream, K_NUMBER_FIRST, K_NUMBER_TOKENS);
            record.text = {key, noise};

            const std::size_t span = static_cast<std::size_t>(slots - K_MIN_CHOICE_OPTIONS + 1);
            const std::size_t count = static_cast<std::size_t>(K_MIN_CHOICE_OPTIONS) + stream.below(span);
            const Text::TokenID right = right_option_of(key);
            const std::vector<Text::TokenID> ids = draw_option_ids(stream, right, count);
            for (std::size_t index = 0; index < ids.size(); ++index)
            {
                record.options.push_back({ids[index]});
                if (ids[index] == right)
                    record.choice = static_cast<std::int32_t>(index);
            }

            return record;
        }

        /** @brief Returns a question of three tokens of one range, of the type @p type. */
        [[nodiscard]] DecisionRecord draw_scalar(SeededStream& stream, DecisionType type)
        {
            DecisionRecord record;
            record.type = type;

            const bool is_yes_no = type == DecisionType::YesNo;
            const Text::TokenID first = is_yes_no ? K_YES_NO_FIRST : K_NUMBER_FIRST;
            const std::int32_t count = is_yes_no ? K_YES_NO_TOKENS : K_NUMBER_TOKENS;
            for (std::int32_t index = 0; index < K_SCALAR_TEXT_TOKENS; ++index)
                record.text.push_back(draw_token(stream, first, count));

            const SushiRuntime::span<const Text::TokenID> text(record.text.data(),
                                                               record.text.size());
            if (is_yes_no)
                record.yes = yes_no_answer_of(text);
            else
                record.number = number_answer_of(text);

            return record;
        }

        /** @brief Returns the record at position @p index of the set, of its type in turn. */
        [[nodiscard]] DecisionRecord draw_record(SeededStream& stream, std::int64_t index,
                                                 std::int32_t slots)
        {
            switch (index % K_TYPES)
            {
                case 0:
                    return draw_choice(stream, slots);
                case 1:
                    return draw_scalar(stream, DecisionType::YesNo);
                default:
                    return draw_scalar(stream, DecisionType::Number);
            }
        }
    } // namespace

    bool yes_no_answer_of(SushiRuntime::span<const Text::TokenID> text) noexcept
    {
        int positives = 0;
        for (const Text::TokenID token : text)
        {
            if (token < K_YES_NO_FIRST + K_YES_NO_TOKENS / 2)
                ++positives;
        }

        return positives >= K_POSITIVES_FOR_YES;
    }

    float number_answer_of(SushiRuntime::span<const Text::TokenID> text) noexcept
    {
        float total = 0.0f;
        for (const Text::TokenID token : text)
            total += static_cast<float>(token - K_NUMBER_CENTRE) / K_NUMBER_SCALE;

        return text.empty() ? 0.0f : total / static_cast<float>(text.size());
    }

    Result<std::vector<DecisionRecord>> generate_decision_set(
        const DecisionSetOptions& options) noexcept
    {
        SA_FAIL_IF(options.records < 0, StatusCode::InvalidDescription,
                   "a decision set of {} records is no set", options.records);
        SA_FAIL_IF(options.options < K_MIN_CHOICE_OPTIONS || options.options > K_CHOICE_KEYS,
                   StatusCode::InvalidDescription,
                   "a decision set holds choices of {} to {} options; {} was asked for",
                   K_MIN_CHOICE_OPTIONS, K_CHOICE_KEYS, options.options);

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            SeededStream stream(options.seed);
            std::vector<DecisionRecord> records;
            records.reserve(static_cast<std::size_t>(options.records));
            for (std::int64_t index = 0; index < options.records; ++index)
                records.push_back(draw_record(stream, index, options.options));

            for (std::size_t index = records.size(); index > 1; --index)
                std::swap(records[index - 1], records[stream.below(index)]);

            return records;
        });
    }

    Status write_decision_set(const std::filesystem::path& file,
                              const DecisionSetOptions& options) noexcept
    {
        SA_TRY_ASSIGN(const std::vector<DecisionRecord> records, generate_decision_set(options));

        return Datasets::write_decision_file(
            file, SushiRuntime::span<const DecisionRecord>(records.data(), records.size()),
            K_DECISION_SET_VOCABULARY);
    }
} // namespace SushiAI::Language::DecisionModels
