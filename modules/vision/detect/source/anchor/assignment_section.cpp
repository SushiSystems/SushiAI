/**************************************************************************/
/* assignment_section.cpp                                                 */
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
 * @file assignment_section.cpp
 * @brief Implements the reader and the writer of an anchor task's "assignment" member.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/assignment_section.hpp>

#include <array>
#include <cstddef>
#include <iterator>
#include <string>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            using Config::Kind;
            using Config::Value;

            /** @brief Lists every key "assignment" may carry. */
            constexpr std::string_view K_ASSIGNMENT_KEYS[] = {"rule", "ratio_limit",
                                                              "neighbour_offset"};

            /** @brief Lists the keys only the shape-ratio rule reads. */
            constexpr std::string_view K_SHAPE_RATIO_KEYS[] = {"ratio_limit", "neighbour_offset"};

            /** @brief Pairs one rule with the string a task section states for it. */
            struct RuleName
            {
                AssignmentRuleKind value; /**< The enumerator. */
                std::string_view name;    /**< Its string under "rule". */
            };

            /** @brief Lists every rule; the reader and the writer both look it up. */
            constexpr RuleName K_RULE_NAMES[] = {{AssignmentRuleKind::BestShape, "best_shape"},
                                                 {AssignmentRuleKind::ShapeRatio, "shape_ratio"}};

            /** @brief Returns the quoted names of every rule, separated by commas. */
            std::string list_rule_names()
            {
                std::array<std::string_view, std::size(K_RULE_NAMES)> names{};
                for (std::size_t index = 0; index < names.size(); ++index)
                    names[index] = K_RULE_NAMES[index].name;
                return Config::list_quoted(Config::KeySpan(names.data(), names.size()));
            }

            /** @brief Reads the optional float @p key of @p reader into @p value. */
            Status read_optional_float(const Config::SectionReader& reader, std::string_view key,
                                       float& value)
            {
                if (!reader.has(key))
                    return Status();
                SA_TRY_ASSIGN(const Value* const member, reader.find(key));
                SA_TRY_ASSIGN(value, Config::read_float32(*member, reader.member(key)));
                return Status();
            }
        } // namespace

        Result<AssignmentChoice> read_assignment_section(const Config::Value* section,
                                                         std::string_view where)
        {
            AssignmentChoice choice;
            if (section == nullptr)
                return Result<AssignmentChoice>(choice);

            const Config::SectionReader reader(*section, where);
            SA_TRY(reader.check_object());
            SA_TRY(reader.check_keys(
                Config::KeySpan(K_ASSIGNMENT_KEYS, std::size(K_ASSIGNMENT_KEYS))));

            SA_TRY_ASSIGN(const Value* const rule, reader.find_of("rule", Kind::String));
            const std::string& stated = rule->as_string(where);
            const RuleName* found = nullptr;
            for (const RuleName& row : K_RULE_NAMES)
                if (row.name == stated)
                    found = &row;
            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} is '{}'; it is one of {}", reader.member("rule"), stated,
                       list_rule_names());
            choice.rule = found->value;

            if (choice.rule != AssignmentRuleKind::ShapeRatio)
            {
                for (std::string_view key : K_SHAPE_RATIO_KEYS)
                    SA_FAIL_IF(reader.has(key), StatusCode::InvalidDescription,
                               "{} is stated, but the rule '{}' does not read it",
                               reader.member(key), stated);
                return Result<AssignmentChoice>(choice);
            }

            SA_TRY(read_optional_float(reader, "ratio_limit", choice.shape_ratio.ratio_limit));
            SA_TRY(read_optional_float(reader, "neighbour_offset",
                                       choice.shape_ratio.neighbour_offset));
            SA_TRY(validate_shape_ratio_options(choice.shape_ratio));
            return Result<AssignmentChoice>(choice);
        }

        bool writes_assignment_section(const AssignmentChoice& choice) noexcept
        {
            return choice.rule != AssignmentRuleKind::BestShape;
        }

        Result<Config::Value> write_assignment_section(const AssignmentChoice& choice,
                                                       std::string_view where)
        {
            const RuleName* found = nullptr;
            for (const RuleName& row : K_RULE_NAMES)
                if (row.value == choice.rule)
                    found = &row;
            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} holds the value {}, which is no rule; the rules are {}",
                       Config::member_of(where, "rule"), static_cast<std::int64_t>(choice.rule),
                       list_rule_names());

            if (choice.rule != AssignmentRuleKind::ShapeRatio)
                return Value::object({"rule"}, {Value::string(std::string(found->name))});

            SA_TRY(validate_shape_ratio_options(choice.shape_ratio));
            return Value::object(
                {"rule", "ratio_limit", "neighbour_offset"},
                {Value::string(std::string(found->name)),
                 Value::real(static_cast<double>(choice.shape_ratio.ratio_limit)),
                 Value::real(static_cast<double>(choice.shape_ratio.neighbour_offset))});
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
