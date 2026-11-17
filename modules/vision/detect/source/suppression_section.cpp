/**************************************************************************/
/* suppression_section.cpp                                                */
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
 * @file suppression_section.cpp
 * @brief Implements the reader and the writer of a task section's "suppression" object.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/suppression_section.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            using Config::Kind;
            using Config::KeySpan;
            using Config::member_of;
            using Config::read_float32;
            using Config::SectionReader;
            using Config::Value;

            /** @brief Lists every key "suppression" may carry without a threshold. */
            constexpr std::string_view K_CRITERION_ONLY_KEYS[] = {"criterion",
                                                                "distance_exponent"};

            /** @brief Names the member that holds the object. */
            constexpr std::string_view K_SUPPRESSION = "suppression";

            /** @brief Lists every key "suppression" may carry. */
            constexpr std::string_view K_SUPPRESSION_KEYS[] = {"criterion", "threshold",
                                                             "distance_exponent"};

            /** @brief Pairs one suppression criterion with its string and what it reads. */
            struct CriterionName
            {
                SuppressionCriterion value; /**< The enumerator. */
                std::string_view name;      /**< Its string under "criterion". */
                bool reads_exponent;        /**< Whether the measure reads distance_exponent. */
            };

            /** @brief Lists every criterion; the reader and the writer both look it up. */
            constexpr CriterionName K_CRITERION_NAMES[] = {
                {SuppressionCriterion::IoU, "iou", false},
                {SuppressionCriterion::DistanceIoU, "distance_iou", true}};

            /** @brief Returns the strings of K_CRITERION_NAMES, quoted and separated by commas. */
            std::string list_criteria()
            {
                std::array<std::string_view, std::size(K_CRITERION_NAMES)> names{};
                for (std::size_t index = 0; index < names.size(); ++index)
                    names[index] = K_CRITERION_NAMES[index].name;
                return Config::list_quoted(KeySpan(names.data(), names.size()));
            }

            /** @brief Returns the row whose string is @p name, or null. */
            const CriterionName* criterion_by_name(std::string_view name) noexcept
            {
                for (const CriterionName& row : K_CRITERION_NAMES)
                    if (row.name == name)
                        return &row;
                return nullptr;
            }

            /** @brief Returns the row of @p value, or null. */
            const CriterionName* criterion_by_value(SuppressionCriterion value) noexcept
            {
                for (const CriterionName& row : K_CRITERION_NAMES)
                    if (row.value == value)
                        return &row;
                return nullptr;
            }

            /** @brief Reports whether @p value is a finite number in [0, 1]. */
            bool is_fraction(float value) noexcept
            {
                return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
            }

            /** @brief Checks every field of @p section, named @p where. */
            Status validate(const SuppressionSection& section, std::string_view where,
                            SuppressionKeys keys)
            {
                const CriterionName* const row = criterion_by_value(section.criterion);
                SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                           "{} holds the value {}, which is no criterion; the criteria are {}",
                           member_of(where, "criterion"),
                           static_cast<std::int64_t>(section.criterion), list_criteria());
                SA_FAIL_IF(keys == SuppressionKeys::WithThreshold
                               && !is_fraction(section.threshold),
                           StatusCode::InvalidDescription,
                           "{} must be from 0 to 1; got {}", member_of(where, "threshold"),
                           section.threshold);
                SA_FAIL_IF(row->reads_exponent && !std::isfinite(section.distance_exponent),
                           StatusCode::InvalidDescription, "{} must be finite; got {}",
                           member_of(where, "distance_exponent"), section.distance_exponent);
                return Status();
            }
        } // namespace

        Result<SuppressionSection> read_suppression_section(const Config::Value& task,
                                                            std::string_view where,
                                                            SuppressionKeys keys) noexcept
        {
            const std::string name = member_of(where, K_SUPPRESSION);
            SA_TRY_ASSIGN(const Value* const object,
                          SectionReader(task, where).find_of(K_SUPPRESSION, Kind::Object));

            const SectionReader reader(*object, name);
            const bool thresholded = keys == SuppressionKeys::WithThreshold;
            SA_TRY(thresholded ? reader.check_keys(KeySpan(K_SUPPRESSION_KEYS,
                                                           std::size(K_SUPPRESSION_KEYS)))
                               : reader.check_keys(KeySpan(K_CRITERION_ONLY_KEYS,
                                                           std::size(K_CRITERION_ONLY_KEYS))));

            SuppressionSection section;
            SA_TRY_ASSIGN(const Value* const criterion, reader.find_of("criterion", Kind::String));
            SA_TRY_ASSIGN(const std::string& stated, criterion->as_string(name));
            const CriterionName* const row = criterion_by_name(stated);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "{} is '{}'; it is one of {}", member_of(name, "criterion"), stated,
                       list_criteria());
            section.criterion = row->value;

            if (thresholded)
            {
                SA_TRY_ASSIGN(const Value* const threshold, reader.find("threshold"));
                SA_TRY_ASSIGN(section.threshold,
                              read_float32(*threshold, reader.member("threshold")));
            }

            if (row->reads_exponent)
            {
                SA_TRY_ASSIGN(const Value* const exponent, reader.find("distance_exponent"));
                SA_TRY_ASSIGN(section.distance_exponent,
                              read_float32(*exponent, reader.member("distance_exponent")));
            }
            else
            {
                SA_FAIL_IF(reader.has("distance_exponent"), StatusCode::InvalidDescription,
                           "{} is stated, but the criterion '{}' does not read it",
                           member_of(name, "distance_exponent"), stated);
            }

            SA_TRY(validate(section, name, keys));
            return Result<SuppressionSection>(section);
        }

        Result<Config::Value> write_suppression_section(const SuppressionSection& section,
                                                        std::string_view where,
                                                        SuppressionKeys keys) noexcept
        {
            SA_TRY(validate(section, member_of(where, K_SUPPRESSION), keys));

            const CriterionName* const row = criterion_by_value(section.criterion);
            std::vector<std::string> names = {"criterion"};
            std::vector<Value> values = {Value::string(std::string(row->name))};
            if (keys == SuppressionKeys::WithThreshold)
            {
                names.emplace_back("threshold");
                values.push_back(Value::real(static_cast<double>(section.threshold)));
            }
            if (row->reads_exponent)
            {
                names.emplace_back("distance_exponent");
                values.push_back(Value::real(static_cast<double>(section.distance_exponent)));
            }

            return Value::object(std::move(names), std::move(values));
        }

    } // namespace Vision::Detect
} // namespace SushiAI
