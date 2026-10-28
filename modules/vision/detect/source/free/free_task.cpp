/**************************************************************************/
/* free_task.cpp                                                          */
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
 * @file free_task.cpp
 * @brief Implements the reader and the writer of an anchor-free task's section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/free_task.hpp>

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
    namespace Vision::Detect::Free
    {
        namespace
        {
            using Config::Kind;
            using Config::KeySpan;
            using Config::member_of;
            using Config::read_float32;
            using Config::SectionReader;
            using Config::Value;

            /** @brief Names the section in a diagnostic; every refusal starts with it. */
            constexpr std::string_view K_TASK = "task";

            /** @brief Lists every key a task section of this type may carry. */
            constexpr std::string_view K_TASK_KEYS[] = {"type", "score_threshold", "suppression",
                                                      "assignment"};

            /** @brief Lists every key "suppression" may carry. */
            constexpr std::string_view K_SUPPRESSION_KEYS[] = {"criterion", "threshold",
                                                             "distance_exponent"};

            /** @brief Lists every key "assignment" may carry, one per branch. */
            constexpr std::string_view K_ASSIGNMENT_KEYS[] = {"one_to_many", "one_to_one"};

            /** @brief Lists every key a branch of "assignment" carries; each is required. */
            constexpr std::string_view K_BRANCH_KEYS[] = {"top_k", "top_k2", "alpha", "beta"};

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

            /** @brief Reports whether @p value is a finite number not below zero. */
            bool is_exponent(float value) noexcept
            {
                return std::isfinite(value) && value >= 0.0f;
            }

            /** @brief Reads "suppression" of @p task into the threshold and the criterion. */
            Status read_suppression(const Value& task, FreeTask& result)
            {
                const std::string where = member_of(K_TASK, "suppression");
                SA_TRY_ASSIGN(const Value* const suppression,
                              SectionReader(task, K_TASK).find_of("suppression", Kind::Object));

                const SectionReader reader(*suppression, where);
                SA_TRY(reader.check_keys(
                    KeySpan(K_SUPPRESSION_KEYS, std::size(K_SUPPRESSION_KEYS))));

                SA_TRY_ASSIGN(const Value* const criterion,
                              reader.find_of("criterion", Kind::String));
                SA_TRY_ASSIGN(const std::string& stated, criterion->as_string(where));
                const CriterionName* const row = criterion_by_name(stated);
                SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                           "{} is '{}'; it is one of {}", member_of(where, "criterion"), stated,
                           list_criteria());
                result.criterion = row->value;

                SA_TRY_ASSIGN(const Value* const threshold, reader.find("threshold"));
                SA_TRY_ASSIGN(result.suppression_threshold,
                              read_float32(*threshold, reader.member("threshold")));

                if (!row->reads_exponent)
                {
                    SA_FAIL_IF(suppression->has("distance_exponent"),
                               StatusCode::InvalidDescription,
                               "{} is stated, but the criterion '{}' does not read it",
                               member_of(where, "distance_exponent"), stated);
                    return Status();
                }

                SA_TRY_ASSIGN(const Value* const exponent, reader.find("distance_exponent"));
                SA_TRY_ASSIGN(result.distance_exponent,
                              read_float32(*exponent, reader.member("distance_exponent")));
                return Status();
            }

            /** @brief Reads one branch of "assignment", named @p where, into @p config. */
            Status read_branch(const Value& branch, std::string_view where, AssignerConfig& config)
            {
                const SectionReader reader(branch, where);
                SA_TRY(reader.check_object());
                SA_TRY(reader.check_keys(KeySpan(K_BRANCH_KEYS, std::size(K_BRANCH_KEYS))));

                SA_TRY_ASSIGN(const Value* const top_k, reader.find_of("top_k", Kind::Int));
                SA_TRY_ASSIGN(config.topk, top_k->as_int(where));
                SA_TRY_ASSIGN(const Value* const top_k2, reader.find_of("top_k2", Kind::Int));
                SA_TRY_ASSIGN(config.topk2, top_k2->as_int(where));

                SA_TRY_ASSIGN(const Value* const alpha, reader.find("alpha"));
                SA_TRY_ASSIGN(config.alpha, read_float32(*alpha, reader.member("alpha")));
                SA_TRY_ASSIGN(const Value* const beta, reader.find("beta"));
                SA_TRY_ASSIGN(config.beta, read_float32(*beta, reader.member("beta")));
                return Status();
            }

            /** @brief Reads "assignment" of @p task, when stated, into both branches. */
            Status read_assignment(const Value& task, FreeTask& result)
            {
                const SectionReader task_reader(task, K_TASK);
                if (!task_reader.has("assignment"))
                    return Status();

                const std::string where = member_of(K_TASK, "assignment");
                SA_TRY_ASSIGN(const Value* const assignment,
                              task_reader.find_of("assignment", Kind::Object));
                const SectionReader reader(*assignment, where);
                SA_TRY(reader.check_keys(
                    KeySpan(K_ASSIGNMENT_KEYS, std::size(K_ASSIGNMENT_KEYS))));

                if (reader.has("one_to_many"))
                {
                    SA_TRY_ASSIGN(const Value* const branch, reader.find("one_to_many"));
                    SA_TRY(read_branch(*branch, reader.member("one_to_many"),
                                       result.one_to_many));
                }
                if (reader.has("one_to_one"))
                {
                    SA_TRY_ASSIGN(const Value* const branch, reader.find("one_to_one"));
                    SA_TRY(read_branch(*branch, reader.member("one_to_one"), result.one_to_one));
                }
                return Status();
            }

            /** @brief Checks the section's fields of one branch, named @p where. */
            Status validate_branch(const AssignerConfig& config, std::string_view where)
            {
                SA_FAIL_IF(config.topk < 1, StatusCode::InvalidDescription,
                           "{} must be at least 1; got {}", member_of(where, "top_k"),
                           config.topk);
                SA_FAIL_IF(config.topk2 < 1 || config.topk2 > config.topk,
                           StatusCode::InvalidDescription,
                           "{} must be from 1 to 'top_k' ({}); got {}",
                           member_of(where, "top_k2"), config.topk, config.topk2);
                SA_FAIL_IF(!is_exponent(config.alpha), StatusCode::InvalidDescription,
                           "{} must be finite and not below zero; got {}",
                           member_of(where, "alpha"), config.alpha);
                SA_FAIL_IF(!is_exponent(config.beta), StatusCode::InvalidDescription,
                           "{} must be finite and not below zero; got {}",
                           member_of(where, "beta"), config.beta);
                return Status();
            }

            /** @brief Checks every field of @p task the section holds. */
            Status validate_free_task(const FreeTask& task)
            {
                SA_FAIL_IF(!is_fraction(task.score_threshold), StatusCode::InvalidDescription,
                           "{} must be from 0 to 1; got {}", member_of(K_TASK, "score_threshold"),
                           task.score_threshold);

                const std::string suppression = member_of(K_TASK, "suppression");
                SA_FAIL_IF(!is_fraction(task.suppression_threshold),
                           StatusCode::InvalidDescription, "{} must be from 0 to 1; got {}",
                           member_of(suppression, "threshold"), task.suppression_threshold);
                const CriterionName* const row = criterion_by_value(task.criterion);
                SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                           "{} holds the value {}, which is no criterion; the criteria are {}",
                           member_of(suppression, "criterion"),
                           static_cast<std::int64_t>(task.criterion), list_criteria());
                SA_FAIL_IF(row->reads_exponent && !std::isfinite(task.distance_exponent),
                           StatusCode::InvalidDescription, "{} must be finite; got {}",
                           member_of(suppression, "distance_exponent"), task.distance_exponent);

                const std::string assignment = member_of(K_TASK, "assignment");
                SA_TRY(validate_branch(task.one_to_many, member_of(assignment, "one_to_many")));
                SA_TRY(validate_branch(task.one_to_one, member_of(assignment, "one_to_one")));
                return Status();
            }

            /** @brief Builds one branch of "assignment" from @p config. */
            Value branch_value(const AssignerConfig& config)
            {
                return Value::object({"top_k", "top_k2", "alpha", "beta"},
                                     {Value::integer(config.topk), Value::integer(config.topk2),
                                      Value::real(static_cast<double>(config.alpha)),
                                      Value::real(static_cast<double>(config.beta))});
            }

            /** @brief Builds "suppression"; the exponent only under a criterion that reads it. */
            Value suppression_value(const FreeTask& task)
            {
                const CriterionName* const row = criterion_by_value(task.criterion);
                std::vector<std::string> keys = {"criterion", "threshold"};
                std::vector<Value> values = {
                    Value::string(std::string(row->name)),
                    Value::real(static_cast<double>(task.suppression_threshold))};
                if (row->reads_exponent)
                {
                    keys.emplace_back("distance_exponent");
                    values.push_back(Value::real(static_cast<double>(task.distance_exponent)));
                }
                return Value::object(std::move(keys), std::move(values));
            }
        } // namespace

        Result<FreeTask> read_free_task(const Config::Value& task) noexcept
        {
            const SectionReader reader(task, K_TASK);
            SA_TRY(reader.check_object());

            SA_TRY_ASSIGN(const Value* const type, reader.find_of("type", Kind::String));
            SA_TRY_ASSIGN(const std::string& stated_type, type->as_string(K_TASK));
            SA_FAIL_IF(stated_type != K_FREE_TASK_TYPE, StatusCode::InvalidDescription,
                       "{} is '{}'; an anchor-free task states '{}'", member_of(K_TASK, "type"),
                       stated_type, K_FREE_TASK_TYPE);
            SA_TRY(reader.check_keys(KeySpan(K_TASK_KEYS, std::size(K_TASK_KEYS))));

            FreeTask result;
            SA_TRY_ASSIGN(const Value* const score, reader.find("score_threshold"));
            SA_TRY_ASSIGN(result.score_threshold,
                          read_float32(*score, reader.member("score_threshold")));
            SA_TRY(read_suppression(task, result));
            SA_TRY(read_assignment(task, result));
            SA_TRY(validate_free_task(result));
            return Result<FreeTask>(std::move(result));
        }

        Result<Config::Value> write_free_task(const FreeTask& task) noexcept
        {
            SA_TRY(validate_free_task(task));

            Value assignment = Value::object({"one_to_many", "one_to_one"},
                                             {branch_value(task.one_to_many),
                                              branch_value(task.one_to_one)});
            return Value::object(
                {"type", "score_threshold", "suppression", "assignment"},
                {Value::string(std::string(K_FREE_TASK_TYPE)),
                 Value::real(static_cast<double>(task.score_threshold)), suppression_value(task),
                 std::move(assignment)});
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI
