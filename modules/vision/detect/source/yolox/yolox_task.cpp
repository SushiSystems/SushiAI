/**************************************************************************/
/* yolox_task.cpp                                                         */
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
 * @file yolox_task.cpp
 * @brief Implements the reader and the writer of a YOLOX task's section.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/yolox/yolox_task.hpp>

#include <cmath>
#include <iterator>
#include <string>
#include <utility>

#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Yolox
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
            constexpr std::string_view K_TASK_KEYS[] = {"type", "score_threshold", "suppression"};

            /** @brief Checks the score threshold of @p task. */
            Status validate_score(const YoloxTask& task)
            {
                SA_FAIL_IF(!std::isfinite(task.score_threshold) || task.score_threshold < 0.0f
                               || task.score_threshold > 1.0f,
                           StatusCode::InvalidDescription, "{} must be from 0 to 1; got {}",
                           member_of(K_TASK, "score_threshold"), task.score_threshold);
                return Status();
            }
        } // namespace

        Result<YoloxTask> read_yolox_task(const Config::Value& task) noexcept
        {
            const SectionReader reader(task, K_TASK);
            SA_TRY(reader.check_object());

            SA_TRY_ASSIGN(const Value* const type, reader.find_of("type", Kind::String));
            SA_TRY_ASSIGN(const std::string& stated_type, type->as_string(K_TASK));
            SA_FAIL_IF(stated_type != K_YOLOX_TASK_TYPE, StatusCode::InvalidDescription,
                       "{} is '{}'; a YOLOX task states '{}'", member_of(K_TASK, "type"),
                       stated_type, K_YOLOX_TASK_TYPE);
            SA_TRY(reader.check_keys(KeySpan(K_TASK_KEYS, std::size(K_TASK_KEYS))));

            YoloxTask result;
            SA_TRY_ASSIGN(const Value* const score, reader.find("score_threshold"));
            SA_TRY_ASSIGN(result.score_threshold,
                          read_float32(*score, reader.member("score_threshold")));
            SA_TRY_ASSIGN(result.suppression, read_suppression_section(task, K_TASK));
            SA_TRY(validate_score(result));
            return Result<YoloxTask>(std::move(result));
        }

        Result<Config::Value> write_yolox_task(const YoloxTask& task) noexcept
        {
            SA_TRY(validate_score(task));
            SA_TRY_ASSIGN(Value suppression, write_suppression_section(task.suppression, K_TASK));

            return Value::object({"type", "score_threshold", "suppression"},
                                 {Value::string(std::string(K_YOLOX_TASK_TYPE)),
                                  Value::real(static_cast<double>(task.score_threshold)),
                                  std::move(suppression)});
        }

    } // namespace Vision::Detect::Yolox
} // namespace SushiAI
