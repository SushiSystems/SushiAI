/**************************************************************************/
/* cosine_schedule_section.cpp                                            */
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
 * @file cosine_schedule_section.cpp
 * @brief Implements the reader and the writer of a cosine schedule section.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/cosine_schedule_section.hpp>

#include <cmath>
#include <string_view>
#include <SushiAI/core/status.hpp>
#include "schedule_section.hpp"

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            using Config::Value;
            using Detail::member_of;
            using Detail::read_real;
            using Detail::read_whole;

            /** @brief Lists every key besides "type", in the order the writer states them. */
            const Detail::SectionField<CosineSchedule> K_FIELDS[] = {
                {"base_rate",
                 [](const Value& member, std::string_view where, CosineSchedule& schedule)
                 {
                     return read_real(member, where, schedule.base_rate);
                 },
                 [](const CosineSchedule& schedule)
                 {
                     return Value::real(schedule.base_rate);
                 }},
                {"warmup_iterations",
                 [](const Value& member, std::string_view where, CosineSchedule& schedule)
                 {
                     return read_whole(member, where, schedule.warmup_iterations);
                 },
                 [](const CosineSchedule& schedule)
                 {
                     return Value::integer(schedule.warmup_iterations);
                 }},
                {"total_iterations",
                 [](const Value& member, std::string_view where, CosineSchedule& schedule)
                 {
                     return read_whole(member, where, schedule.total_iterations);
                 },
                 [](const CosineSchedule& schedule)
                 {
                     return Value::integer(schedule.total_iterations);
                 }},
                {"final_rate",
                 [](const Value& member, std::string_view where, CosineSchedule& schedule)
                 {
                     return read_real(member, where, schedule.final_rate);
                 },
                 [](const CosineSchedule& schedule)
                 {
                     return Value::real(schedule.final_rate);
                 }}};

            /**
             * @brief Checks the numbers JSON cannot hold, then what the schedule's own check
             *        refuses.
             * @return Success, or InvalidDescription naming the key of a rate that is not
             *         finite, or the failure of validate_cosine_schedule named as the
             *         section's.
             */
            Status validate_section(const CosineSchedule& schedule)
            {
                SA_FAIL_IF(!std::isfinite(schedule.base_rate), StatusCode::InvalidDescription,
                           "{} must be finite; got {}", member_of("base_rate"),
                           schedule.base_rate);
                SA_FAIL_IF(!std::isfinite(schedule.final_rate), StatusCode::InvalidDescription,
                           "{} must be finite; got {}", member_of("final_rate"),
                           schedule.final_rate);

                return SA_ANNOTATE(validate_cosine_schedule(schedule), "{}",
                                   Detail::K_SCHEDULE_SECTION);
            }
        } // namespace

        Result<CosineSchedule> read_cosine_schedule_section(const Config::Value& section)
        {
            CosineSchedule result;
            SA_TRY(Detail::read_fields(section, K_COSINE_SCHEDULE_TYPE, K_FIELDS, result));
            SA_TRY(validate_section(result));
            return result;
        }

        Result<Config::Value> write_cosine_schedule_section(const CosineSchedule& schedule)
        {
            SA_TRY(validate_section(schedule));
            return Detail::write_fields(K_COSINE_SCHEDULE_TYPE, K_FIELDS, schedule);
        }

    } // namespace Optim
} // namespace SushiAI
