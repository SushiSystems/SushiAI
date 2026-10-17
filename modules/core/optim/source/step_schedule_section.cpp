/**************************************************************************/
/* step_schedule_section.cpp                                              */
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
 * @file step_schedule_section.cpp
 * @brief Implements the reader and the writer of the schedule section of a training file.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/step_schedule_section.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/failure/status.hpp>
#include "schedule_section.hpp"

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            using Config::Value;
            using Detail::entry_of;
            using Detail::member_of;
            using Detail::read_real;
            using Detail::read_whole;

            /**
             * @brief Reads the list @p member, named @p where, into @p field entry by entry.
             * @param read_entry Reads one entry, named by its second argument.
             * @return Success, or InvalidDescription for a value that is not an array or for
             *         the first entry @p read_entry refuses.
             */
            template<typename Entry>
            Status read_list(const Value& member, std::string_view where,
                             Status (*read_entry)(const Value&, std::string_view, Entry&),
                             std::vector<Entry>& field)
            {
                SA_TRY_ASSIGN(const std::vector<Value>* const listed,
                              Config::read_array(member, where));

                const std::vector<Value>& entries = *listed;
                std::vector<Entry> read(entries.size());
                for (std::size_t i = 0; i < entries.size(); ++i)
                    SA_TRY(read_entry(entries[i], entry_of(where, i), read[i]));

                field = std::move(read);
                return Status();
            }

            /** @brief Builds the list of the reals of @p field. */
            Value reals_value(const std::vector<double>& field)
            {
                std::vector<Value> entries;
                entries.reserve(field.size());
                for (double entry : field)
                    entries.push_back(Value::real(entry));
                return Value::array(std::move(entries));
            }

            /** @brief Builds the list of the whole numbers of @p field. */
            Value wholes_value(const std::vector<std::int64_t>& field)
            {
                std::vector<Value> entries;
                entries.reserve(field.size());
                for (std::int64_t entry : field)
                    entries.push_back(Value::integer(entry));
                return Value::array(std::move(entries));
            }

            /** @brief Lists every key besides "type", in the order the writer states them. */
            const Detail::SectionField<StepSchedule> K_FIELDS[] = {
                {"base_rate",
                 [](const Value& member, std::string_view where, StepSchedule& schedule)
                 {
                     return read_real(member, where, schedule.base_rate);
                 },
                 [](const StepSchedule& schedule)
                 {
                     return Value::real(schedule.base_rate);
                 }},
                {"warmup_iterations",
                 [](const Value& member, std::string_view where, StepSchedule& schedule)
                 {
                     return read_whole(member, where, schedule.warmup_iterations);
                 },
                 [](const StepSchedule& schedule)
                 {
                     return Value::integer(schedule.warmup_iterations);
                 }},
                {"warmup_power",
                 [](const Value& member, std::string_view where, StepSchedule& schedule)
                 {
                     return read_real(member, where, schedule.warmup_power);
                 },
                 [](const StepSchedule& schedule)
                 {
                     return Value::real(schedule.warmup_power);
                 }},
                {"steps",
                 [](const Value& member, std::string_view where, StepSchedule& schedule)
                 {
                     return read_list(member, where, &read_whole, schedule.steps);
                 },
                 [](const StepSchedule& schedule)
                 {
                     return wholes_value(schedule.steps);
                 }},
                {"scales",
                 [](const Value& member, std::string_view where, StepSchedule& schedule)
                 {
                     return read_list(member, where, &read_real, schedule.scales);
                 },
                 [](const StepSchedule& schedule)
                 {
                     return reals_value(schedule.scales);
                 }}};

            /**
             * @brief Checks the numbers JSON cannot hold, then what the schedule's own check
             *        refuses.
             * @return Success, or InvalidDescription naming the key of a rate, power or
             *         scale that is not finite, or the failure of validate_step_schedule
             *         named as the section's.
             */
            Status validate_section(const StepSchedule& schedule)
            {
                SA_FAIL_IF(!std::isfinite(schedule.base_rate), StatusCode::InvalidDescription,
                           "{} must be finite; got {}", member_of("base_rate"),
                           schedule.base_rate);
                SA_FAIL_IF(!std::isfinite(schedule.warmup_power), StatusCode::InvalidDescription,
                           "{} must be finite; got {}", member_of("warmup_power"),
                           schedule.warmup_power);

                for (std::size_t i = 0; i < schedule.scales.size(); ++i)
                {
                    SA_FAIL_IF(!std::isfinite(schedule.scales[i]), StatusCode::InvalidDescription,
                               "{} must be finite; got {}", entry_of(member_of("scales"), i),
                               schedule.scales[i]);
                }

                return SA_ANNOTATE(validate_step_schedule(schedule), "{}",
                                   Detail::K_SCHEDULE_SECTION);
            }
        } // namespace

        Result<StepSchedule> read_step_schedule_section(const Config::Value& section)
        {
            StepSchedule result;
            SA_TRY(Detail::read_fields(section, K_STEP_SCHEDULE_TYPE, K_FIELDS, result));
            SA_TRY(validate_section(result));
            return result;
        }

        Result<Config::Value> write_step_schedule_section(const StepSchedule& schedule)
        {
            SA_TRY(validate_section(schedule));
            return Detail::write_fields(K_STEP_SCHEDULE_TYPE, K_FIELDS, schedule);
        }

    } // namespace Optim
} // namespace SushiAI
