/**************************************************************************/
/* rate_schedule_section.cpp                                              */
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
 * @file rate_schedule_section.cpp
 * @brief Implements the table that reads a schedule section by its type, and its writer.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/rate_schedule_section.hpp>

#include <array>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <SushiAI/core/status.hpp>
#include <SushiAI/optim/cosine_schedule_section.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include "schedule_section.hpp"

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            using Config::Value;

            /** @brief Holds the type a section states and the reader of that type. */
            struct ScheduleKind
            {
                /** @brief Holds the "type" a section of this kind states. */
                std::string_view type;

                /** @brief Reads a section of this kind. */
                Result<RateSchedule> (*read)(const Value&);
            };

            /** @brief Lists every schedule a section may state, one row per RateSchedule type. */
            const ScheduleKind K_KINDS[] = {
                {K_STEP_SCHEDULE_TYPE,
                 [](const Value& section) -> Result<RateSchedule>
                 {
                     SA_TRY_ASSIGN(StepSchedule read, read_step_schedule_section(section));
                     return RateSchedule(std::move(read));
                 }},
                {K_COSINE_SCHEDULE_TYPE,
                 [](const Value& section) -> Result<RateSchedule>
                 {
                     SA_TRY_ASSIGN(CosineSchedule read, read_cosine_schedule_section(section));
                     return RateSchedule(std::move(read));
                 }}};

            /** @brief Returns the type of every row of K_KINDS, quoted, comma separated. */
            [[nodiscard]] std::string list_types()
            {
                std::array<std::string_view, std::size(K_KINDS)> types{};
                for (std::size_t index = 0; index < types.size(); ++index)
                    types[index] = K_KINDS[index].type;
                return Config::list_quoted(Config::KeySpan(types.data(), types.size()));
            }

            /**
             * @brief Returns the "type" @p section states, or "step" when it states none.
             * @return InvalidDescription for a section that is no object or a type that is
             *         no string.
             */
            [[nodiscard]] Result<std::string> stated_type(const Value& section)
            {
                const Config::SectionReader reader(section, Detail::K_SCHEDULE_SECTION);
                SA_TRY(reader.check_object());

                if (!reader.has(Detail::K_SCHEDULE_TYPE_KEY))
                    return std::string(K_STEP_SCHEDULE_TYPE);

                const std::string where = reader.member(Detail::K_SCHEDULE_TYPE_KEY);
                SA_TRY_ASSIGN(const std::string* const stated,
                              Config::read_string(section.at(Detail::K_SCHEDULE_TYPE_KEY, where),
                                                  where));

                return std::string(*stated);
            }

            /** @brief Writes the section of a step schedule. */
            [[nodiscard]] Result<Value> write_held(const StepSchedule& schedule)
            {
                return write_step_schedule_section(schedule);
            }

            /** @brief Writes the section of a cosine schedule. */
            [[nodiscard]] Result<Value> write_held(const CosineSchedule& schedule)
            {
                return write_cosine_schedule_section(schedule);
            }
        } // namespace

        Result<RateSchedule> read_rate_schedule_section(const Config::Value& section)
        {
            SA_TRY_ASSIGN(const std::string type, stated_type(section));

            const ScheduleKind* found = nullptr;
            for (const ScheduleKind& kind : K_KINDS)
            {
                if (kind.type == type)
                    found = &kind;
            }

            SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                       "{} is '{}', which is no schedule; this build has: {}",
                       Detail::member_of(Detail::K_SCHEDULE_TYPE_KEY), type, list_types());
            return found->read(section);
        }

        Result<Config::Value> write_rate_schedule_section(const RateSchedule& schedule)
        {
            return std::visit([](const auto& held)
            {
                return write_held(held);
            }, schedule);
        }

    } // namespace Optim
} // namespace SushiAI
