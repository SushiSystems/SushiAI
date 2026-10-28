/**************************************************************************/
/* schedule_section.hpp                                                   */
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
 * @file schedule_section.hpp
 * @brief Defines what the readers and writers of every schedule section share.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace Detail
        {
            /** @brief Names the section in a diagnostic; every refusal starts with it. */
            inline constexpr std::string_view K_SCHEDULE_SECTION = "schedule";

            /** @brief Names the key that states which schedule the section describes. */
            inline constexpr std::string_view K_SCHEDULE_TYPE_KEY = "type";

            using Config::entry_of;

            /** @brief Returns the text that names the member @p key of the section. */
            [[nodiscard]] inline std::string member_of(std::string_view key)
            {
                return Config::member_of(K_SCHEDULE_SECTION, key);
            }

            /** @brief Reads the number @p member, named @p where, into @p field. */
            [[nodiscard]] inline Status read_real(const Config::Value& member,
                                                  std::string_view where, double& field)
            {
                SA_TRY_ASSIGN(field, Config::read_real(member, where));
                return Status();
            }

            /** @brief Reads the whole number @p member, named @p where, into @p field. */
            [[nodiscard]] inline Status read_whole(const Config::Value& member,
                                                   std::string_view where, std::int64_t& field)
            {
                SA_TRY_ASSIGN(field, Config::read_whole(member, where));
                return Status();
            }

            /** @brief Holds how one key of a section is read into and written from a field. */
            template<typename Schedule>
            struct SectionField
            {
                /** @brief Holds the key, which is the name of the field it states. */
                std::string_view key;

                /** @brief Reads the member, named by the second argument, into the schedule. */
                Status (*read)(const Config::Value&, std::string_view, Schedule&);

                /** @brief Builds the member from the schedule. */
                Config::Value (*write)(const Schedule&);
            };

            /** @brief Returns "type" and the key of every field, in the order a section lists. */
            template<typename Schedule, std::size_t N>
            [[nodiscard]] std::array<std::string_view, N + 1> keys_of(
                const SectionField<Schedule> (&fields)[N])
            {
                std::array<std::string_view, N + 1> keys{};
                keys[0] = K_SCHEDULE_TYPE_KEY;
                for (std::size_t index = 0; index < N; ++index)
                    keys[index + 1] = fields[index].key;
                return keys;
            }

            /**
             * @brief Reads every field @p section states into @p schedule.
             * @return Success, or InvalidDescription for a section that is no object, a key
             *         outside @p fields, another "type" than @p type, or a field's refusal.
             */
            template<typename Schedule, std::size_t N>
            [[nodiscard]] Status read_fields(const Config::Value& section, std::string_view type,
                                             const SectionField<Schedule> (&fields)[N],
                                             Schedule& schedule)
            {
                const Config::SectionReader reader(section, K_SCHEDULE_SECTION);
                const std::array<std::string_view, N + 1> keys = keys_of(fields);
                const std::string what = "a " + std::string(type) + " schedule";
                SA_TRY(reader.check_object());
                SA_TRY(reader.check_keys(Config::KeySpan(keys.data(), keys.size())));
                SA_TRY(reader.check_type(K_SCHEDULE_TYPE_KEY, type, what));

                for (const SectionField<Schedule>& field : fields)
                {
                    if (!section.has(field.key))
                        continue;

                    const std::string where = member_of(field.key);
                    SA_TRY_ASSIGN(const Config::Value& member, section.at(field.key, where));
                    SA_TRY(field.read(member, where, schedule));
                }
                return Status();
            }

            /** @brief Builds the section of @p schedule: "type", then every field in order. */
            template<typename Schedule, std::size_t N>
            [[nodiscard]] Config::Value write_fields(std::string_view type,
                                                     const SectionField<Schedule> (&fields)[N],
                                                     const Schedule& schedule)
            {
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                keys.reserve(N + 1);
                values.reserve(N + 1);

                keys.emplace_back(K_SCHEDULE_TYPE_KEY);
                values.push_back(Config::Value::string(std::string(type)));
                for (const SectionField<Schedule>& field : fields)
                {
                    keys.emplace_back(field.key);
                    values.push_back(field.write(schedule));
                }
                return Config::Value::object(std::move(keys), std::move(values));
            }

        } // namespace Detail
    } // namespace Optim
} // namespace SushiAI
