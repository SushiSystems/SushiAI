/**************************************************************************/
/* profile_records.hpp                                                    */
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
 * @file profile_records.hpp
 * @brief Declares the helpers that tests use to start a profiling session and read its record.
 * @author Mustafa Garip
 * @see modules/core/profile_records/README.md
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <SushiAI/profile/zone_groups.hpp>
#include <SushiProfiler/config/profiler_config.hpp>
#include <SushiProfiler/record/run_record.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Returns a configuration that follows @p timeline for @p warmup warm-up periods
         *        and @p active recorded ones, with no linger and nothing written.
         */
        inline SushiProfiler::ProfilerConfig following_timeline(std::string_view timeline,
                                                                std::uint32_t warmup,
                                                                std::uint32_t active)
        {
            SushiProfiler::ProfilerConfig config;
            config.timeline = std::string(timeline);
            config.window.warmup = warmup;
            config.window.active = active;
            config.buffers.linger_ms = 0U;
            return config;
        }

        /** @brief Returns what the zones of @p period ran, as site name to group. */
        inline std::map<std::string, std::string> zones_of(const SushiProfiler::RunRecord& record,
                                                           const SushiProfiler::PeriodRecord& period)
        {
            std::map<std::string, std::string> zones;
            for (const SushiProfiler::ZoneTree& tree : period.zones)
            {
                for (const SushiProfiler::ZoneNode& node : tree.nodes)
                {
                    const SushiProfiler::SiteRecord& site = record.sites[node.site];
                    zones[site.name] = site.group;
                }
            }
            return zones;
        }

        /** @brief Returns the zones of @p period whose group is one of @p groups. */
        template<std::size_t Count>
        std::map<std::string, std::string> zones_in_groups(
            const SushiProfiler::RunRecord& record, const SushiProfiler::PeriodRecord& period,
            const std::array<std::string_view, Count>& groups)
        {
            std::map<std::string, std::string> kept;
            for (const auto& [name, group] : zones_of(record, period))
            {
                if (std::find(groups.begin(), groups.end(), group) != groups.end())
                    kept.emplace(name, group);
            }
            return kept;
        }

        /**
         * @brief Returns the zones of @p period whose group is in Profile::K_ZONE_GROUPS, which
         *        leaves out the zones of the libraries below SushiAI, such as the groups
         *        "Runtime" and "BLAS".
         */
        inline std::map<std::string, std::string> sushiai_zones_of(
            const SushiProfiler::RunRecord& record, const SushiProfiler::PeriodRecord& period)
        {
            return zones_in_groups(record, period, Profile::K_ZONE_GROUPS);
        }

        /** @brief Writes @p text to @p path, making the folder that holds it. */
        inline void write_text(const std::filesystem::path& path, const std::string& text)
        {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream file(path, std::ios::binary);
            file << text;
        }

        /** @brief Returns the text of the file at @p path, or an empty string. */
        inline std::string read_text(const std::filesystem::path& path)
        {
            std::ifstream file(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(file),
                               std::istreambuf_iterator<char>());
        }

        /**
         * @brief Writes a profile configuration file that follows `train.step` for two periods
         *        and writes the record @p record_name under @p folder.
         * @return The path of the configuration file, `profile.json` in @p folder.
         */
        inline std::filesystem::path write_profile_config(const std::filesystem::path& folder,
                                                          std::string_view record_name)
        {
            const std::filesystem::path path = folder / "profile.json";
            write_text(path,
                       "{\"sushiprofiler\": 1, \"timeline\": \"train.step\","
                       " \"window\": {\"warmup\": 0, \"active\": 2},"
                       " \"writers\": {\"record\": {\"path\": \"" + std::string(record_name)
                           + "\"}},"
                       " \"output\": \"" + folder.generic_string() + "\","
                       " \"buffers\": {\"linger_ms\": 0}}");
            return path;
        }

    } // namespace Test
} // namespace SushiAI
