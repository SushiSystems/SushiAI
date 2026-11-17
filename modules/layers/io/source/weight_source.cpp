/**************************************************************************/
/* weight_source.cpp                                                      */
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
 * @file weight_source.cpp
 * @brief Defines the weight source registry, the sniffing of a container and the reading of a file.
 * @author Mustafa Garip
 */

#include <SushiAI/io/weight_source.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Joins the formats of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_formats(SushiRuntime::span<const WeightSourceEntry> rows)
            {
                std::string listed;
                for (const WeightSourceEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.format).append("'");
                }

                return listed;
            }

            /** @brief Returns the whole contents of @p in as bytes. */
            [[nodiscard]] std::vector<std::byte> slurp(std::ifstream& in)
            {
                std::vector<std::byte> bytes;
                in.seekg(0, std::ios::end);
                const std::streamoff size = in.tellg();
                if (size <= 0)
                    return bytes;
                in.seekg(0, std::ios::beg);
                bytes.resize(static_cast<std::size_t>(size));
                in.read(reinterpret_cast<char*>(bytes.data()), size);
                return bytes;
            }
        } // namespace

        Status WeightSourceRegistry::add(const WeightSourceEntry& row) noexcept
        {
            SA_FATAL_IF(row.format.empty(), "A weight source row needs a non-empty format");
            SA_FATAL_IF(row.matches == nullptr || row.open == nullptr,
                        "The weight source row '{}' lacks its matcher or its opener", row.format);

            for (const WeightSourceEntry& held : rows_)
            {
                SA_FAIL_IF(held.format == row.format, StatusCode::InvalidDescription,
                           "weight source format '{}' is already registered", row.format);
            }

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        SushiRuntime::span<const WeightSourceEntry> WeightSourceRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const WeightSourceEntry>(rows_.data(), rows_.size());
        }

        Result<std::unique_ptr<WeightSource>> open_weight_source(
            const WeightSourceRegistry& registry, std::vector<std::byte> bytes,
            const WeightSourceOptions& options) noexcept
        {
            const SushiRuntime::span<const std::byte> view(bytes.data(), bytes.size());
            const WeightSourceEntry* match = nullptr;
            for (const WeightSourceEntry& row : registry.rows())
            {
                if (row.matches(view))
                {
                    match = &row;
                    break;
                }
            }

            SA_FAIL_IF(match == nullptr, StatusCode::InvalidFile,
                       "'{}' is not a weight file this build reads; the formats are {}",
                       options.origin, list_formats(registry.rows()));
            return match->open(std::move(bytes), options);
        }

        Result<std::unique_ptr<WeightSource>> open_weight_file(
            const WeightSourceRegistry& registry, const std::filesystem::path& path,
            WeightSourceOptions options) noexcept
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());
            std::vector<std::byte> bytes = slurp(in);
            SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "could not read '{}'", path.string());

            const std::string origin = path.string();
            if (options.origin.empty())
                options.origin = origin;
            return open_weight_source(registry, std::move(bytes), options);
        }

    } // namespace IO
} // namespace SushiAI
