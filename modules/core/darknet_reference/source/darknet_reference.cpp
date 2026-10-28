/**************************************************************************/
/* darknet_reference.cpp                                                  */
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
 * @file darknet_reference.cpp
 * @brief Implements the readers of the files a Darknet run dumped.
 * @author Mustafa Garip
 */

#include <SushiAI/darknet_reference/darknet_reference.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds the folder of the data directory that holds one folder per run. */
            constexpr std::string_view K_REFERENCE_FOLDER = "references";

            /** @brief Holds the bytes of one stored element. */
            constexpr std::uintmax_t K_ELEMENT_BYTES = 4;

            /** @brief Holds the largest extent and element count the loader accepts. */
            constexpr std::int64_t K_MAX_ELEMENTS = std::int64_t{1} << 31;

            /** @brief Holds the axes of a feature map. */
            constexpr std::size_t K_FEATURE_AXES = 3;

            /** @brief Returns the value of K_REFERENCE_DATA_VARIABLE; empty when it is unset. */
            std::string read_data_directory()
            {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                const char* const value = std::getenv(K_REFERENCE_DATA_VARIABLE);
#pragma clang diagnostic pop
                return value == nullptr ? std::string() : std::string(value);
            }

            /** @brief Locates @p relative under the directory the environment names. */
            DarknetFile locate(const std::filesystem::path& relative)
            {
                DarknetFile file;
                const std::string directory = read_data_directory();
                if (directory.empty())
                {
                    file.reason = std::string(K_REFERENCE_DATA_VARIABLE) + " is not set, so " +
                                  relative.generic_string() + " cannot be located";
                    return file;
                }

                file.path = std::filesystem::path(directory) / relative;
                std::error_code ignored;
                if (!std::filesystem::is_regular_file(file.path, ignored))
                {
                    file.reason = "the reference file is missing: " + file.path.string();
                    return file;
                }

                file.found = true;
                return file;
            }

            /** @brief Checks that the string @p key of @p root is @p expected. */
            Status expect_text(const Config::Value& root, std::string_view key,
                               std::string_view expected, std::string_view origin)
            {
                SA_FAIL_IF(!root.has(key), StatusCode::InvalidFile, "{} has no '{}'", origin, key);
                SA_TRY_ASSIGN(const Config::Value& value, root.at(key, origin));
                SA_FAIL_IF(value.kind() != Config::Kind::String, StatusCode::InvalidFile,
                           "'{}' of {} is a {}, not a string", key, origin,
                           Config::kind_name(value.kind()));
                SA_TRY_ASSIGN(const std::string& text, value.as_string(origin));
                SA_FAIL_IF(text != expected, StatusCode::InvalidFile,
                           "'{}' of {} is '{}', and the loader reads '{}' only", key, origin,
                           text, expected);

                return Status();
            }

            /**
             * @brief Reads the `shape` of @p root into @p extents.
             * @return Success, or InvalidFile for a missing shape, an entry that is not a whole
             *         number in [0, K_MAX_ELEMENTS], or a product above K_MAX_ELEMENTS.
             */
            Status read_extents(const Config::Value& root, std::string_view origin,
                                std::vector<std::int64_t>& extents)
            {
                SA_FAIL_IF(!root.has("shape"), StatusCode::InvalidFile, "{} has no 'shape'",
                           origin);
                SA_TRY_ASSIGN(const Config::Value& shape, root.at("shape", origin));
                SA_FAIL_IF(shape.kind() != Config::Kind::Array, StatusCode::InvalidFile,
                           "'shape' of {} is a {}, not an array", origin,
                           Config::kind_name(shape.kind()));

                std::int64_t count = 1;
                extents.clear();
                SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, shape.as_array(origin));
                for (const Config::Value& entry : entries)
                {
                    SA_FAIL_IF(entry.kind() != Config::Kind::Int, StatusCode::InvalidFile,
                               "'shape' of {} holds a {}, not a whole number", origin,
                               Config::kind_name(entry.kind()));
                    SA_TRY_ASSIGN(const std::int64_t extent, entry.as_int(origin));
                    SA_FAIL_IF(extent < 0 || extent > K_MAX_ELEMENTS, StatusCode::InvalidFile,
                               "'shape' of {} holds the extent {}", origin, extent);
                    count *= extent;
                    SA_FAIL_IF(count > K_MAX_ELEMENTS, StatusCode::InvalidFile,
                               "'shape' of {} names more than {} elements", origin,
                               K_MAX_ELEMENTS);
                    extents.push_back(extent);
                }

                return Status();
            }

            /** @brief Returns the product of @p extents; 1 for no extent. */
            std::size_t count_elements(const std::vector<std::int64_t>& extents) noexcept
            {
                std::size_t count = 1;
                for (const std::int64_t extent : extents)
                    count *= static_cast<std::size_t>(extent);

                return count;
            }

            /**
             * @brief Reads @p count little-endian float32 values, the whole of the file @p path.
             * @return The values; FileNotFound when the file cannot be opened; InvalidFile when
             *         it does not hold exactly @p count values or a read fails.
             */
            Result<std::vector<float>> read_floats(const std::filesystem::path& path,
                                                   std::size_t count)
            {
                std::error_code failure;
                const std::uintmax_t bytes = std::filesystem::file_size(path, failure);
                SA_FAIL_IF(static_cast<bool>(failure), StatusCode::FileNotFound,
                           "could not measure '{}': {}", path.string(), failure.message());
                SA_FAIL_IF(bytes != count * K_ELEMENT_BYTES, StatusCode::InvalidFile,
                           "'{}' holds {} bytes, but its shape names {} values of {} bytes",
                           path.string(), bytes, count, K_ELEMENT_BYTES);

                std::ifstream in(path, std::ios::binary);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                           path.string());
                std::vector<unsigned char> raw(static_cast<std::size_t>(bytes));
                in.read(reinterpret_cast<char*>(raw.data()),
                        static_cast<std::streamsize>(raw.size()));
                SA_FAIL_IF(!in && !raw.empty(), StatusCode::InvalidFile,
                           "'{}' ended before its {} bytes", path.string(), bytes);

                std::vector<float> values(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const unsigned char* const at = raw.data() + i * K_ELEMENT_BYTES;
                    const std::uint32_t bits =
                        static_cast<std::uint32_t>(at[0]) | static_cast<std::uint32_t>(at[1]) << 8 |
                        static_cast<std::uint32_t>(at[2]) << 16 |
                        static_cast<std::uint32_t>(at[3]) << 24;
                    std::memcpy(&values[i], &bits, sizeof(bits));
                }

                return Result<std::vector<float>>(std::move(values));
            }

            /**
             * @brief Moves the `[C, H, W]` values of @p tensor to `[H, W, C]`, extents included.
             * @return Success, or Unsupported for an `order` other than `CHW` over three axes.
             */
            Status move_channels_last(DarknetTensor& tensor, std::string_view origin)
            {
                SA_TRY(expect_text(tensor.description, "order", "CHW", origin));
                SA_FAIL_IF(tensor.extents.size() != K_FEATURE_AXES, StatusCode::Unsupported,
                           "{} orders {} axes as CHW", origin, tensor.extents.size());

                const std::size_t channels = static_cast<std::size_t>(tensor.extents[0]);
                const std::size_t height = static_cast<std::size_t>(tensor.extents[1]);
                const std::size_t width = static_cast<std::size_t>(tensor.extents[2]);
                std::vector<float> moved(tensor.values.size());
                for (std::size_t channel = 0; channel < channels; ++channel)
                {
                    for (std::size_t row = 0; row < height; ++row)
                    {
                        const float* const from =
                            tensor.values.data() + (channel * height + row) * width;
                        for (std::size_t column = 0; column < width; ++column)
                            moved[(row * width + column) * channels + channel] = from[column];
                    }
                }

                tensor.values = std::move(moved);
                tensor.extents = {tensor.extents[1], tensor.extents[2], tensor.extents[0]};
                return Status();
            }
        } // namespace

        DarknetFile find_darknet_file(std::string_view file_name) noexcept
        {
            return locate(std::filesystem::path(file_name));
        }

        DarknetReference find_darknet_reference(std::string_view model, std::string_view name) noexcept
        {
            const std::filesystem::path folder =
                std::filesystem::path(K_REFERENCE_FOLDER) / std::filesystem::path(model);
            const DarknetFile description = locate(folder / (std::string(name) + ".json"));
            const DarknetFile payload = locate(folder / (std::string(name) + ".bin"));

            DarknetReference reference;
            reference.description = description.path;
            reference.payload = payload.path;
            reference.found = description.found && payload.found;
            if (!reference.found)
                reference.reason = description.found ? payload.reason : description.reason;

            return reference;
        }

        Result<DarknetTensor> load_darknet_tensor(const DarknetReference& reference) noexcept
        {
            SA_FAIL_IF(!reference.found, StatusCode::FileNotFound, "{}", reference.reason);

            const std::string origin = reference.description.string();
            DarknetTensor tensor;
            SA_TRY_ASSIGN(tensor.description, Config::parse_json_file(reference.description));
            SA_TRY(expect_text(tensor.description, "dtype", "float32", origin));
            SA_TRY(expect_text(tensor.description, "byte_order", "little", origin));
            SA_TRY(read_extents(tensor.description, origin, tensor.extents));
            SA_TRY_ASSIGN(tensor.values,
                          read_floats(reference.payload, count_elements(tensor.extents)));
            if (tensor.description.has("order"))
                SA_TRY(move_channels_last(tensor, origin));

            return Result<DarknetTensor>(std::move(tensor));
        }

    } // namespace Test
} // namespace SushiAI
