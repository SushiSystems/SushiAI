/**************************************************************************/
/* safetensors_reader.cpp                                                 */
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
 * @file safetensors_reader.cpp
 * @brief Defines the reader of a safetensors file and the widening of its tensors to float32.
 * @author Mustafa Garip
 */

#include <SushiAI/io/safetensors_reader.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include "float_widening.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Names the header entry that holds free-form metadata, not a tensor. */
            constexpr std::string_view K_METADATA_KEY = "__metadata__";

            /** @brief Holds the bytes of the length field that opens a file. */
            constexpr std::size_t K_LENGTH_BYTES = 8;

            /** @brief Pairs a dtype name of the file with the type the reader gives it. */
            struct TypeRow
            {
                std::string_view name;  /**< The spelling in the header. */
                SafetensorsType type;   /**< The type it names. */
                std::size_t bytes;      /**< The bytes of one element. */
            };

            /** @brief Lists the dtypes the reader reads. */
            constexpr TypeRow K_TYPES[] = {
                {"F16", SafetensorsType::Float16, 2},
                {"BF16", SafetensorsType::BFloat16, 2},
                {"F32", SafetensorsType::Float32, 4},
                {"F64", SafetensorsType::Float64, 8},
            };

            /** @brief Holds a tensor with the range its header entry gave, relative to the data. */
            struct Pending
            {
                SafetensorsTensor tensor; /**< The tensor, its offset still relative to the data. */
                std::size_t begin = 0;    /**< The first byte of its range. */
                std::size_t end = 0;      /**< One past the last byte of its range. */
            };

            /** @brief Returns the little-endian unsigned 64-bit integer at the start of @p bytes. */
            std::uint64_t load_length(SushiRuntime::span<const std::byte> bytes) noexcept
            {
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < K_LENGTH_BYTES; ++i)
                    value |= static_cast<std::uint64_t>(static_cast<unsigned>(bytes[i])) << (8u * i);
                return value;
            }

            /** @brief Returns the whole numbers of the array @p value, or a failure naming @p name. */
            Result<std::vector<std::int64_t>> read_integers(const Config::Value& value,
                                                            std::string_view what,
                                                            std::string_view name)
            {
                SA_FAIL_IF(value.kind() != Config::Kind::Array, StatusCode::InvalidFile,
                           "tensor '{}': its {} is missing or is no array", name, what);
                SA_TRY_ASSIGN(const std::vector<Config::Value>& items, value.as_array(what));
                std::vector<std::int64_t> out;
                out.reserve(items.size());
                for (const Config::Value& item : items)
                {
                    SA_FAIL_IF(item.kind() != Config::Kind::Int, StatusCode::InvalidFile,
                               "tensor '{}': its {} holds a value that is no whole number", name,
                               what);
                    SA_TRY_ASSIGN(const std::int64_t number, item.as_int(what));
                    out.push_back(number);
                }
                return out;
            }

            /** @brief Returns the row of the dtype spelled @p text, or null when there is none. */
            const TypeRow* find_type(std::string_view text) noexcept
            {
                for (const TypeRow& row : K_TYPES)
                {
                    if (row.name == text)
                        return &row;
                }
                return nullptr;
            }

            /** @brief Reads the header entry @p entry of the tensor @p name into a pending tensor. */
            Result<Pending> read_entry(const Config::Value& entry, const std::string& name,
                                       std::size_t data_size)
            {
                SA_FAIL_IF(entry.kind() != Config::Kind::Object, StatusCode::InvalidFile,
                           "tensor '{}': its header entry is no object", name);
                const Config::Value missing;
                const Config::Value& dtype = entry.at_or("dtype", missing);
                SA_FAIL_IF(dtype.kind() != Config::Kind::String, StatusCode::InvalidFile,
                           "tensor '{}': its dtype is missing or is no string", name);
                SA_TRY_ASSIGN(const std::string& type_text, dtype.as_string("dtype"));
                const TypeRow* const row = find_type(type_text);
                SA_FAIL_IF(row == nullptr, StatusCode::Unsupported,
                           "tensor '{}': its dtype '{}' is not one the reader reads (F16, BF16, "
                           "F32 and F64 are)",
                           name, type_text);

                SA_TRY_ASSIGN(const std::vector<std::int64_t> extents,
                              read_integers(entry.at_or("shape", missing), "shape", name));
                SA_FAIL_IF(extents.size() > static_cast<std::size_t>(K_MAX_RANK),
                           StatusCode::InvalidFile,
                           "tensor '{}': its shape has {} extents, and at most {} are read", name,
                           extents.size(), K_MAX_RANK);
                std::size_t count = 1;
                for (const std::int64_t extent : extents)
                {
                    SA_FAIL_IF(extent <= 0, StatusCode::InvalidFile,
                               "tensor '{}': its shape holds the extent {}, and an extent must be "
                               "positive",
                               name, extent);
                    SA_FAIL_IF(static_cast<std::size_t>(extent) > data_size / count,
                               StatusCode::InvalidFile,
                               "tensor '{}': its shape needs more elements than the {} data "
                               "bytes of the file hold",
                               name, data_size);
                    count *= static_cast<std::size_t>(extent);
                }

                SA_TRY_ASSIGN(const std::vector<std::int64_t> range,
                              read_integers(entry.at_or("data_offsets", missing), "data_offsets",
                                            name));
                SA_FAIL_IF(range.size() != 2, StatusCode::InvalidFile,
                           "tensor '{}': its data_offsets hold {} values, and two are needed",
                           name, range.size());
                SA_FAIL_IF(range[0] < 0 || range[1] < range[0], StatusCode::InvalidFile,
                           "tensor '{}': its data_offsets [{}, {}] run backwards or start below "
                           "zero",
                           name, range[0], range[1]);
                SA_FAIL_IF(static_cast<std::uint64_t>(range[1]) > data_size, StatusCode::InvalidFile,
                           "tensor '{}': its data_offsets [{}, {}] end outside the {} data bytes "
                           "of the file",
                           name, range[0], range[1], data_size);

                Pending out;
                out.begin = static_cast<std::size_t>(range[0]);
                out.end = static_cast<std::size_t>(range[1]);
                SA_FAIL_IF(out.end - out.begin != count * row->bytes, StatusCode::InvalidFile,
                           "tensor '{}': its shape and {} need {} bytes, and its data_offsets "
                           "[{}, {}] hold {}",
                           name, row->name, count * row->bytes, range[0], range[1],
                           out.end - out.begin);
                out.tensor.name = name;
                out.tensor.type = row->type;
                SA_TRY_ASSIGN(out.tensor.shape,
                              Shape::make(SushiRuntime::span<const std::int64_t>(extents.data(),
                                                                                 extents.size())));
                out.tensor.length = out.end - out.begin;
                return out;
            }
        } // namespace

        std::size_t safetensors_type_size(SafetensorsType type) noexcept
        {
            for (const TypeRow& row : K_TYPES)
            {
                if (row.type == type)
                    return row.bytes;
            }
            return 0;
        }

        std::string_view safetensors_type_name(SafetensorsType type) noexcept
        {
            for (const TypeRow& row : K_TYPES)
            {
                if (row.type == type)
                    return row.name;
            }
            return "";
        }

        SafetensorsReader::SafetensorsReader(SushiRuntime::span<const std::byte> bytes,
                                             std::vector<SafetensorsTensor> tensors) noexcept
            : bytes_(bytes), tensors_(std::move(tensors))
        {

        }

        Result<SafetensorsReader> SafetensorsReader::read(SushiRuntime::span<const std::byte> bytes,
                                                          std::string_view origin) noexcept
        {
            SA_FAIL_IF(bytes.size() < K_LENGTH_BYTES, StatusCode::InvalidFile,
                       "'{}' holds {} bytes, too few for the 8-byte length of a safetensors header",
                       origin, bytes.size());
            const std::uint64_t header_size = load_length(bytes);
            SA_FAIL_IF(header_size > bytes.size() - K_LENGTH_BYTES, StatusCode::InvalidFile,
                       "'{}': the header states {} bytes, and the file holds {} after its length "
                       "field",
                       origin, header_size, bytes.size() - K_LENGTH_BYTES);
            const std::size_t header_bytes = static_cast<std::size_t>(header_size);
            const std::size_t data_start = K_LENGTH_BYTES + header_bytes;
            const std::size_t data_size = bytes.size() - data_start;

            const std::string_view text(
                reinterpret_cast<const char*>(bytes.data() + K_LENGTH_BYTES), header_bytes);
            SA_TRY_ASSIGN(const Config::Value root,
                          Config::parse_json(text, std::string(origin) + " header"));
            SA_FAIL_IF(root.kind() != Config::Kind::Object, StatusCode::InvalidFile,
                       "'{}': the header is no JSON object", origin);

            std::vector<Pending> pending;
            const std::vector<std::string>& keys = root.keys();
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == K_METADATA_KEY)
                    continue;
                SA_TRY_ASSIGN(Pending next, read_entry(root.values()[i], keys[i], data_size));
                pending.push_back(std::move(next));
            }

            std::sort(pending.begin(), pending.end(), [](const Pending& a, const Pending& b)
            {
                if (a.begin != b.begin)
                    return a.begin < b.begin;
                return a.tensor.name < b.tensor.name;
            });
            std::vector<SafetensorsTensor> tensors;
            tensors.reserve(pending.size());
            for (std::size_t i = 0; i < pending.size(); ++i)
            {
                SA_FAIL_IF(i > 0 && pending[i].begin < pending[i - 1].end, StatusCode::InvalidFile,
                           "'{}': tensors '{}' and '{}' overlap: the first ends at byte {} and "
                           "the second starts at {}",
                           origin, tensors.back().name, pending[i].tensor.name,
                           pending[i - 1].end, pending[i].begin);
                pending[i].tensor.offset = data_start + pending[i].begin;
                tensors.push_back(std::move(pending[i].tensor));
            }
            return SafetensorsReader(bytes, std::move(tensors));
        }

        SushiRuntime::span<const SafetensorsTensor> SafetensorsReader::tensors() const noexcept
        {
            return SushiRuntime::span<const SafetensorsTensor>(tensors_.data(), tensors_.size());
        }

        const SafetensorsTensor* SafetensorsReader::find(std::string_view name) const noexcept
        {
            for (const SafetensorsTensor& tensor : tensors_)
            {
                if (tensor.name == name)
                    return &tensor;
            }
            return nullptr;
        }

        SushiRuntime::span<const std::byte> SafetensorsReader::contents(
            const SafetensorsTensor& tensor) const noexcept
        {
            return SushiRuntime::span<const std::byte>(bytes_.data() + tensor.offset,
                                                       tensor.length);
        }

        std::vector<float> SafetensorsReader::read_float32(
            const SafetensorsTensor& tensor) const noexcept
        {
            const SushiRuntime::span<const std::byte> raw = contents(tensor);
            const std::size_t width = safetensors_type_size(tensor.type);
            std::vector<float> out(raw.size() / width);
            for (std::size_t n = 0; n < out.size(); ++n)
            {
                const std::byte* const at = raw.data() + n * width;
                std::uint64_t bits = 0;
                for (std::size_t i = 0; i < width; ++i)
                    bits |= static_cast<std::uint64_t>(static_cast<unsigned>(at[i])) << (8u * i);
                switch (tensor.type)
                {
                    case SafetensorsType::Float16:
                        out[n] = Detail::widen_half(static_cast<std::uint16_t>(bits));
                        break;
                    case SafetensorsType::BFloat16:
                        out[n] = Detail::widen_bfloat16(static_cast<std::uint16_t>(bits));
                        break;
                    case SafetensorsType::Float32:
                    {
                        const std::uint32_t word = static_cast<std::uint32_t>(bits);
                        std::memcpy(&out[n], &word, sizeof(float));
                        break;
                    }
                    case SafetensorsType::Float64:
                    {
                        double wide = 0.0;
                        std::memcpy(&wide, &bits, sizeof(double));
                        out[n] = static_cast<float>(wide);
                        break;
                    }
                }
            }
            return out;
        }
    } // namespace IO
} // namespace SushiAI
