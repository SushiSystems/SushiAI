/**************************************************************************/
/* safetensors_writer.cpp                                                 */
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
 * @file safetensors_writer.cpp
 * @brief Implements the writer of a safetensors file.
 * @author Mustafa Garip
 */

#include <SushiAI/io/safetensors_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the key of the metadata object, which no tensor may take. */
            constexpr std::string_view K_METADATA_KEY = "__metadata__";

            /** @brief Holds the multiple the header length is padded to. */
            constexpr std::size_t K_HEADER_ALIGNMENT = 8;

            /** @brief Returns the name the file spells @p dtype as, or empty for none. */
            [[nodiscard]] std::string_view type_name(DType dtype) noexcept
            {
                switch (dtype)
                {
                    case DType::Half:    return "F16";
                    case DType::Float32: return "F32";
                    case DType::Float64: return "F64";
                    default:             return {};
                }
            }

            /** @brief Returns the header of @p tensors as a JSON value. */
            [[nodiscard]] Result<Config::Value> compose_header(
                SushiRuntime::span<const ImportedTensor> tensors)
            {
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                std::uint64_t offset = 0;
                for (const ImportedTensor& tensor : tensors)
                {
                    const TensorRecord& record = tensor.record;
                    const std::string_view type = type_name(record.dtype);
                    SA_FAIL_IF(type.empty(), StatusCode::Unsupported,
                               "the tensor '{}' is {}, and safetensors is written for float32, "
                               "float64 and half",
                               record.name, dtype_name(record.dtype));
                    SA_FAIL_IF(record.name == K_METADATA_KEY, StatusCode::InvalidDescription,
                               "a tensor may not be named '{}'", record.name);
                    SA_FAIL_IF(tensor.payload.size() != record.payload_bytes(),
                               StatusCode::InvalidDescription,
                               "the tensor '{}' holds {} bytes, but its shape {} needs {}",
                               record.name, tensor.payload.size(), record.shape.to_string(),
                               record.payload_bytes());

                    std::vector<Config::Value> shape;
                    for (std::int32_t axis = 0; axis < record.shape.rank(); ++axis)
                        shape.push_back(Config::Value::integer(record.shape[axis]));
                    const std::uint64_t end = offset + tensor.payload.size();
                    std::vector<Config::Value> range = {
                        Config::Value::integer(static_cast<std::int64_t>(offset)),
                        Config::Value::integer(static_cast<std::int64_t>(end))};
                    offset = end;

                    keys.push_back(record.name);
                    values.push_back(Config::Value::object(
                        {"dtype", "shape", "data_offsets"},
                        {Config::Value::string(std::string(type)),
                         Config::Value::array(std::move(shape)),
                         Config::Value::array(std::move(range))}));
                }

                keys.emplace_back(K_METADATA_KEY);
                values.push_back(Config::Value::object({"format"}, {Config::Value::string("pt")}));
                return Config::Value::object(std::move(keys), std::move(values));
            }
        } // namespace

        Result<std::vector<std::byte>> write_safetensors(
            SushiRuntime::span<const ImportedTensor> tensors) noexcept
        {
            SA_TRY_ASSIGN(const Config::Value header,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]() { return compose_header(tensors); }));
            SA_TRY_ASSIGN(std::string text, Config::write_json(header, 0));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                while (text.size() % K_HEADER_ALIGNMENT != 0)
                    text.push_back(' ');

                std::vector<std::byte> out;
                const std::uint64_t length = text.size();
                for (std::size_t i = 0; i < 8; ++i)
                    out.push_back(static_cast<std::byte>((length >> (8u * i)) & 0xFFu));
                out.insert(out.end(), reinterpret_cast<const std::byte*>(text.data()),
                           reinterpret_cast<const std::byte*>(text.data()) + text.size());
                for (const ImportedTensor& tensor : tensors)
                    out.insert(out.end(), tensor.payload.begin(), tensor.payload.end());
                return out;
            });
        }
    } // namespace IO
} // namespace SushiAI
