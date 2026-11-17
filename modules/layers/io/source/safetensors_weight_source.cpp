/**************************************************************************/
/* safetensors_weight_source.cpp                                          */
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
 * @file safetensors_weight_source.cpp
 * @brief Defines the weight source of a safetensors file over SafetensorsReader.
 * @author Mustafa Garip
 */

#include <SushiAI/io/safetensors_weight_source.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/safetensors_reader.hpp>
#include <SushiAI/io/weight_source.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the bytes of the length field that opens a file. */
            constexpr std::size_t K_LENGTH_BYTES = 8;

            /** @brief Names the byte a header opens with. */
            constexpr unsigned char K_HEADER_OPEN = '{';

            /** @brief Reads the tensors of a safetensors file. */
            class SafetensorsWeightSource final : public WeightSource
            {
                public:
                    /** @brief Takes the bytes the reader was read from, and the reader. */
                    SafetensorsWeightSource(std::vector<std::byte> bytes, SafetensorsReader reader,
                                            std::string origin)
                        : bytes_(std::move(bytes)), reader_(std::move(reader)),
                          origin_(std::move(origin))
                    {
                        for (const SafetensorsTensor& tensor : reader_.tensors())
                        {
                            tensors_.push_back(SourceTensor{
                                tensor.name, tensor.shape,
                                std::string(safetensors_type_name(tensor.type))});
                        }
                    }

                    SafetensorsWeightSource(const SafetensorsWeightSource&) = delete;
                    SafetensorsWeightSource& operator=(const SafetensorsWeightSource&) = delete;
                    SafetensorsWeightSource(SafetensorsWeightSource&&) = delete;
                    SafetensorsWeightSource& operator=(SafetensorsWeightSource&&) = delete;
                    ~SafetensorsWeightSource() override = default;

                    std::string_view origin() const noexcept override
                    {
                        return origin_;
                    }

                    SushiRuntime::span<const SourceTensor> tensors() const noexcept override
                    {
                        return SushiRuntime::span<const SourceTensor>(tensors_.data(),
                                                                      tensors_.size());
                    }

                    Result<std::vector<float>> read_float32(
                        std::size_t index) const noexcept override
                    {
                        // The reader lists the tensors in the order tensors_ copied.
                        return reader_.read_float32(reader_.tensors()[index]);
                    }

                private:
                    /** @brief Holds the file, which the reader's view points into. */
                    std::vector<std::byte> bytes_;

                    /** @brief Holds the reader the values are read through. */
                    SafetensorsReader reader_;

                    /** @brief Names the file in a message. */
                    std::string origin_;

                    /** @brief Holds the tensors as the reader lists them. */
                    std::vector<SourceTensor> tensors_;
            };
        } // namespace

        bool is_safetensors_file(SushiRuntime::span<const std::byte> bytes) noexcept
        {
            if (bytes.size() <= K_LENGTH_BYTES)
                return false;
            std::uint64_t length = 0;
            for (std::size_t i = 0; i < K_LENGTH_BYTES; ++i)
                length |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                          << (8u * i);
            return length <= bytes.size() - K_LENGTH_BYTES &&
                   static_cast<unsigned char>(bytes[K_LENGTH_BYTES]) == K_HEADER_OPEN;
        }

        Result<std::unique_ptr<WeightSource>> open_safetensors_source(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept
        {
            // The reader keeps a view of these bytes; moving a vector keeps its buffer in place.
            SA_TRY_ASSIGN(SafetensorsReader reader,
                          SafetensorsReader::read(
                              SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                              options.origin));
            return Result<std::unique_ptr<WeightSource>>(std::make_unique<SafetensorsWeightSource>(
                std::move(bytes), std::move(reader), std::string(options.origin)));
        }

    } // namespace IO
} // namespace SushiAI
