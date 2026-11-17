/**************************************************************************/
/* npz_weight_source.cpp                                                  */
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
 * @file npz_weight_source.cpp
 * @brief Defines the weight source of a .npz file over ZipArchive and NpyReader.
 * @author Mustafa Garip
 */

#include <SushiAI/io/npz_weight_source.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/npy_reader.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/io/zip_archive.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the extension every entry of an npz carries. */
            constexpr std::string_view K_NPY_SUFFIX = ".npy";

            /** @brief Returns whether @p name ends in ".npy" after at least one character. */
            bool is_npy_name(std::string_view name) noexcept
            {
                return name.size() > K_NPY_SUFFIX.size() &&
                       name.substr(name.size() - K_NPY_SUFFIX.size()) == K_NPY_SUFFIX;
            }

            /** @brief Reads the arrays of a .npz file. */
            class NpzWeightSource final : public WeightSource
            {
                public:
                    /** @brief Takes the bytes the archive views, the archive and the origin. */
                    NpzWeightSource(std::vector<std::byte> bytes, ZipArchive archive,
                                    std::string origin)
                        : bytes_(std::move(bytes)), archive_(std::move(archive)),
                          origin_(std::move(origin))
                    {
                    }

                    NpzWeightSource(const NpzWeightSource&) = delete;
                    NpzWeightSource& operator=(const NpzWeightSource&) = delete;
                    NpzWeightSource(NpzWeightSource&&) = delete;
                    NpzWeightSource& operator=(NpzWeightSource&&) = delete;
                    ~NpzWeightSource() override = default;

                    /** @brief Reads the header of every entry and lists its array. */
                    Status collect() noexcept
                    {
                        for (const ZipEntry& entry : archive_.entries())
                        {
                            SA_FAIL_IF(!is_npy_name(entry.name), StatusCode::InvalidFile,
                                       "'{}': the entry '{}' is no .npy file", origin_,
                                       entry.name);
                            SA_TRY_ASSIGN(NpyReader reader,
                                          NpyReader::read(archive_.contents(entry),
                                                          origin_ + ":" + entry.name));
                            tensors_.push_back(SourceTensor{
                                entry.name.substr(0, entry.name.size() - K_NPY_SUFFIX.size()),
                                reader.array().shape,
                                std::string(npy_type_name(reader.array().type))});
                            readers_.push_back(std::move(reader));
                        }
                        return Status();
                    }

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
                        return Result<std::vector<float>>(readers_[index].read_float32());
                    }

                private:
                    /** @brief Holds the file, which the archive's stored entries point into. */
                    std::vector<std::byte> bytes_;

                    /** @brief Holds the archive, and the bytes of its deflated entries. */
                    ZipArchive archive_;

                    /** @brief Names the file in a message. */
                    std::string origin_;

                    /** @brief Holds the tensors in the order of the entries. */
                    std::vector<SourceTensor> tensors_;

                    /** @brief Holds the reader of each entry, at the position of its tensor. */
                    std::vector<NpyReader> readers_;
            };
        } // namespace

        bool is_npz_file(SushiRuntime::span<const std::byte> bytes) noexcept
        {
            Result<ZipArchive> archive = ZipArchive::read(bytes, std::string_view());
            if (!archive.has_value() || archive.value().entries().empty())
                return false;
            for (const ZipEntry& entry : archive.value().entries())
            {
                if (!is_npy_name(entry.name))
                    return false;
            }
            return true;
        }

        Result<std::unique_ptr<WeightSource>> open_npz_source(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept
        {
            // The archive keeps views of these bytes; moving a vector keeps its buffer in place.
            SA_TRY_ASSIGN(ZipArchive archive,
                          ZipArchive::read(
                              SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                              options.origin));
            auto source = std::make_unique<NpzWeightSource>(std::move(bytes), std::move(archive),
                                                            std::string(options.origin));
            SA_TRY(source->collect());
            return Result<std::unique_ptr<WeightSource>>(std::move(source));
        }

    } // namespace IO
} // namespace SushiAI
