/**************************************************************************/
/* torch_weight_source.cpp                                                */
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
 * @file torch_weight_source.cpp
 * @brief Defines the weight source of a PyTorch file over TorchArchive.
 * @author Mustafa Garip
 */

#include <SushiAI/io/torch_weight_source.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/torch_archive.hpp>
#include <SushiAI/io/weight_source.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the first bytes of a zip archive: "PK", then 3 and 4. */
            constexpr unsigned char K_ZIP_MAGIC[] = {0x50, 0x4B, 0x03, 0x04};

            /** @brief Reads the tensors of a state dictionary out of a PyTorch archive. */
            class TorchWeightSource final : public WeightSource
            {
                public:
                    /** @brief Takes the bytes the archive was read from, and the archive. */
                    TorchWeightSource(std::vector<std::byte> bytes, TorchArchive archive,
                                      std::string origin) noexcept
                        : bytes_(std::move(bytes)), archive_(std::move(archive)),
                          origin_(std::move(origin))
                    {

                    }

                    TorchWeightSource(const TorchWeightSource&) = delete;
                    TorchWeightSource& operator=(const TorchWeightSource&) = delete;
                    TorchWeightSource(TorchWeightSource&&) = delete;
                    TorchWeightSource& operator=(TorchWeightSource&&) = delete;
                    ~TorchWeightSource() override = default;

                    /** @brief Lists the entries of the dictionary @p key names, or of the root. */
                    Status collect(std::string_view key)
                    {
                        const PickleValue* dictionary = &archive_.root();
                        if (!key.empty())
                        {
                            dictionary = find_entry(*dictionary, key);
                            SA_FAIL_IF(dictionary == nullptr, StatusCode::InvalidFile,
                                       "'{}' holds no state dictionary under the key '{}'",
                                       origin_, key);
                        }
                        SA_FAIL_IF(dictionary->kind != PickleKind::Dict &&
                                       dictionary->kind != PickleKind::Call,
                                   StatusCode::InvalidFile,
                                   "'{}' holds {} where a state dictionary is expected", origin_,
                                   key.empty() ? "a root that is no dictionary"
                                               : "an entry that is no dictionary");

                        for (const auto& entry : dictionary->entries)
                        {
                            SA_FAIL_IF(entry.first->kind != PickleKind::String,
                                       StatusCode::InvalidFile,
                                       "'{}' holds a dictionary key that is no text", origin_);
                            const std::string& name = entry.first->text;
                            SA_FAIL_IF(!TorchArchive::holds_tensor(*entry.second),
                                       StatusCode::InvalidFile,
                                       "'{}': the entry '{}' is no tensor; if the file is a "
                                       "checkpoint, name the key of its state dictionary as "
                                       "\"state_dict\" in the name map",
                                       origin_, name);
                            SA_TRY_ASSIGN(const TorchTensor tensor,
                                          archive_.tensor(*entry.second, name));
                            tensors_.push_back(SourceTensor{name, tensor.shape,
                                                            dtype_name(tensor.dtype)});
                            stored_.push_back(tensor);
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
                        return archive_.read_float32(stored_[index], tensors_[index].name);
                    }

                private:
                    /** @brief Holds the file, which the archive's views point into. */
                    std::vector<std::byte> bytes_;

                    /** @brief Holds the archive the tensors are read from. */
                    TorchArchive archive_;

                    /** @brief Names the file in a message. */
                    std::string origin_;

                    /** @brief Holds the tensors as they are listed. */
                    std::vector<SourceTensor> tensors_;

                    /** @brief Holds where each listed tensor lies, at the same position. */
                    std::vector<TorchTensor> stored_;
            };
        } // namespace

        bool is_torch_file(SushiRuntime::span<const std::byte> bytes) noexcept
        {
            if (bytes.size() < sizeof(K_ZIP_MAGIC))
                return false;
            for (std::size_t i = 0; i < sizeof(K_ZIP_MAGIC); ++i)
            {
                if (static_cast<unsigned char>(bytes[i]) != K_ZIP_MAGIC[i])
                    return false;
            }
            return true;
        }

        Result<std::unique_ptr<WeightSource>> open_torch_source(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept
        {
            // The archive keeps views of these bytes; moving a vector keeps its buffer in place.
            SA_TRY_ASSIGN(TorchArchive archive,
                          TorchArchive::read(
                              SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()), {},
                              options.origin));
            auto source = std::make_unique<TorchWeightSource>(std::move(bytes), std::move(archive),
                                                              std::string(options.origin));
            SA_TRY(source->collect(options.state_key));
            return Result<std::unique_ptr<WeightSource>>(std::move(source));
        }

    } // namespace IO
} // namespace SushiAI
