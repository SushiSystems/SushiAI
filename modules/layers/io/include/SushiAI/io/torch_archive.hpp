/**************************************************************************/
/* torch_archive.hpp                                                      */
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
 * @file torch_archive.hpp
 * @brief Declares the reader of a PyTorch zip file: its pickle tree and the tensors it names.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/ULTRALYTICS.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/zip_archive.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names one tensor of a PyTorch file: where its elements lie and how. */
        struct TorchTensor
        {
            /** @brief Holds the element type of its storage. */
            DType dtype = DType::Float32;

            /** @brief Holds its extents. */
            Shape shape{};

            /** @brief Holds the step of each axis in elements, outermost first. */
            std::vector<std::int64_t> strides;

            /** @brief Holds the element of the storage its first element is. */
            std::int64_t storage_offset = 0;

            /** @brief Holds the archive entry of its storage. */
            const ZipEntry* storage = nullptr;
        };

        /**
         * @class TorchArchive
         * @brief Holds a PyTorch zip file read: its archive, its pickle tree and its tensors.
         *
         * The archive holds a view of the bytes it was read from, which must outlive it.
         */
        class TorchArchive final
        {
            public:
                /**
                 * @brief Reads the archive and the pickle of @p bytes.
                 * @param accepted Globals the pickle may name beside PyTorch's own rebuilders,
                 *                 in the form read_pickle takes.
                 * @return What ZipArchive::read and read_pickle return; InvalidFile for an
                 *         archive with no data.pkl or a byte order other than little-endian.
                 */
                [[nodiscard]] static Result<TorchArchive> read(
                    SushiRuntime::span<const std::byte> bytes,
                    SushiRuntime::span<const std::string_view> accepted,
                    std::string_view origin) noexcept;

                /** @brief Returns the root of the pickle tree. */
                [[nodiscard]] const PickleValue& root() const noexcept;

                /** @brief Returns whether @p value is a tensor or a parameter the pickle built. */
                [[nodiscard]] static bool holds_tensor(const PickleValue& value) noexcept;

                /**
                 * @brief Returns the tensor @p value builds.
                 * @pre holds_tensor(value).
                 * @return InvalidFile naming @p context when its arguments or its storage do not
                 *         fit; Unsupported for a storage type the reader lacks.
                 */
                [[nodiscard]] Result<TorchTensor> tensor(const PickleValue& value,
                                                         std::string_view context) const noexcept;

                /**
                 * @brief Returns the elements of @p tensor as float32, row-major and contiguous.
                 * @return Unsupported naming @p context for a type that is not half, float32 or
                 *         float64; InvalidFile for an element past the end of its storage.
                 */
                [[nodiscard]] Result<std::vector<float>> read_float32(
                    const TorchTensor& tensor, std::string_view context) const noexcept;

            private:
                /** @brief Holds the archive, the root of its pickle and the name of the file. */
                TorchArchive(ZipArchive archive, std::string prefix, PickleRef root,
                             std::string origin) noexcept;

                /** @brief Holds the entries of the file. */
                ZipArchive archive_;

                /** @brief Holds the folder every entry of the file sits in, with its '/'. */
                std::string prefix_;

                /** @brief Holds the root of the pickle tree. */
                PickleRef root_;

                /** @brief Names the file in a message. */
                std::string origin_;
        };
    } // namespace IO
} // namespace SushiAI
