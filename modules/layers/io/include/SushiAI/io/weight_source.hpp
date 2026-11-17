/**************************************************************************/
/* weight_source.hpp                                                      */
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
 * @file weight_source.hpp
 * @brief Declares the seam between a weight file of any container and the strict import.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names one tensor of a weight file: its key, its extents and how it is stored. */
        struct SourceTensor
        {
            /** @brief Holds the key the file states the tensor under. */
            std::string name;

            /** @brief Holds its extents. */
            Shape shape{};

            /** @brief Holds the element type as the file spells it, for a message. */
            std::string stored_type;
        };

        /**
         * @class WeightSource
         * @brief Lists the tensors of one weight file and reads any of them as float32.
         *
         * One implementation per container: it knows the bytes and no model. Every
         * implementation owns the bytes it reads, so a source outlives the buffer it was built
         * from.
         */
        class WeightSource
        {
            public:
                /** @brief Releases the bytes of the file. */
                virtual ~WeightSource() = default;

                /** @brief Returns the name of the file, as a message quotes it. */
                [[nodiscard]] virtual std::string_view origin() const noexcept = 0;

                /** @brief Returns the tensors in the order the file lists them. */
                [[nodiscard]] virtual SushiRuntime::span<const SourceTensor> tensors()
                    const noexcept = 0;

                /**
                 * @brief Returns the elements of tensors()[@p index] as float32, row-major.
                 * @pre @p index is below tensors().size().
                 * @return Unsupported naming the tensor for an element type that does not widen
                 *         to float32; InvalidFile for bytes the file does not hold.
                 */
                [[nodiscard]] virtual Result<std::vector<float>> read_float32(
                    std::size_t index) const noexcept = 0;
        };

        /** @brief Holds what a container needs besides its bytes. */
        struct WeightSourceOptions
        {
            /** @brief Names the file in a message, usually the path it came from. */
            std::string_view origin;

            /**
             * @brief Holds the key of the state dictionary inside the file, or nothing when the
             *        file is the dictionary. A container that has no nesting ignores it.
             */
            std::string_view state_key;
        };

        /** @brief Tells whether @p bytes are the container a row opens. */
        using WeightSourceMatcher = bool (*)(SushiRuntime::span<const std::byte> bytes) noexcept;

        /**
         * @brief Takes the bytes of a file of the container a row names and opens its source.
         * @param bytes The whole file; the source keeps them.
         * @return The source, or the failure of the container's reader.
         */
        using WeightSourceOpener = Result<std::unique_ptr<WeightSource>> (*)(
            std::vector<std::byte> bytes, const WeightSourceOptions& options) noexcept;

        /** @brief One row of a WeightSourceRegistry: a container, how to know it and open it. */
        struct WeightSourceEntry
        {
            /** @brief The container's name, such as "torch", for a message. */
            std::string_view format;

            /** @brief Tells from the bytes whether the file is of this container. */
            WeightSourceMatcher matches;

            /** @brief Opens a file the matcher accepted. */
            WeightSourceOpener open;
        };

        /** @brief Holds the containers a weight file may be in, one row each. */
        class WeightSourceRegistry final
        {
            public:
                /**
                 * @brief Appends a row.
                 * @param row Its format text must outlive the registry; an empty format or a null
                 *            function is fatal.
                 * @return InvalidDescription when a row holds the format.
                 */
                [[nodiscard]] Status add(const WeightSourceEntry& row) noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const WeightSourceEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<WeightSourceEntry> rows_;
        };

        /**
         * @brief Opens @p bytes through the first row whose matcher accepts them.
         * @param bytes The whole file; the source keeps them.
         * @return The source; InvalidFile naming @p options.origin and every format the registry
         *         holds when no row accepts the bytes.
         */
        [[nodiscard]] Result<std::unique_ptr<WeightSource>> open_weight_source(
            const WeightSourceRegistry& registry, std::vector<std::byte> bytes,
            const WeightSourceOptions& options) noexcept;

        /**
         * @brief Reads the file at @p path and opens it as open_weight_source does.
         * @param options Its origin is the path when it states none.
         * @return FileNotFound for an unreadable file, else what open_weight_source returns.
         */
        [[nodiscard]] Result<std::unique_ptr<WeightSource>> open_weight_file(
            const WeightSourceRegistry& registry, const std::filesystem::path& path,
            WeightSourceOptions options) noexcept;

    } // namespace IO
} // namespace SushiAI
