/**************************************************************************/
/* byte_source.hpp                                                        */
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
 * @file byte_source.hpp
 * @brief Declares the port the format readers read their files and folders through.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names what a path leads to. */
    enum class PathKind : std::uint8_t
    {
        Missing,   /**< The path names nothing. */
        File,      /**< The path names a regular file. */
        Directory, /**< The path names a directory. */
        Other,     /**< The path names something else, such as a device. */
    };

    /** @brief Names what ByteSource::list keeps of a directory. */
    enum class EntryKind : std::uint8_t
    {
        Directory, /**< Entries that are directories. */
        File,      /**< Entries that are regular files. */
    };

    /** @brief Names how far below a directory ByteSource::list looks. */
    enum class ListDepth : std::uint8_t
    {
        Children, /**< The entries of the directory itself. */
        Tree,     /**< The entries of the directory and of every directory below it. */
    };

    /**
     * @class ByteSource
     * @brief Gives a reader the bytes of the files and the names of the folders it opens.
     *
     * A reader that takes a ByteSource reads the same layout from the file system, through
     * the overload that takes no source, and from memory, through MemorySource. Every
     * message a call returns opens with the path and names the reader it is given.
     */
    class ByteSource
    {
        public:
            virtual ~ByteSource() = default;

            /** @brief Returns what a path leads to. */
            [[nodiscard]] virtual PathKind kind_of(const std::filesystem::path& path) const = 0;

            /**
             * @brief Returns the byte count of a regular file.
             * @param reader Names the calling reader in a message.
             * @return The size; FileNotFound when the path names nothing; InvalidFile when it
             *         names something else.
             */
            [[nodiscard]] virtual Result<std::uint64_t> size_of(
                const std::filesystem::path& file, std::string_view reader) const = 0;

            /**
             * @brief Returns every byte of a regular file.
             * @param reader Names the calling reader in a message.
             * @return The bytes; what size_of returns; InvalidFile when the read stops early.
             */
            [[nodiscard]] virtual Result<std::string> read(const std::filesystem::path& file,
                                                           std::string_view reader) const = 0;

            /**
             * @brief Returns the bytes of a regular file from an offset, as many as it holds.
             * @param offset The index of the first byte.
             * @param count The most bytes to return.
             * @param reader Names the calling reader in a message.
             * @return Fewer than @p count bytes, or none, when the file ends first; what size_of
             *         returns; InvalidFile when the file cannot be opened.
             */
            [[nodiscard]] virtual Result<std::string> read_range(
                const std::filesystem::path& file, std::uint64_t offset, std::uint64_t count,
                std::string_view reader) const = 0;

            /**
             * @brief Returns the entries of one kind in or below a directory, sorted by path.
             * @param reader Names the calling reader in a message.
             * @return The paths, each opening with @p directory; FileNotFound when it is not
             *         a directory; InvalidFile when it cannot be listed.
             */
            [[nodiscard]] virtual Result<std::vector<std::filesystem::path>> list(
                const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
                std::string_view reader) const = 0;
    };

    /**
     * @class MemorySource
     * @brief Serves files and folders that were added to it, with no file system behind them.
     *
     * A path is the one the reader builds, compared as std::filesystem::path compares it. A
     * folder exists when a file lies below it or add_directory named it.
     */
    class MemorySource final : public ByteSource
    {
        public:
            /** @brief Adds a file, or replaces the bytes of the one at @p path. */
            void add_file(const std::filesystem::path& path, std::string bytes);

            /** @brief Adds a folder that holds no file, or none yet. */
            void add_directory(const std::filesystem::path& path);

            [[nodiscard]] PathKind kind_of(const std::filesystem::path& path) const override;

            [[nodiscard]] Result<std::uint64_t> size_of(const std::filesystem::path& file,
                                                        std::string_view reader) const override;

            [[nodiscard]] Result<std::string> read(const std::filesystem::path& file,
                                                   std::string_view reader) const override;

            [[nodiscard]] Result<std::string> read_range(const std::filesystem::path& file,
                                                         std::uint64_t offset,
                                                         std::uint64_t count,
                                                         std::string_view reader) const override;

            [[nodiscard]] Result<std::vector<std::filesystem::path>> list(
                const std::filesystem::path& directory, EntryKind kind, ListDepth depth,
                std::string_view reader) const override;

        private:
            /** @brief Holds the bytes of every file, by path. */
            std::map<std::filesystem::path, std::string> files_;

            /** @brief Holds the folders that add_directory named. */
            std::set<std::filesystem::path> directories_;
    };
} // namespace SushiAI::Vision::Datasets
