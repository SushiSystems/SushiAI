/**************************************************************************/
/* datasets_test_support.hpp                                              */
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
 * @file datasets_test_support.hpp
 * @brief Declares the scratch directory and the token files the language dataset tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

#include <SushiAI/language/text/token.hpp>

/** @brief Holds what the tests of the language datasets module share. */
namespace SushiAI::Language::Datasets::Support
{
    /** @brief Owns a directory under the system's temporary directory for one test. */
    class ScratchDirectory final
    {
        public:
            /** @brief Creates the directory empty. */
            explicit ScratchDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path_);
                std::filesystem::create_directories(path_);
            }

            /** @brief Removes the directory and what it holds. */
            ~ScratchDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            /** @brief Forbids a copy, which would remove the directory twice. */
            ScratchDirectory(const ScratchDirectory&) = delete;

            /** @brief Forbids copy assignment. */
            ScratchDirectory& operator=(const ScratchDirectory&) = delete;

            /** @brief Forbids a move, which would leave two owners of one path. */
            ScratchDirectory(ScratchDirectory&&) = delete;

            /** @brief Forbids move assignment. */
            ScratchDirectory& operator=(ScratchDirectory&&) = delete;

            /** @brief Returns the directory. */
            [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

            /** @brief Writes bytes to a file of the directory and returns its path. */
            std::filesystem::path write(const char* name, std::string_view bytes) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                return file;
            }

        private:
            /// @brief Holds the directory this object created and removes.
            std::filesystem::path path_;
    };

    /** @brief Returns every byte of a file. */
    inline std::string read_bytes(const std::filesystem::path& file)
    {
        std::ifstream in(file, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    /** @brief Writes one byte over the byte at an offset of a file. */
    inline void overwrite_byte(const std::filesystem::path& file, std::size_t offset, char byte)
    {
        std::fstream stream(file, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
        stream.put(byte);
    }

    /** @brief Returns the ids 0, 1, ... count - 1, so that an id is its own position. */
    inline std::vector<Text::TokenID> rising_ids(std::size_t count)
    {
        std::vector<Text::TokenID> ids(count);
        for (std::size_t i = 0; i < count; ++i)
            ids[i] = static_cast<Text::TokenID>(i);

        return ids;
    }

    /**
     * @brief Returns whether @p part holds, byte for byte, rows @p rows of @p whole.
     * @param batch_rows The rows of @p whole.
     */
    inline ::testing::AssertionResult holds_rows_of(const Data::Batch& whole,
                                                    std::int64_t batch_rows,
                                                    Data::RowRange rows,
                                                    const Data::Batch& part)
    {
        if (SushiAI::Test::read_tensor_bytes(part.inputs)
            != SushiAI::Test::read_row_bytes(whole.inputs, batch_rows, rows.first,
                                             rows.count))
            return ::testing::AssertionFailure() << "the inputs differ";

        if (part.targets.size() != whole.targets.size())
            return ::testing::AssertionFailure()
                   << part.targets.size() << " roles against " << whole.targets.size();

        for (std::size_t t = 0; t < whole.targets.size(); ++t)
        {
            if (part.targets[t].role != whole.targets[t].role
                || SushiAI::Test::read_tensor_bytes(part.targets[t].values)
                       != SushiAI::Test::read_row_bytes(whole.targets[t].values, batch_rows,
                                                        rows.first, rows.count))
                return ::testing::AssertionFailure()
                       << "the role '" << whole.targets[t].role << "' differs";
        }

        return ::testing::AssertionSuccess();
    }
} // namespace SushiAI::Language::Datasets::Support
