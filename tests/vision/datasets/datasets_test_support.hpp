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
 * @brief Declares the scratch directory, the IDX bytes and the images the dataset tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

/** @brief Holds what the tests of the datasets module share. */
namespace SushiAI::Vision::Datasets::Support
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

            /** @brief Creates a directory below the scratch directory and returns its path. */
            std::filesystem::path make_directory(const char* relative) const
            {
                const std::filesystem::path directory = path_ / relative;
                std::filesystem::create_directories(directory);
                return directory;
            }

            /** @brief Writes bytes to a file, creating its folders, and returns its path. */
            std::filesystem::path write(const char* relative,
                                        const std::vector<std::uint8_t>& bytes) const
            {
                const std::filesystem::path file = path_ / relative;
                std::filesystem::create_directories(file.parent_path());

                std::ofstream out(file, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
                return file;
            }

            /** @brief Writes text to a file, creating its folders, and returns its path. */
            std::filesystem::path write_text(const char* relative, std::string_view text) const
            {
                return write(relative, std::vector<std::uint8_t>(text.begin(), text.end()));
            }

        private:
            /// @brief Holds the directory this object created and removes.
            std::filesystem::path path_;
    };

    /**
     * @brief Returns the bytes of an IDX file: two zero bytes, the code, the rank, the
     *        extents with the most significant byte first, then the payload as it is.
     */
    inline std::vector<std::uint8_t> idx_bytes(std::uint8_t code,
                                               const std::vector<std::uint32_t>& extents,
                                               const std::vector<std::uint8_t>& payload)
    {
        std::vector<std::uint8_t> bytes = {0, 0, code, static_cast<std::uint8_t>(extents.size())};

        for (std::uint32_t extent : extents)
        {
            bytes.push_back(static_cast<std::uint8_t>(extent >> 24));
            bytes.push_back(static_cast<std::uint8_t>(extent >> 16));
            bytes.push_back(static_cast<std::uint8_t>(extent >> 8));
            bytes.push_back(static_cast<std::uint8_t>(extent));
        }

        bytes.insert(bytes.end(), payload.begin(), payload.end());
        return bytes;
    }

    /** @brief Returns count bytes that rise by one from a first value, wrapping at 256. */
    inline std::vector<std::uint8_t> rising_bytes(std::size_t count, std::uint8_t first)
    {
        std::vector<std::uint8_t> bytes(count);
        for (std::size_t i = 0; i < count; ++i)
            bytes[i] = static_cast<std::uint8_t>(first + i);

        return bytes;
    }

    /** @brief Returns an image whose every byte is one value. */
    inline Imaging::Image uniform_image(Imaging::ImageSize size, std::int32_t channels,
                                        std::uint8_t value)
    {
        Imaging::Image image(size, channels);
        image.fill(value);
        return image;
    }

    /** @brief Returns an image of one channel that holds the given bytes, row by row. */
    inline Imaging::Image gray_image(Imaging::ImageSize size,
                                     const std::vector<std::uint8_t>& bytes)
    {
        Imaging::Image image(size, 1);
        for (std::size_t i = 0; i < image.byte_count() && i < bytes.size(); ++i)
            image.data()[i] = bytes[i];

        return image;
    }

    /**
     * @brief Writes an image as a raw image file, which vision/codecs decodes by its leading
     *        bytes whatever its extension, and returns its path.
     */
    inline std::filesystem::path write_raw_image(const ScratchDirectory& scratch,
                                                 const char* relative,
                                                 const Imaging::Image& image)
    {
        const std::filesystem::path file = scratch.path() / relative;
        std::filesystem::create_directories(file.parent_path());

        SA_EXPECT_OK(Imaging::write_raw_image_file(
            file, image.view(),
            image.channels() == 1 ? Imaging::ChannelOrder::Gray : Imaging::ChannelOrder::RGB));
        return file;
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
} // namespace SushiAI::Vision::Datasets::Support
