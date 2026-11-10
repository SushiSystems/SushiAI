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
 * @file datasets_test_support.hpp
 * @brief Declares the scratch directory, the IDX bytes and the images the dataset tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <support/scratch_names.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

#include <SushiAI/vision/datasets/byte_source.hpp>
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
                : path_(std::filesystem::temp_directory_path() / SushiAI::Test::unique_scratch_name(name))
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

    /** @brief Returns bytes as the text a MemorySource holds a file as. */
    inline std::string text_of(const std::vector<std::uint8_t>& bytes)
    {
        return std::string(bytes.begin(), bytes.end());
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

    /** @brief Returns the bytes of a raw image file that holds an image, as the text of a file. */
    inline std::string raw_image_text(const Imaging::Image& image)
    {
        std::ostringstream out(std::ios::binary);
        SA_EXPECT_OK(Imaging::write_raw_image(
            out, image.view(),
            image.channels() == 1 ? Imaging::ChannelOrder::Gray : Imaging::ChannelOrder::RGB));
        return out.str();
    }

    /** @brief Adds an image to a memory source as a raw image file, and returns its path. */
    inline std::filesystem::path add_raw_image(MemorySource& source, const char* relative,
                                               const Imaging::Image& image)
    {
        const std::filesystem::path file = relative;
        source.add_file(file, raw_image_text(image));
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
    /** @brief Holds the size of the gray images the collator tests collate. */
    inline constexpr Imaging::ImageSize K_BENCH_SIZE{1, 2};

    /**
     * @brief Provides a host engine, a two-file list and two gray samples to the tests of the
     *        collators; a test derives from it under its own Small_ suite name.
     *
     * The first sample is level 51 of "first.png" and the second is level 255 of "second.png".
     * The batches it allocates are host tensors, so no device and no process-wide engine is
     * reached.
     */
    class CollatorBench : public ::testing::Test
    {
        protected:
            /** @brief Names the two files and makes one gray sample of K_BENCH_SIZE per file. */
            void SetUp() override
            {
                list.files = {"first.png", "second.png"};

                samples.resize(2);
                samples[0].image = uniform_image(K_BENCH_SIZE, 1, 51);
                samples[0].file = 0;
                samples[1].image = uniform_image(K_BENCH_SIZE, 1, 255);
                samples[1].file = 1;
            }

            /** @brief Returns the samples as the span a collator takes. */
            SushiRuntime::span<const FittedSample> span() const
            {
                return SushiRuntime::span<const FittedSample>(samples.data(), samples.size());
            }

            /** @brief Allocates a batch of two samples of a shape over some roles. */
            Data::Batch allocate(const Shape& sample,
                                 const std::vector<Data::TargetDescription>& targets)
            {
                const Data::InputDescription input{sample, DType::Float32};
                Result<Data::Batch> batch =
                    Data::allocate_batch(backend.engine(), 2, input, targets);
                EXPECT_TRUE(batch.status().is_ok());
                return std::move(batch).value();
            }

            /** @brief Allocates a batch of two gray images of K_BENCH_SIZE over some roles. */
            Data::Batch allocate_images(const std::vector<Data::TargetDescription>& targets)
            {
                return allocate(Shape{K_BENCH_SIZE.height, K_BENCH_SIZE.width, 1}, targets);
            }

            /** @brief Holds the files the samples name. */
            SampleList list;

            /** @brief Holds the fitted samples of the batch. */
            std::vector<FittedSample> samples;

            /** @brief Holds the host engine the batches are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Writes one value into every element of every role of a batch. */
    inline void fill_targets(Data::Batch& batch, std::int32_t value)
    {
        for (Data::BatchTarget& target : batch.targets)
        {
            if (target.values.dtype == DType::Float32)
                std::fill_n(target.values.data_as<float>(), target.values.num_elements,
                            static_cast<float>(value));
            else
                std::fill_n(target.values.data_as<std::int32_t>(), target.values.num_elements,
                            value);
        }
    }
} // namespace SushiAI::Vision::Datasets::Support
