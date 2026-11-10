/**************************************************************************/
/* mnist.hpp                                                              */
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
 * @file mnist.hpp
 * @brief Declares the MNIST dataset loader and its configuration factory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>

namespace SushiAI::Vision::Datasets
{
    /**
     * @enum MNISTLayout
     * @brief How one MNIST sample is shaped in a batch.
     */
    enum class MNISTLayout : std::uint8_t
    {
        Flat,  /**< The pixels as one row: [784] for MNIST. */
        Image, /**< The pixels as one channel image: [28, 28, 1] for MNIST. */
    };

    /**
     * @struct MNISTOptions
     * @brief Where the IDX files are and how much of them to use.
     */
    struct MNISTOptions
    {
        /** @brief The directory holding the four standard IDX files. */
        std::filesystem::path directory = "data/mnist";

        /** @brief Samples per batch. */
        int64_t batch_size = 64;

        /** @brief Cap on how many samples to load; 0 means all of them. */
        int64_t max_samples = 4096;

        /** @brief Load the test split rather than the training split. */
        bool test_split = false;

        /** @brief The shape of one sample; Flat unless an image is wanted. */
        MNISTLayout layout = MNISTLayout::Flat;
    };

    /**
     * @brief Whether the IDX files for @p options are present and readable.
     * @param options Where to look and which split.
     * @return True when both the image and label files exist.
     */
    bool mnist_available(const MNISTOptions& options) noexcept;

    /**
     * @brief Whether the IDX files for @p options are present in @p source.
     * @param options Where to look and which split.
     * @param source What the paths are looked up in.
     * @return True when both the image and label files are regular files of @p source.
     */
    bool mnist_available(const MNISTOptions& options, const ByteSource& source) noexcept;

    /**
     * @brief The path pair @c mnist_available checks, for a diagnostic message.
     * @param options Where to look and which split.
     * @return "images: <path>, labels: <path>".
     */
    std::string mnist_expected_paths(const MNISTOptions& options) noexcept;

    /**
     * @brief Loads MNIST from IDX files.
     * @param engine Engine allocating device tensors.
     * @param options Directory, split, and sample count limits.
     * @return The dataset, never null; or FileNotFound when a file of the split is absent,
     *         InvalidFile when one is unreadable or malformed, InvalidDescription when the
     *         batch size is not positive, or the failure of an allocation.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> try_load_mnist(
        SushiBLAS::Engine& engine,
        const MNISTOptions& options = {}) noexcept;

    /**
     * @brief Loads MNIST from IDX files read through @p source.
     *
     * The overload above calls this with the file system.
     * @param source What the two IDX files are read from.
     * @return What the overload above returns.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> try_load_mnist(
        SushiBLAS::Engine& engine,
        const MNISTOptions& options,
        const ByteSource& source) noexcept;

    /**
     * @brief Builds the MNIST dataset from configuration attributes.
     * @param attributes Holds "directory" (required), "split" ("train" or "test"),
     *                   "layout" ("flat" or "image"), "batch_size" and "max_samples"; a
     *                   relative directory resolves against the working directory.
     * @return The dataset, never null; or InvalidDescription for a "split" or "layout"
     *         that names neither of its values, or the failure try_load_mnist returns.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_mnist(
        SushiBLAS::Engine& engine,
        const Config::Value& attributes,
        std::string_view context) noexcept;

    /**
     * @brief Builds the MNIST dataset from configuration attributes, reading through @p source.
     *
     * The overload above calls this with the file system.
     * @param source What the two IDX files are read from.
     * @return What the overload above returns.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_mnist(
        SushiBLAS::Engine& engine,
        const Config::Value& attributes,
        std::string_view context,
        const ByteSource& source) noexcept;

} // namespace SushiAI::Vision::Datasets
