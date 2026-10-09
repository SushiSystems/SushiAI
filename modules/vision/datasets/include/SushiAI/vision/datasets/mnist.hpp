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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI::Vision::Datasets
{
    /**
     * @enum MnistLayout
     * @brief How one MNIST sample is shaped in a batch.
     */
    enum class MnistLayout : std::uint8_t
    {
        Flat,  /**< The pixels as one row: [784] for MNIST. */
        Image, /**< The pixels as one channel image: [28, 28, 1] for MNIST. */
    };

    /**
     * @struct MnistOptions
     * @brief Where the IDX files are and how much of them to use.
     */
    struct MnistOptions
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
        MnistLayout layout = MnistLayout::Flat;
    };

    /**
     * @brief Whether the IDX files for @p options are present and readable.
     * @param options Where to look and which split.
     * @return True when both the image and label files exist.
     */
    bool mnist_available(const MnistOptions& options);

    /**
     * @brief The path pair @c mnist_available checks, for a diagnostic message.
     * @param options Where to look and which split.
     * @return "images: <path>, labels: <path>".
     */
    std::string mnist_expected_paths(const MnistOptions& options);

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
        const MnistOptions& options = {});

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
        std::string_view context);

} // namespace SushiAI::Vision::Datasets
