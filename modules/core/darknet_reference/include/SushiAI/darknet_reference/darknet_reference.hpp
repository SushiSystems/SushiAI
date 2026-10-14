/**************************************************************************/
/* darknet_reference.hpp                                                  */
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
 * @file darknet_reference.hpp
 * @brief Declares the readers of the files a Darknet run dumped.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds the environment variable that names the reference data directory. */
        inline constexpr const char* K_REFERENCE_DATA_VARIABLE = "SA_REFERENCE_DATA_DIR";

        /** @brief Holds where one file of the reference data is, or why a test skips. */
        struct DarknetFile
        {
            bool found = false;         /**< Whether the file exists. */
            std::string reason;         /**< Names the variable or the file that is missing. */
            std::filesystem::path path; /**< The file, when the variable is set. */
        };

        /** @brief Holds where the two files of one reference tensor are, or why a test skips. */
        struct DarknetReference
        {
            bool found = false;                /**< Whether both files exist. */
            std::string reason;                /**< Names the variable or the file missing. */
            std::filesystem::path description; /**< The `.json` file. */
            std::filesystem::path payload;     /**< The `.bin` file. */
        };

        /** @brief Holds one reference tensor as a test compares it. */
        struct DarknetTensor
        {
            std::vector<std::int64_t> extents; /**< `[H, W, C]` for a feature map, or as stored. */
            std::vector<float> values;         /**< The elements, row-major over extents. */
            Config::Value description;         /**< The whole object of the `.json` file. */
        };

        /**
         * @brief Locates a file directly in the reference data directory, such as a `.cfg`.
         * @param file_name The file's name, with its extension.
         * @return The path, or the reason for a skip.
         */
        [[nodiscard]] DarknetFile find_darknet_file(std::string_view file_name);

        /**
         * @brief Locates `references/<model>/<name>.json` and the `.bin` beside it.
         * @param model The directory of one run, such as "yolov3_tiny_thresh_0p01".
         * @param name The tensor's name, without an extension.
         * @return The two paths, or the reason for a skip.
         */
        [[nodiscard]] DarknetReference find_darknet_reference(std::string_view model,
                                                              std::string_view name);

        /**
         * @brief Reads one reference tensor and moves a `CHW` feature map to `HWC`.
         * @return The tensor; FileNotFound for a reference that was not found; InvalidFile for
         *         a `dtype` other than float32, a `byte_order` other than little, a shape that
         *         is not a list of whole numbers, or a `.bin` whose byte count differs from the
         *         shape's; Unsupported for an `order` other than `CHW` over three axes.
         */
        [[nodiscard]] Result<DarknetTensor> load_darknet_tensor(const DarknetReference& reference);

    } // namespace Test
} // namespace SushiAI

/**
 * @brief Declares @p file as a located file of the reference data, or skips the test naming it.
 * @pre Used as a statement in a function returning void.
 */
#define SA_FIND_DARKNET_FILE_OR_SKIP(file, file_name)                                  \
    const ::SushiAI::Test::DarknetFile file =                                          \
        ::SushiAI::Test::find_darknet_file(file_name);                                 \
    if (!file.found)                                                                   \
        GTEST_SKIP() << file.reason

/**
 * @brief Declares @p reference as a located reference tensor, or skips the test naming the
 *        file that is missing.
 * @pre Used as a statement in a function returning void.
 */
#define SA_FIND_DARKNET_REFERENCE_OR_SKIP(reference, model, name)                      \
    const ::SushiAI::Test::DarknetReference reference =                                \
        ::SushiAI::Test::find_darknet_reference(model, name);                          \
    if (!reference.found)                                                              \
        GTEST_SKIP() << reference.reason
