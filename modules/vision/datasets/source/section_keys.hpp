/**************************************************************************/
/* section_keys.hpp                                                       */
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
 * @file section_keys.hpp
 * @brief Declares the names of the keys every data section may hold, whatever its task.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>

namespace SushiAI::Vision::Datasets::Detail
{
    /** @brief Names the key that selects the row of the reader table. */
    inline constexpr std::string_view K_READER_KEY = "reader";

    /** @brief Names the key that selects the row of the task table. */
    inline constexpr std::string_view K_TASK_KEY = "task";

    /** @brief Names the key of the directory of the dataset. */
    inline constexpr std::string_view K_ROOT_KEY = "root";

    /** @brief Names the key of the split. */
    inline constexpr std::string_view K_SPLIT_KEY = "split";

    /** @brief Names the key of the samples per batch. */
    inline constexpr std::string_view K_BATCH_SIZE_KEY = "batch_size";

    /** @brief Names the key of the input height and width. */
    inline constexpr std::string_view K_INPUT_KEY = "input";

    /** @brief Names the key of the channels of the input. */
    inline constexpr std::string_view K_CHANNELS_KEY = "channels";

    /** @brief Names the key of the class count. */
    inline constexpr std::string_view K_CLASSES_KEY = "classes";

    /** @brief Names the key of how an image is fitted to the input. */
    inline constexpr std::string_view K_FIT_KEY = "fit";

    /** @brief Names the key of the directory of the raw cache. */
    inline constexpr std::string_view K_CACHE_KEY = "cache";

    /** @brief Names the key of the longest side an image is cached at. */
    inline constexpr std::string_view K_CACHE_SIDE_KEY = "cache_side";

    /** @brief Names the key that gives every epoch its own order. */
    inline constexpr std::string_view K_SHUFFLE_KEY = "shuffle";

    /** @brief Names the key of the seed of the order and of the pipeline. */
    inline constexpr std::string_view K_SEED_KEY = "seed";

    /** @brief Names the key that selects the row of the pipeline table. */
    inline constexpr std::string_view K_PIPELINE_KEY = "pipeline";

    /** @brief Names the key of the count of threads that read batches ahead. */
    inline constexpr std::string_view K_WORKERS_KEY = "workers";

    /** @brief Names the key of the count of batches read ahead. */
    inline constexpr std::string_view K_PREFETCH_KEY = "prefetch";

    /** @brief Lists the keys a dataset of images reads and a dataset of rows refuses. */
    inline constexpr std::string_view K_IMAGE_KEYS[] = {
        K_FIT_KEY, K_CACHE_KEY, K_CACHE_SIDE_KEY, K_SHUFFLE_KEY, K_SEED_KEY, K_PIPELINE_KEY,
    };

    /** @brief Returns "<prefix> '<key>'", what a diagnostic calls one value of a section. */
    [[nodiscard]] inline std::string named(std::string_view prefix, std::string_view key)
    {
        return std::string(prefix) + " '" + std::string(key) + "'";
    }
} // namespace SushiAI::Vision::Datasets::Detail
