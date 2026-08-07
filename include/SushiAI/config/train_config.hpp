/**************************************************************************/
/* train_config.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file train_config.hpp
 * @brief One training run, and one evaluation set, as documents.
 *
 * @section train What a train.json says
 *
 * @code
 * {
 *   "sushiai_train": 1,
 *   "model": "model.json",
 *   "objective": { "type": "cross_entropy" },
 *   "data":      { "type": "rings", "batch_size": 64, "batches": 64,
 *                  "classes": 4, "seed": 3735928559 },
 *   "optimizer": { "type": "adamw", "learning_rate": 3e-3, "weight_decay": 0.01 },
 *   "epochs": 50,
 *   "checkpoint": "heavy_mlp.ckpt"
 * }
 * @endcode
 *
 * **The architecture is referenced rather than embedded**, because inference
 * reads the same model.json and two copies of one architecture are two things
 * that drift apart silently.
 *
 * @section paths Relative paths resolve against the config file's directory
 *
 * Not the process working directory. So
 * @c "sushiai train examples/heavy_mlp/train.json" works from the repository
 * root, and a configuration directory can be moved without editing what is
 * inside it. A path is joined lexically rather than canonicalised, because
 * @c checkpoint names a file that does not exist yet.
 *
 * @section eval What an eval.json says
 *
 * @code
 * { "sushiai_data": 1,
 *   "data": { "type": "rings", "batch_size": 512, "batches": 1,
 *             "classes": 4, "seed": 3735928575 } }
 * @endcode
 *
 * Inference takes its data as a separate document because the architecture
 * file carries no data block - it describes a model - and an optional flag
 * that always errors when omitted would be a lie about optionality.
 *
 * Host lane: no SushiBLAS engine, no SYCL.
 */

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief The train-file version this build reads. */
        inline constexpr std::int64_t TRAIN_VERSION = 1;

        /** @brief The standalone data-file version this build reads. */
        inline constexpr std::int64_t DATA_VERSION = 1;

        /**
         * @struct DataSpec
         * @brief Where a run's batches come from.
         */
        struct DataSpec
        {
            /** @brief The @c Data::DATASET_REGISTRY key, e.g. "rings". */
            std::string type;

            /** @brief Every other key; the dataset's own factory validates them. */
            Value attributes;
        };

        /**
         * @struct TrainSpec
         * @brief One training run, end to end.
         */
        struct TrainSpec
        {
            /** @brief The architecture file, resolved against this file's directory. */
            std::filesystem::path model_path;

            /** @brief The @c Train::OBJECTIVE_REGISTRY key, e.g. "cross_entropy". */
            std::string objective_type;

            /** @brief The objective's own keys. */
            Value objective_attributes;

            /** @brief Where the batches come from. */
            DataSpec data;

            /** @brief The @c Optim::OPTIMIZER_REGISTRY key, e.g. "adamw". */
            std::string optimizer_type;

            /** @brief The optimizer's own keys. */
            Value optimizer_attributes;

            /** @brief How many passes over the dataset; at least one. */
            std::int64_t epochs = 1;

            /** @brief Where to write the checkpoint; empty means do not write one. */
            std::filesystem::path checkpoint;
        };

        /**
         * @brief Turns a parsed train document into a @ref TrainSpec.
         * @param root The parsed document.
         * @param origin What to call it in a diagnostic, usually a path.
         * @param base The directory relative paths resolve against.
         * @return The spec.
         * @throws Error If the version is absent or wrong, a key is unknown, a
         *         required key is absent, or @c epochs is not positive.
         */
        TrainSpec parse_train(const Value& root, std::string_view origin,
                              const std::filesystem::path& base);

        /**
         * @brief Reads, parses and validates the train file at @p path.
         *
         * Relative paths inside it resolve against @c path's own directory.
         *
         * @param path The file to read.
         * @return The spec.
         * @throws Error If the file cannot be read, is not valid JSON, or is
         *         not a valid train document.
         */
        TrainSpec load_train(const std::filesystem::path& path);

        /**
         * @brief Turns a parsed data document into a @ref DataSpec.
         * @param root The parsed document.
         * @param origin What to call it in a diagnostic, usually a path.
         * @return The spec.
         * @throws Error If the version is absent or wrong, or a key is unknown.
         */
        DataSpec parse_data(const Value& root, std::string_view origin);

        /**
         * @brief Reads, parses and validates the data file at @p path.
         * @param path The file to read.
         * @return The spec.
         * @throws Error If the file cannot be read, is not valid JSON, or is
         *         not a valid data document.
         */
        DataSpec load_data(const std::filesystem::path& path);

    } // namespace Config
} // namespace SushiAI
