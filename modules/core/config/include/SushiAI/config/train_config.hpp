/**************************************************************************/
/* train_config.hpp                                                       */
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
 * @file train_config.hpp
 * @brief Declares training and evaluation configuration loaders and specifications.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief The train-file version this build writes; it reads this one and every older. */
        inline constexpr std::int64_t K_TRAIN_VERSION = 3;

        /** @brief The first train-file version that may carry the "cluster" key. */
        inline constexpr std::int64_t K_TRAIN_CLUSTER_VERSION = 2;

        /** @brief The first train-file version that may carry the "deterministic" key. */
        inline constexpr std::int64_t K_TRAIN_DETERMINISTIC_VERSION = 3;

        /** @brief The standalone data-file version this build reads. */
        inline constexpr std::int64_t K_DATA_VERSION = 1;

        /** @brief Names the key and the type of a data block that states a reader, not a type. */
        inline constexpr std::string_view K_READER_DATA_TYPE = "reader";

        /**
         * @struct DataSpec
         * @brief Where a run's batches come from.
         */
        struct DataSpec
        {
            /**
             * @brief The type a @c Data::DatasetRegistry row is found by, e.g. "rings".
             *
             * A block that spells "reader" and no "type" holds @ref K_READER_DATA_TYPE.
             */
            std::string type;

            /** @brief Every other key, and "reader" too; the dataset's factory validates them. */
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

            /** @brief The type a @c Train::ObjectiveRegistry row is found by, e.g. "anchor". */
            std::string objective_type;

            /** @brief The objective's own keys. */
            Value objective_attributes;

            /** @brief Where the batches come from. */
            DataSpec data;

            /** @brief The @c Optim::K_OPTIMIZER_REGISTRY key, e.g. "adamw". */
            std::string optimizer_type;

            /** @brief The optimizer's own keys. */
            Value optimizer_attributes;

            /**
             * @brief Holds the "schedule" object of the file, null when the file has none.
             *
             * This module reads none of it; @c Optim::read_rate_schedule does.
             */
            Value schedule;

            /** @brief How many passes over the dataset; at least one. */
            std::int64_t epochs = 1;

            /** @brief The "seed" as the file states it, 0 when absent; read_run_seed() checks it. */
            std::int64_t seed = 0;

            /** @brief The weight file the run starts from; empty means initialise the model. */
            std::filesystem::path weights;

            /** @brief The checkpoint whose model state and run state the run goes on from. */
            std::filesystem::path resume;

            /** @brief Where to write the checkpoint; empty means do not write one. */
            std::filesystem::path checkpoint;

            /**
             * @brief Holds the "cluster" object of the file, null when the file has none.
             *
             * This module reads none of it; @c Distribution::ClusterSection::read does.
             */
            Value cluster;

            /**
             * @brief The "deterministic" key; true gives the same weights from the same
             *        files, seed and device, bit for bit.
             *
             * While it is true, a key that asks for an unordered path is refused when the file
             * is read, naming the key. No key asks for one today.
             */
            bool deterministic = true;

            /** @brief The directory the file's relative paths resolve against. */
            std::filesystem::path base;
        };

        /**
         * @brief Parses a train document into a @ref TrainSpec.
         * @param root Parsed document value tree.
         * @param origin Context description for diagnostics.
         * @param base Directory relative paths resolve against.
         * @return The parsed specification; InvalidFile for a version this build does not read,
         *         InvalidDescription for a missing key or an invalid value.
         */
        [[nodiscard]] Result<TrainSpec> parse_train(const Value& root, std::string_view origin,
                                                    const std::filesystem::path& base);

        /**
         * @brief Reads, parses, and validates the train configuration file.
         * @param path Path to train configuration file.
         * @return The parsed specification; FileNotFound or InvalidFile when the file cannot be
         *         read or parsed, else what parse_train returns.
         */
        [[nodiscard]] Result<TrainSpec> load_train(const std::filesystem::path& path);

        /**
         * @brief Checks that a run states at most one file to start from.
         * @param origin What to call the document in a diagnostic, usually a path.
         * @return InvalidDescription when @p spec states both "weights" and "resume".
         */
        [[nodiscard]] Status check_start(const TrainSpec& spec, std::string_view origin);

        /**
         * @brief Returns the seed of the random sites of a run, as a training graph takes it.
         * @param origin What to call the document in a diagnostic, usually a path.
         * @return The seed, or InvalidDescription when @p spec states a negative one.
         */
        [[nodiscard]] Result<std::uint64_t> read_run_seed(const TrainSpec& spec,
                                                          std::string_view origin);

        /**
         * @brief Turns a parsed data document into a @ref DataSpec.
         * @param root The parsed document.
         * @param origin What to call it in a diagnostic, usually a path.
         * @return The spec; InvalidFile when the version is absent or wrong, InvalidDescription
         *         when a key is unknown.
         */
        [[nodiscard]] Result<DataSpec> parse_data(const Value& root, std::string_view origin);

        /**
         * @brief Reads, parses and validates the data file at @p path.
         * @param path The file to read.
         * @return The spec; FileNotFound or InvalidFile when the file cannot be read or is not
         *         valid JSON, else what parse_data returns.
         */
        [[nodiscard]] Result<DataSpec> load_data(const std::filesystem::path& path);

    } // namespace Config
} // namespace SushiAI
