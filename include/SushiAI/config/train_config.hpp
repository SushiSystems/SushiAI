/****************************************************************/
/* train_config.hpp                                             */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
         * @brief Parses a train document into a @ref TrainSpec.
         * @param root Parsed document value tree.
         * @param origin Context description for diagnostics.
         * @param base Directory relative paths resolve against.
         * @return Parsed training specification.
         * @throws Error If required keys are missing or values invalid.
         */
        TrainSpec parse_train(const Value& root, std::string_view origin,
                              const std::filesystem::path& base);

        /**
         * @brief Reads, parses, and validates the train configuration file.
         * @param path Path to train configuration file.
         * @return Parsed training specification.
         * @throws Error If the file cannot be read or parsed.
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
