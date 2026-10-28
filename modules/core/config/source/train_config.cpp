/**************************************************************************/
/* train_config.cpp                                                       */
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
 * @file train_config.cpp
 * @brief Implements training and data configuration parsing.
 * @author Mustafa Garip
 */

#include <SushiAI/config/train_config.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /** @brief Every key a train document may carry at the top level. */
            constexpr std::string_view K_TRAIN_KEYS[] = {
                "sushiai_train", "model", "objective", "data", "optimizer",
                "schedule", "epochs", "seed", "weights", "resume", "checkpoint", "cluster",
                "deterministic",
            };

            /** @brief Every key a train document before K_TRAIN_DETERMINISTIC_VERSION may carry. */
            constexpr std::string_view K_TRAIN_KEYS_BEFORE_DETERMINISTIC[] = {
                "sushiai_train", "model", "objective", "data", "optimizer",
                "schedule", "epochs", "seed", "weights", "resume", "checkpoint", "cluster",
            };

            /** @brief Every key a train document before K_TRAIN_CLUSTER_VERSION may carry. */
            constexpr std::string_view K_TRAIN_KEYS_BEFORE_CLUSTER[] = {
                "sushiai_train", "model", "objective", "data", "optimizer",
                "schedule", "epochs", "seed", "weights", "resume", "checkpoint",
            };

            /** @brief Every key a standalone data document may carry. */
            constexpr std::string_view K_DATA_KEYS[] = {"sushiai_data", "data"};

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief Resolves @p text against @p base path lexically.
             * @param base Configuration directory path.
             * @param text Relative or absolute target path.
             * @return Resolved filesystem path.
             */
            std::filesystem::path resolve(const std::filesystem::path& base,
                                          const std::string& text)
            {
                const std::filesystem::path spelled(text);
                if (spelled.is_absolute()) return spelled;
                return base / spelled;
            }

            /**
             * @brief Reads the version of a train document: this build's or an older one.
             * @param root The document root.
             * @param origin What to call the document in a diagnostic.
             * @return The version, from 1 to K_TRAIN_VERSION; InvalidFile when the version is
             *         absent, no integer, or outside that range.
             */
            Result<std::int64_t> read_train_version(const Value& root, std::string_view origin)
            {
                // A version this build reads is checked as itself; any other as the current.
                std::int64_t expected = K_TRAIN_VERSION;
                if (root.kind() == Kind::Object && root.has("sushiai_train"))
                {
                    SA_TRY_ASSIGN(const Value& stated, root.at("sushiai_train", origin));
                    if (stated.kind() == Kind::Int)
                    {
                        SA_TRY_ASSIGN(const std::int64_t version, stated.as_int("'sushiai_train'"));
                        if (version >= 1 && version <= K_TRAIN_VERSION)
                            expected = version;
                    }
                }

                SA_TRY(require_version(root, "sushiai_train", expected, origin));
                return expected;
            }

            /**
             * @brief Reads a { "type": ..., ... } block into a type and its attributes.
             * @param root The document root.
             * @param key Which block, e.g. "optimizer".
             * @param origin What to call the document in a diagnostic.
             * @param type Set to the block's "type".
             * @param attributes Set to everything else the block carried.
             */
            Status read_typed_block(const Value& root, std::string_view key,
                                    std::string_view origin,
                                    std::string& type, Value& attributes)
            {
                const std::string context = "'" + std::string(origin) + "'s '"
                                          + std::string(key) + "'";
                SA_TRY_ASSIGN(const Value& block, root.at(key, origin));

                SA_FAIL_IF(block.kind() != Kind::Object, StatusCode::InvalidDescription,
                           "{} must be an object, but it is {}",
                           context, kind_name(block.kind()));

                SA_TRY_ASSIGN(const Value& stated_type, block.at("type", context));
                SA_TRY_ASSIGN(const std::string& stated, stated_type.as_string(context + " 'type'"));
                type = stated;

                // Stripping the type discriminator isolates component-specific attributes.
                attributes = block.without("type");
                return Status();
            }

            /** @brief Names the keys of a data block that hold the dataset's folder. */
            constexpr std::string_view K_DATA_FOLDER_KEYS[] = {"root", "directory"};

            /**
             * @brief Returns @p block with each relative string folder key resolved against @p base.
             * @param base The document's directory; empty leaves every folder as stated.
             * @return @p block itself when it is no object, holds no such string, or @p base is
             *         empty.
             */
            Value with_resolved_folders(const Value& block, const std::filesystem::path& base)
            {
                if (base.empty() || block.kind() != Kind::Object)
                    return block;
                std::vector<std::string> keys = block.keys();
                std::vector<Value> values = block.values();
                for (std::size_t index = 0; index < keys.size(); ++index)
                {
                    const bool folder = std::find(std::begin(K_DATA_FOLDER_KEYS),
                                                  std::end(K_DATA_FOLDER_KEYS),
                                                  keys[index]) != std::end(K_DATA_FOLDER_KEYS);
                    if (!folder || values[index].kind() != Kind::String)
                        continue;
                    Result<const std::string&> stated = values[index].as_string(keys[index]);
                    values[index] = Value::string(resolve(base, stated.value()).generic_string());
                }
                return Value::object(std::move(keys), std::move(values));
            }

            /**
             * @brief Reads the "data" block: a typed block, or one that states a reader.
             * @param root The document root.
             * @param origin What to call the document in a diagnostic.
             * @param base The document's directory, which a relative data folder resolves against.
             * @param data A block with "reader" and no "type" gets the type
             *             K_READER_DATA_TYPE and keeps every key; any other is a typed block.
             */
            Status read_data_block(const Value& root, std::string_view origin,
                                   const std::filesystem::path& base, DataSpec& data)
            {
                const Value absent;
                const Value& block = root.at_or("data", absent);

                if (block.has(K_READER_DATA_TYPE) && !block.has("type"))
                {
                    data.type = std::string(K_READER_DATA_TYPE);
                    data.attributes = with_resolved_folders(block, base);
                    return Status();
                }

                SA_TRY(read_typed_block(root, "data", origin, data.type, data.attributes));
                data.attributes = with_resolved_folders(data.attributes, base);
                return Status();
            }

            /**
             * @brief Reads the string under @p key of @p root as a path resolved against @p base.
             * @return The path, or InvalidDescription when the key is absent or no string.
             */
            Result<std::filesystem::path> read_path(const Value& root, std::string_view key,
                                                    std::string_view origin,
                                                    const std::filesystem::path& base)
            {
                SA_TRY_ASSIGN(const Value& stated, root.at(key, origin));
                SA_TRY_ASSIGN(const std::string& text,
                              stated.as_string("'" + std::string(origin) + "'s '"
                                               + std::string(key) + "'"));
                return resolve(base, text);
            }
        }

        Result<TrainSpec> parse_train(const Value& root, std::string_view origin,
                                      const std::filesystem::path& base) noexcept
        {
            // An older version refuses the keys it predates as unknown keys.
            SA_TRY_ASSIGN(const std::int64_t version, read_train_version(root, origin));
            if (version < K_TRAIN_CLUSTER_VERSION)
            {
                SA_TRY(root.reject_unknown_keys(origin, as_span(K_TRAIN_KEYS_BEFORE_CLUSTER)));
            }
            else if (version < K_TRAIN_DETERMINISTIC_VERSION)
            {
                SA_TRY(root.reject_unknown_keys(origin,
                                                as_span(K_TRAIN_KEYS_BEFORE_DETERMINISTIC)));
            }
            else
            {
                SA_TRY(root.reject_unknown_keys(origin, as_span(K_TRAIN_KEYS)));
            }

            TrainSpec spec;
            spec.base = base;
            const std::string origin_text(origin);

            SA_TRY_ASSIGN(spec.model_path, read_path(root, "model", origin, base));

            SA_TRY(read_typed_block(root, "objective", origin,
                                    spec.objective_type, spec.objective_attributes));
            SA_TRY(read_typed_block(root, "optimizer", origin,
                                    spec.optimizer_type, spec.optimizer_attributes));
            SA_TRY(read_data_block(root, origin, base, spec.data));

            // The section stays opaque here; an absent key leaves the null value.
            spec.schedule = root.at_or("schedule", Value());

            const Value one = Value::integer(1);
            SA_TRY_ASSIGN(spec.epochs,
                          root.at_or("epochs", one).as_int("'" + origin_text + "'s 'epochs'"));
            SA_FAIL_IF(spec.epochs <= 0, StatusCode::InvalidDescription,
                       "'{}'s 'epochs' must be positive, but it is {}", origin, spec.epochs);

            // The range of the seed is checked by read_run_seed(), which returns a status.
            const Value zero = Value::integer(0);
            SA_TRY_ASSIGN(spec.seed,
                          root.at_or("seed", zero).as_int("'" + origin_text + "'s 'seed'"));

            // Starting weights are optional; a run without them initialises the model.
            if (root.has("weights"))
            {
                SA_TRY_ASSIGN(spec.weights, read_path(root, "weights", origin, base));
            }

            // A resumed run takes its model state and its run state from one checkpoint.
            if (root.has("resume"))
            {
                SA_TRY_ASSIGN(spec.resume, read_path(root, "resume", origin, base));
            }

            // Checkpoint path is optional; training can run without writing snapshots.
            if (root.has("checkpoint"))
            {
                SA_TRY_ASSIGN(spec.checkpoint, read_path(root, "checkpoint", origin, base));
            }

            // The section stays opaque here, as "schedule" does; an absent key leaves null.
            spec.cluster = root.at_or("cluster", Value());

            // No key asks for an unordered path yet, so a true value refuses none.
            const Value ordered = Value::boolean(true);
            SA_TRY_ASSIGN(spec.deterministic,
                          root.at_or("deterministic", ordered)
                              .as_bool("'" + origin_text + "'s 'deterministic'"));

            return spec;
        }

        Result<TrainSpec> load_train(const std::filesystem::path& path) noexcept
        {
            // Relative paths resolve against the configuration file's parent directory.
            std::filesystem::path base = path.parent_path();
            if (base.empty()) base = ".";

            SA_TRY_ASSIGN(const Value root, parse_json_file(path));
            return parse_train(root, path.string(), base);
        }

        Status check_start(const TrainSpec& spec, std::string_view origin) noexcept
        {
            SA_FAIL_IF(!spec.weights.empty() && !spec.resume.empty(),
                       StatusCode::InvalidDescription,
                       "'{}' states both 'weights' and 'resume'; 'weights' starts a run from "
                       "model state alone and 'resume' goes on from model state and run state, "
                       "so a file states one of them",
                       origin);
            return Status();
        }

        Result<std::uint64_t> read_run_seed(const TrainSpec& spec, std::string_view origin) noexcept
        {
            SA_FAIL_IF(spec.seed < 0, StatusCode::InvalidDescription,
                       "'{}'s 'seed' must not be negative, but it is {}", origin, spec.seed);
            return static_cast<std::uint64_t>(spec.seed);
        }

        Result<DataSpec> parse_data(const Value& root, std::string_view origin,
                                    const std::filesystem::path& base) noexcept
        {
            SA_TRY(require_version(root, "sushiai_data", K_DATA_VERSION, origin));
            SA_TRY(root.reject_unknown_keys(origin, as_span(K_DATA_KEYS)));

            DataSpec data;
            SA_TRY(read_data_block(root, origin, base, data));
            return data;
        }

        Result<DataSpec> load_data(const std::filesystem::path& path) noexcept
        {
            // A relative "root" resolves against the data file's parent directory.
            std::filesystem::path base = path.parent_path();
            if (base.empty()) base = ".";

            SA_TRY_ASSIGN(const Value root, parse_json_file(path));
            return parse_data(root, path.string(), base);
        }

    } // namespace Config
} // namespace SushiAI
