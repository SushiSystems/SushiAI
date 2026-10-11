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

#include <utility>
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
             * @brief Reads a { "type": ..., ... } block into a type and its attributes.
             * @param root The document root.
             * @param key Which block, e.g. "optimizer".
             * @param origin What to call the document in a diagnostic.
             * @param type Set to the block's "type".
             * @param attributes Set to everything else the block carried.
             */
            void read_typed_block(const Value& root, std::string_view key,
                                  std::string_view origin,
                                  std::string& type, Value& attributes)
            {
                const std::string context = "'" + std::string(origin) + "'s '"
                                          + std::string(key) + "'";
                const Value& block = root.at(key, origin);

                SA_THROW_IF(block.kind() != Kind::Object,
                            "{} must be an object, but it is {}",
                            context, kind_name(block.kind()));

                type = block.at("type", context).as_string(context + " 'type'");

                // Stripping the type discriminator isolates component-specific attributes.
                attributes = block.without("type");
            }

            /**
             * @brief Reads the "data" block: a typed block, or one that states a reader.
             * @param root The document root.
             * @param origin What to call the document in a diagnostic.
             * @param data A block with "reader" and no "type" gets the type
             *             K_READER_DATA_TYPE and keeps every key; any other is a typed block.
             */
            void read_data_block(const Value& root, std::string_view origin, DataSpec& data)
            {
                const Value absent;
                const Value& block = root.at_or("data", absent);

                if (block.has(K_READER_DATA_TYPE) && !block.has("type"))
                {
                    data.type = std::string(K_READER_DATA_TYPE);
                    data.attributes = block;
                    return;
                }

                read_typed_block(root, "data", origin, data.type, data.attributes);
            }
        }

        TrainSpec parse_train(const Value& root, std::string_view origin,
                              const std::filesystem::path& base)
        {
            require_version(root, "sushiai_train", K_TRAIN_VERSION, origin);
            root.reject_unknown_keys(origin, as_span(K_TRAIN_KEYS));

            TrainSpec spec;
            const std::string origin_text(origin);

            spec.model_path = resolve(
                base, root.at("model", origin).as_string("'" + origin_text + "'s 'model'"));

            read_typed_block(root, "objective", origin,
                             spec.objective_type, spec.objective_attributes);
            read_typed_block(root, "optimizer", origin,
                             spec.optimizer_type, spec.optimizer_attributes);
            read_data_block(root, origin, spec.data);

            // The section stays opaque here; an absent key leaves the null value.
            spec.schedule = root.at_or("schedule", Value());

            const Value one = Value::integer(1);
            spec.epochs = root.at_or("epochs", one).as_int("'" + origin_text + "'s 'epochs'");
            SA_THROW_IF(spec.epochs <= 0,
                        "'{}'s 'epochs' must be positive, but it is {}", origin, spec.epochs);

            // The range of the seed is checked by read_run_seed(), which returns a status.
            const Value zero = Value::integer(0);
            spec.seed = root.at_or("seed", zero).as_int("'" + origin_text + "'s 'seed'");

            // Starting weights are optional; a run without them initialises the model.
            if (root.has("weights"))
                spec.weights = resolve(
                    base, root.at("weights", origin)
                              .as_string("'" + origin_text + "'s 'weights'"));

            // A resumed run takes its model state and its run state from one checkpoint.
            if (root.has("resume"))
                spec.resume = resolve(
                    base, root.at("resume", origin)
                              .as_string("'" + origin_text + "'s 'resume'"));

            // Checkpoint path is optional; training can run without writing snapshots.
            if (root.has("checkpoint"))
                spec.checkpoint = resolve(
                    base, root.at("checkpoint", origin)
                              .as_string("'" + origin_text + "'s 'checkpoint'"));

            return spec;
        }

        TrainSpec load_train(const std::filesystem::path& path)
        {
            // Relative paths resolve against the configuration file's parent directory.
            std::filesystem::path base = path.parent_path();
            if (base.empty()) base = ".";

            return parse_train(parse_json_file(path), path.string(), base);
        }

        Status check_start(const TrainSpec& spec, std::string_view origin)
        {
            SA_FAIL_IF(!spec.weights.empty() && !spec.resume.empty(),
                       StatusCode::InvalidDescription,
                       "'{}' states both 'weights' and 'resume'; 'weights' starts a run from "
                       "model state alone and 'resume' goes on from model state and run state, "
                       "so a file states one of them",
                       origin);
            return Status();
        }

        Result<std::uint64_t> read_run_seed(const TrainSpec& spec, std::string_view origin)
        {
            SA_FAIL_IF(spec.seed < 0, StatusCode::InvalidDescription,
                       "'{}'s 'seed' must not be negative, but it is {}", origin, spec.seed);
            return static_cast<std::uint64_t>(spec.seed);
        }

        DataSpec parse_data(const Value& root, std::string_view origin)
        {
            require_version(root, "sushiai_data", K_DATA_VERSION, origin);
            root.reject_unknown_keys(origin, as_span(K_DATA_KEYS));

            DataSpec data;
            read_data_block(root, origin, data);
            return data;
        }

        DataSpec load_data(const std::filesystem::path& path)
        {
            return parse_data(parse_json_file(path), path.string());
        }

    } // namespace Config
} // namespace SushiAI
