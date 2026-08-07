/**************************************************************************/
/* train_config.cpp                                                       */
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
            constexpr std::string_view TRAIN_KEYS[] = {
                "sushiai_train", "model", "objective", "data", "optimizer",
                "epochs", "checkpoint",
            };

            /** @brief Every key a standalone data document may carry. */
            constexpr std::string_view DATA_KEYS[] = {"sushiai_data", "data"};

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief Joins @p text onto @p base unless it is already absolute.
             *
             * Lexical rather than canonical, because a checkpoint names a file
             * that does not exist yet and weakly_canonical would be resolving
             * a path against a directory that may not exist either.
             *
             * @param base The config file's own directory.
             * @param text The path as the file spelled it.
             * @return The resolved path.
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

                SA_THROW_IF(block.kind() != Kind::OBJECT,
                            "{} must be an object, but it is {}",
                            context, kind_name(block.kind()));

                type = block.at("type", context).as_string(context + " 'type'");

                // Everything except the discriminator, so the factory this
                // selects never has to allow the key that selected it.
                attributes = block.without("type");
            }
        }

        TrainSpec parse_train(const Value& root, std::string_view origin,
                              const std::filesystem::path& base)
        {
            require_version(root, "sushiai_train", TRAIN_VERSION, origin);
            root.reject_unknown_keys(origin, as_span(TRAIN_KEYS));

            TrainSpec spec;
            const std::string origin_text(origin);

            spec.model_path = resolve(
                base, root.at("model", origin).as_string("'" + origin_text + "'s 'model'"));

            read_typed_block(root, "objective", origin,
                             spec.objective_type, spec.objective_attributes);
            read_typed_block(root, "optimizer", origin,
                             spec.optimizer_type, spec.optimizer_attributes);
            read_typed_block(root, "data", origin, spec.data.type, spec.data.attributes);

            const Value one = Value::integer(1);
            spec.epochs = root.at_or("epochs", one).as_int("'" + origin_text + "'s 'epochs'");
            SA_THROW_IF(spec.epochs <= 0,
                        "'{}'s 'epochs' must be positive, but it is {}", origin, spec.epochs);

            // Optional: a run that writes no checkpoint is legitimate - a smoke
            // test, or a sweep whose only output is the printed loss.
            if (root.has("checkpoint"))
                spec.checkpoint = resolve(
                    base, root.at("checkpoint", origin)
                              .as_string("'" + origin_text + "'s 'checkpoint'"));

            return spec;
        }

        TrainSpec load_train(const std::filesystem::path& path)
        {
            // The config file's own directory, not the process working
            // directory - so `sushiai train examples/heavy_mlp/train.json`
            // works from the repository root and a configuration directory can
            // be moved without editing what is inside it.
            std::filesystem::path base = path.parent_path();
            if (base.empty()) base = ".";

            return parse_train(parse_json_file(path), path.string(), base);
        }

        DataSpec parse_data(const Value& root, std::string_view origin)
        {
            require_version(root, "sushiai_data", DATA_VERSION, origin);
            root.reject_unknown_keys(origin, as_span(DATA_KEYS));

            DataSpec data;
            read_typed_block(root, "data", origin, data.type, data.attributes);
            return data;
        }

        DataSpec load_data(const std::filesystem::path& path)
        {
            return parse_data(parse_json_file(path), path.string());
        }

    } // namespace Config
} // namespace SushiAI
