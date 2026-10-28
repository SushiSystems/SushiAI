/**************************************************************************/
/* imported_model_writer.cpp                                              */
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
 * @file imported_model_writer.cpp
 * @brief Implements the writer of the three files of an imported model.
 * @author Mustafa Garip
 */

#include <SushiAI/io/imported_model_writer.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the key that states the version of a training file. */
            constexpr std::string_view K_VERSION_KEY = "sushiai_train";

            /** @brief Holds the key that names the model file. */
            constexpr std::string_view K_MODEL_KEY = "model";

            /** @brief Holds the key that names the weight file a run starts from. */
            constexpr std::string_view K_WEIGHTS_KEY = "weights";

            /** @brief Holds the key of the data block, which no foreign file states. */
            constexpr std::string_view K_DATA_KEY = "data";

            /** @brief Holds the key that names the checkpoint a run writes. */
            constexpr std::string_view K_CHECKPOINT_KEY = "checkpoint";

            /** @brief Lists the keys the writer states, which a recipe may not. */
            constexpr std::string_view K_WRITER_KEYS[] = {
                K_VERSION_KEY, K_MODEL_KEY, K_WEIGHTS_KEY, K_DATA_KEY, K_CHECKPOINT_KEY,
            };

            /** @brief Returns whether the writer states @p key itself. */
            [[nodiscard]] bool is_writer_key(std::string_view key) noexcept
            {
                for (const std::string_view own : K_WRITER_KEYS)
                {
                    if (own == key)
                        return true;
                }

                return false;
            }

            /**
             * @brief Checks that every tensor can be a record of one weight file.
             * @return InvalidDescription naming the tensor: an empty or an overlong name, a
             *         name stated twice, bytes that differ from the record, or too many.
             */
            [[nodiscard]] Status validate_tensors(SushiRuntime::span<const ImportedTensor> tensors)
            {
                SA_FAIL_IF(tensors.size() > K_MAX_TENSOR_COUNT, StatusCode::InvalidDescription,
                           "the imported model holds {} tensors, and a weight file holds at "
                           "most {}",
                           tensors.size(), K_MAX_TENSOR_COUNT);

                return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                              [&]() -> Status
                {
                    std::unordered_set<std::string_view> named;
                    for (const ImportedTensor& tensor : tensors)
                    {
                        const std::string& name = tensor.record.name;
                        SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                                   "an imported tensor has no name");
                        SA_FAIL_IF(name.size() > K_MAX_NAME_LENGTH,
                                   StatusCode::InvalidDescription,
                                   "the imported tensor '{}' has a name of {} bytes, and a "
                                   "weight file holds at most {}",
                                   name, name.size(), K_MAX_NAME_LENGTH);
                        SA_FAIL_IF(!named.insert(name).second, StatusCode::InvalidDescription,
                                   "the imported model states the tensor '{}' twice", name);
                        SA_FAIL_IF(tensor.payload.size() != tensor.record.payload_bytes(),
                                   StatusCode::InvalidDescription,
                                   "the imported tensor '{}' holds {} bytes, but its shape {} "
                                   "needs {}",
                                   name, tensor.payload.size(), tensor.record.shape.to_string(),
                                   tensor.record.payload_bytes());
                    }

                    return Status();
                });
            }

            /** @brief Returns InvalidDescription unless the recipe is null or an object. */
            [[nodiscard]] Status validate_recipe(const Config::Value& recipe)
            {
                SA_FAIL_IF(!recipe.is_null() && recipe.kind() != Config::Kind::Object,
                           StatusCode::InvalidDescription,
                           "the training recipe of an imported model must be an object, but it "
                           "is {}",
                           Config::kind_name(recipe.kind()));

                for (const std::string& key : recipe.keys())
                {
                    SA_FAIL_IF(is_writer_key(key), StatusCode::InvalidDescription,
                               "the training recipe of an imported model states '{}', which "
                               "the writer of the training file states",
                               key);
                }

                return Status();
            }

            /** @brief Returns Incompatible naming the first of @p paths that exists. */
            [[nodiscard]] Status refuse_existing(
                SushiRuntime::span<const std::filesystem::path> paths)
            {
                for (const std::filesystem::path& path : paths)
                {
                    std::error_code ignored;
                    SA_FAIL_IF(std::filesystem::exists(path, ignored), StatusCode::Incompatible,
                               "'{}' exists already, and the import was not told to replace it",
                               path.string());
                }

                return Status();
            }

            /** @brief Writes @p text to @p path, or returns InvalidFile naming the path. */
            [[nodiscard]] Status write_text(const std::filesystem::path& path,
                                            std::string_view text)
            {
                std::ofstream out(path, std::ios::binary);
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                           path.string());

                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                out.flush();
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing '{}'", path.string());
                return Status();
            }

            /**
             * @brief Writes @p tensors to @p path as one weight file.
             * @pre validate_tensors succeeds on @p tensors.
             * @return InvalidFile naming the path when the file cannot be written.
             */
            [[nodiscard]] Status write_weights(const std::filesystem::path& path,
                                               SushiRuntime::span<const ImportedTensor> tensors)
            {
                std::ofstream out(path, std::ios::binary);
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                           path.string());

                SA_TRY_ASSIGN(CheckpointWriter writer,
                              CheckpointWriter::create(
                                  out, static_cast<std::uint32_t>(tensors.size())));
                for (const ImportedTensor& tensor : tensors)
                {
                    SA_TRY(writer.write(tensor.record, tensor.payload.data()));
                }
                writer.finish();

                out.flush();
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing '{}'", path.string());
                return Status();
            }

        } // namespace

        Result<Config::Value> compose_training_file(const ImportedModel& imported) noexcept
        {
            SA_TRY(validate_recipe(imported.training));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<std::string> keys;
                std::vector<Config::Value> values;
                const auto state = [&](std::string_view key, Config::Value value)
                {
                    keys.emplace_back(key);
                    values.push_back(std::move(value));
                };

                state(K_VERSION_KEY, Config::Value::integer(Config::K_TRAIN_VERSION));
                state(K_MODEL_KEY, Config::Value::string(std::string(K_IMPORTED_MODEL_FILE)));
                if (!imported.weights.empty())
                    state(K_WEIGHTS_KEY,
                          Config::Value::string(std::string(K_IMPORTED_WEIGHTS_FILE)));

                // No foreign file names a dataset, so the block is null until a person fills it.
                state(K_DATA_KEY, Config::Value());
                state(K_CHECKPOINT_KEY,
                      Config::Value::string(std::string(K_IMPORTED_CHECKPOINT_FILE)));

                const Config::Value absent;
                for (const std::string& key : imported.training.keys())
                    state(key, imported.training.at_or(key, absent));

                return Config::Value::object(std::move(keys), std::move(values));
            });
        }

        Result<WrittenImport> write_imported_model(const ImportedModel& imported,
                                                   const std::filesystem::path& directory,
                                                   ExistingFiles existing) noexcept
        {
            SA_TRY_ASSIGN(const std::string model_text, Config::write_model(imported.model));
            SA_TRY_ASSIGN(const Config::Value training, compose_training_file(imported));
            SA_TRY_ASSIGN(const std::string training_text, Config::write_json(training));
            SA_TRY(validate_tensors(imported.weights));

            const bool weighted = !imported.weights.empty();
            SA_TRY_ASSIGN(WrittenImport written,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                WrittenImport planned;
                planned.files.push_back(directory / std::filesystem::path(K_IMPORTED_MODEL_FILE));
                if (weighted)
                    planned.files.push_back(
                        directory / std::filesystem::path(K_IMPORTED_WEIGHTS_FILE));
                planned.files.push_back(
                    directory / std::filesystem::path(K_IMPORTED_TRAINING_FILE));
                return planned;
            }));

            if (existing == ExistingFiles::Refuse)
            {
                SA_TRY(refuse_existing(written.files));
            }

            std::error_code failure;
            std::filesystem::create_directories(directory, failure);
            SA_FAIL_IF(failure, StatusCode::InvalidFile, "could not create '{}': {}",
                       directory.string(), failure.message());

            SA_TRY(write_text(written.files.front(), model_text));
            if (weighted)
            {
                SA_TRY(write_weights(written.files[1], imported.weights));
            }
            SA_TRY(write_text(written.files.back(), training_text));

            return written;
        }

    } // namespace IO
} // namespace SushiAI
