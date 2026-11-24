/**************************************************************************/
/* model_files.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file model_files.cpp
 * @brief Implements the reader of the three files of a SushiAI model into an ImportedModel.
 * @author Mustafa Garip
 */

#include <SushiAI/io/model_files.hpp>

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Reads the model-state records of the weight file at @p path. */
            [[nodiscard]] Result<std::vector<ImportedTensor>> read_weights(
                const std::filesystem::path& path)
            {
                std::ifstream in(path, std::ios::binary);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open the weight file '{}'",
                           path.string());

                SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
                std::vector<ImportedTensor> tensors;
                SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    tensors.reserve(reader.count());
                    return Status();
                }));
                while (reader.read_count() < reader.count())
                {
                    ImportedTensor tensor;
                    SA_TRY_ASSIGN(tensor.record, reader.next());
                    SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                    {
                        tensor.payload.resize(tensor.record.payload_bytes());
                        return Status();
                    }));
                    SA_TRY(SA_ANNOTATE(reader.read_payload(tensor.record, tensor.payload.data()),
                                       "'{}'", path.string()));
                    tensors.push_back(std::move(tensor));
                }

                return tensors;
            }

            /** @brief Splits the training file at @p path into its data block and its recipe. */
            [[nodiscard]] Status read_training(const std::filesystem::path& path,
                                               ImportedModel& model)
            {
                SA_TRY_ASSIGN(const Config::Value root, Config::parse_json_file(path));
                SA_FAIL_IF(root.kind() != Config::Kind::Object, StatusCode::InvalidFile,
                           "the training file '{}' is no object", path.string());

                const Config::Value absent;
                model.data = root.at_or("data", absent);
                Config::Value recipe = root;
                for (const std::string_view key : K_TRAINING_WRITER_KEYS)
                    recipe = recipe.without(key);
                if (!recipe.keys().empty())
                    model.training = std::move(recipe);
                return Status();
            }

        } // namespace

        Result<ImportedModel> read_model_files(const ModelFilePaths& paths) noexcept
        {
            ImportedModel model;
            SA_TRY_ASSIGN(model.model, Config::load_model(paths.model));
            if (!paths.weights.empty())
            {
                SA_TRY_ASSIGN(model.weights, read_weights(paths.weights));
            }
            if (!paths.training.empty())
            {
                SA_TRY(read_training(paths.training, model));
            }

            return model;
        }

    } // namespace IO
} // namespace SushiAI
