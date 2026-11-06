/**************************************************************************/
/* model_records.hpp                                                      */
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
 * @file model_records.hpp
 * @brief Declares the writer and the strict reader of the model-state records of a checkpoint.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI::IO::Detail
{
    /**
     * @brief Writes one record per entry of @p state, in the order the state lists them.
     * @pre Every graph that writes the state has run; the records wait for its queues.
     * @return Success, or the failure of CheckpointWriter::write.
     */
    [[nodiscard]] Status write_model_records(CheckpointWriter& writer, NN::ModelState& state);

    /**
     * @brief Reads every model-state record of @p reader into @p state, strictly both ways.
     * @param path Names the file in a message.
     * @param rules The rules applied to each record name before the lookup.
     * @return Incompatible for a record that matches no entry, repeats, or differs in dtype
     *         or shape, and for an entry left unsupplied; InvalidFile for a file cut short.
     */
    [[nodiscard]] Status read_model_records(CheckpointReader& reader,
                                            const std::filesystem::path& path,
                                            NN::ModelState& state,
                                            SushiRuntime::span<const RenameRule> rules);

} // namespace SushiAI::IO::Detail
