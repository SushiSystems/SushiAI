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
     * @pre The tasks recorded on the state's engine have run.
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
