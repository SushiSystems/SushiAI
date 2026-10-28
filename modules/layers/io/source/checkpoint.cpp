/**************************************************************************/
/* checkpoint.cpp                                                         */
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
 * @file checkpoint.cpp
 * @brief Writes and loads checkpoints over the entries of a model state.
 * @author Mustafa Garip
 */

#include <SushiAI/io/checkpoint.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>
#include "model_records.hpp"

namespace SushiAI
{
    namespace IO
    {
        Status save_checkpoint(const std::filesystem::path& path, NN::ModelState& state) noexcept
        {
            std::ofstream out(path, std::ios::binary);
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                       path.string());

            SA_TRY_ASSIGN(CheckpointWriter writer,
                          CheckpointWriter::create(
                              out, static_cast<std::uint32_t>(state.entries().size())));
            SA_TRY(Detail::write_model_records(writer, state));
            writer.finish();

            SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing the checkpoint to '{}'",
                       path.string());
            return Status();
        }

        Status load_checkpoint(const std::filesystem::path& path, NN::ModelState& state) noexcept
        {
            return import_weights(path, state, {});
        }

    } // namespace IO
} // namespace SushiAI
