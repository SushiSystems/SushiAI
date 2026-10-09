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
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        Status save_checkpoint(const std::filesystem::path& path, NN::ModelState& state,
                               SushiBLAS::Engine& engine)
        {
            SA_TRY(SushiAI::Detail::run_recorded(engine));

            std::ofstream out(path, std::ios::binary);
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                       path.string());

            const SushiRuntime::span<const NN::StateEntry> entries = state.entries();
            SA_TRY_ASSIGN(CheckpointWriter writer,
                          CheckpointWriter::create(
                              out, static_cast<std::uint32_t>(entries.size())));
            for (const NN::StateEntry& entry : entries)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(entry.state));
                SA_TRY(writer.write(Detail::record_for(entry.name, *tensor), tensor->data()));
            }
            writer.finish();

            SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing the checkpoint to '{}'",
                       path.string());
            return Status();
        }

        Status load_checkpoint(const std::filesystem::path& path, NN::ModelState& state)
        {
            return import_weights(path, state, {});
        }

    } // namespace IO
} // namespace SushiAI
