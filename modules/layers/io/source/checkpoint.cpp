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
 * @file checkpoint.cpp
 * @brief Writes and loads checkpoints over the entries of a model state.
 * @author Mustafa Garip
 */

#include <SushiAI/io/checkpoint.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>
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

        Status load_checkpoint(SushiRuntime::span<const std::uint8_t> bytes,
                               std::string_view origin, NN::ModelState& state) noexcept
        {
            return import_weights(bytes, origin, state, {});
        }

    } // namespace IO
} // namespace SushiAI
