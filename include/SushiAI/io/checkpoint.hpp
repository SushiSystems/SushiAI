/**************************************************************************/
/* checkpoint.hpp                                                         */
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
 * @file checkpoint.hpp
 * @brief Declares checkpoint save and load over a model state, and the record of a tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace Detail
        {
            /**
             * @brief Describes @p tensor as a checkpoint record called @p name.
             * @param name The dotted parameter name.
             * @param tensor The tensor to describe.
             * @return The record.
             */
            inline TensorRecord record_for(const std::string& name,
                                           const SushiBLAS::Tensor& tensor)
            {
                TensorRecord record;
                record.name = name;
                record.dtype = tensor.dtype;
                record.shape = Shape(SushiRuntime::span<const int64_t>(
                    tensor.shape.data(), static_cast<std::size_t>(tensor.rank)));
                return record;
            }
        }

        /**
         * @brief Writes every entry of a model state to disk, parameters and buffers alike.
         * @param path Target filesystem path for the checkpoint.
         * @param state The tensors to write, in the order the state lists them.
         * @param engine Engine whose recorded tasks run before the tensor memory is read.
         * @return Success, or InvalidFile when the file cannot be written.
         */
        [[nodiscard]] Status save_checkpoint(const std::filesystem::path& path,
                                             NN::ModelState& state, SushiBLAS::Engine& engine);

        /**
         * @brief Loads a checkpoint into a model state, strictly in both directions.
         * @param path Checkpoint path to read from disk.
         * @param state The state every entry of which the file must supply exactly once.
         * @return Success, or the failure of import_weights() with no rename rule.
         */
        [[nodiscard]] Status load_checkpoint(const std::filesystem::path& path,
                                             NN::ModelState& state);

    } // namespace IO
} // namespace SushiAI
