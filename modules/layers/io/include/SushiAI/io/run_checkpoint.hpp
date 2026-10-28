/**************************************************************************/
/* run_checkpoint.hpp                                                     */
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
 * @file run_checkpoint.hpp
 * @brief Declares checkpoint save and load over a model state and an optimizer's run state.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Writes a model state and, in the run-state section, what @p optimizer keeps.
         * @param optimizer Its type names the section; its scalars and tensors fill it.
         * @param progress Fills the four records that close the section; the default is a
         *                 run at its start under the seed 0.
         * @pre The tasks that wrote @p state and @p optimizer have run.
         * @return Success, or InvalidFile when the file cannot be written.
         */
        [[nodiscard]] Status save_run_checkpoint(const std::filesystem::path& path,
                                                 NN::ModelState& state,
                                                 Optim::Optimizer& optimizer,
                                                 const RunProgress& progress = RunProgress()) noexcept;

        /**
         * @brief Loads a model state and an optimizer's run state, both strictly both ways.
         * @param optimizer Every scalar and tensor of its run_state() must be supplied once.
         * @param progress Takes the four progress records, which the file must supply
         *                 whether or not this is null; a failed load leaves it as it was.
         * @return Incompatible for no run state, another optimizer type, or a record that is
         *         missing, surplus, repeated or of another type or shape; see README.md.
         */
        [[nodiscard]] Status load_run_checkpoint(const std::filesystem::path& path,
                                                 NN::ModelState& state,
                                                 Optim::Optimizer& optimizer,
                                                 RunProgress* progress = nullptr) noexcept;

    } // namespace IO
} // namespace SushiAI
