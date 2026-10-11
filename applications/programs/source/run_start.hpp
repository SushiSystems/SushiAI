/**************************************************************************/
/* run_start.hpp                                                          */
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
 * @file run_start.hpp
 * @brief Declares where the model state and the run state of `sushiai train` start from.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Returns what the header prints for where the model state comes from. */
        [[nodiscard]] std::string describe_start(const Config::TrainSpec& spec);

        /**
         * @brief Gives the model state, and on a resume the run state, its starting values.
         * @param spec "resume" loads both from one checkpoint, "weights" the model state
         *             alone, and neither runs the initialisers of the model.
         * @param start Takes the seed of @p spec and, on a resume, where the saved run stood.
         * @return Incompatible for a checkpoint of another seed, or the failure of a load.
         * @pre Config::check_start(spec) has returned success.
         */
        [[nodiscard]] Status start_state(const Config::TrainSpec& spec, SushiBLAS::Engine& engine,
                                         NN::ModelState& state, Optim::Optimizer& optimizer,
                                         IO::RunProgress* start = nullptr);

    } // namespace Apps
} // namespace SushiAI
