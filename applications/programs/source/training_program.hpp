/**************************************************************************/
/* training_program.hpp                                                   */
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
 * @file training_program.hpp
 * @brief Declares the assembly `sushiai train` runs once it holds a parsed training file.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "training_program.hpp".
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/result.hpp>
#include "training_run.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @struct TrainingOutcome
         * @brief Holds what a finished training run reports to its caller.
         */
        struct TrainingOutcome
        {
            /** @brief What the row of the run table returned. */
            RunSummary summary;

            /** @brief The updates the optimizer has taken. */
            std::uint64_t optimizer_steps = 0;
        };

        /**
         * @brief Trains the model a parsed training file describes and saves its checkpoint.
         * @param origin What to call the file in a diagnostic and in the header.
         * @param spec The parsed file; its paths already resolve against its directory.
         * @param engine The engine of the run; a cluster file needs training_runtime_config().
         * @return The outcome, or the first failure of a step; InvalidDescription when the step
         *         compiled more than once.
         */
        [[nodiscard]] Result<TrainingOutcome> run_training_file(std::string_view origin,
                                                                const Config::TrainSpec& spec,
                                                                SushiBLAS::Engine& engine);
    } // namespace Apps
} // namespace SushiAI
