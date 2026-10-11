/**************************************************************************/
/* run_start.cpp                                                          */
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
 * @file run_start.cpp
 * @brief Implements the start of a run: initialised, from weights, or resumed.
 * @author Mustafa Garip
 */

#include "run_start.hpp"

#include <SushiAI/core/result.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/nn/init.hpp>

namespace SushiAI
{
    namespace Apps
    {
        std::string describe_start(const Config::TrainSpec& spec)
        {
            if (!spec.resume.empty())
                return "resumed from " + spec.resume.string();

            return spec.weights.empty() ? std::string("initialised") : spec.weights.string();
        }

        Status start_state(const Config::TrainSpec& spec, SushiBLAS::Engine& engine,
                           NN::ModelState& state, Optim::Optimizer& optimizer,
                           IO::RunProgress* start)
        {
            IO::RunProgress progress;
            SA_TRY_ASSIGN(progress.seed, Config::read_run_seed(spec, "the training file"));

            if (!spec.resume.empty())
            {
                IO::RunProgress saved;
                SA_TRY(IO::load_run_checkpoint(spec.resume, state, optimizer, &saved));
                SA_TRY(IO::check_run_seed(saved, progress.seed, spec.resume.string()));
                progress = saved;
            }
            else if (!spec.weights.empty())
            {
                // A weight file supplies every parameter and buffer, so no initialiser runs.
                SA_TRY(IO::load_checkpoint(spec.weights, state));
            }
            else
            {
                SA_TRY(NN::initialize(engine, state));
            }

            if (start != nullptr)
                *start = progress;

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
