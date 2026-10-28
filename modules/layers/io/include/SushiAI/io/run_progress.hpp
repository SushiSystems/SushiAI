/**************************************************************************/
/* run_progress.hpp                                                       */
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
 * @file run_progress.hpp
 * @brief Declares where a training run stood when its checkpoint was written.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names the run-state record that holds the seed the run was started with. */
        inline constexpr std::string_view K_RUN_SEED_STATE = "run.seed";

        /** @brief Names the run-state record that holds the run count of the training step. */
        inline constexpr std::string_view K_RUN_STEP_RUNS_STATE = "run.step_runs";

        /** @brief Names the run-state record that holds the index of the epoch in progress. */
        inline constexpr std::string_view K_RUN_EPOCH_STATE = "run.epoch";

        /** @brief Names the run-state record that holds the next batch of that epoch. */
        inline constexpr std::string_view K_RUN_BATCH_STATE = "run.batch";

        /**
         * @struct RunProgress
         * @brief Holds the seed of a run and the place its next update starts from.
         */
        struct RunProgress
        {
            /** @brief Holds the seed the training graph of the run was traced with. */
            std::uint64_t seed = 0;

            /** @brief Holds the runs the compiled training step has finished. */
            std::uint64_t step_runs = 0;

            /** @brief Holds the index of the epoch the next batch is read in, from 0. */
            std::uint64_t epoch = 0;

            /** @brief Holds the index of the next batch of that epoch, from 0. */
            std::uint64_t batch = 0;
        };

        /**
         * @brief Refuses to go on from a checkpoint under another seed than it was written with.
         * @param saved What the checkpoint @p file holds.
         * @param seed The seed of the run that resumes.
         * @return Incompatible, naming @p file and both seeds, unless they are equal.
         */
        [[nodiscard]] Status check_run_seed(const RunProgress& saved, std::uint64_t seed,
                                            std::string_view file) noexcept;

    } // namespace IO
} // namespace SushiAI
