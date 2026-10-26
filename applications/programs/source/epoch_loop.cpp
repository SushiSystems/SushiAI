/**************************************************************************/
/* epoch_loop.cpp                                                         */
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
 * @file epoch_loop.cpp
 * @brief Implements run_epochs.
 * @author Mustafa Garip
 */

#include "epoch_loop.hpp"

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/profile/timelines.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Status run_epochs(const PreparedRun& run, const EpochSteps& steps)
        {
            const std::int64_t epochs = run.spec.epochs;
            const std::int64_t first_epoch = static_cast<std::int64_t>(run.start.epoch);
            const std::size_t batches = run.dataset.batch_count();
            for (std::int64_t epoch = 1; epoch <= epochs; ++epoch)
            {
                if (steps.enter_epoch)
                    steps.enter_epoch(first_epoch + epoch - 1);

                const std::size_t first =
                    epoch == 1 ? static_cast<std::size_t>(run.start.batch) : 0;
                for (std::size_t index = first; index < batches; ++index)
                {
                    Profile::mark_period(Profile::Timeline::TrainStep);
                    if (run.schedule != nullptr)
                    {
                        SA_ZONE("optim", "schedule");
                        Optim::apply_step_schedule(*run.schedule, run.optimizer);
                    }

                    SA_TRY(steps.run_step(index));
                }

                if (is_reported_epoch(epoch, epochs))
                    steps.report(epoch, epochs);
            }

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
