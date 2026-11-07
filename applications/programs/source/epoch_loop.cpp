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
        Status run_epoch_loop(const EpochPlan& plan, const EpochSteps& steps)
        {
            for (std::int64_t epoch = 1; epoch <= plan.epochs; ++epoch)
            {
                if (steps.enter_epoch)
                    steps.enter_epoch(plan.first_epoch + epoch - 1);

                const std::size_t first = epoch == 1 ? plan.first_batch : 0;
                for (std::size_t index = first; index < plan.batches; ++index)
                {
                    Profile::mark_period(Profile::Timeline::TrainStep);
                    if (steps.set_rate)
                        steps.set_rate();

                    SA_TRY(steps.run_step(index));
                }

                if (is_reported_epoch(epoch, plan.epochs))
                    steps.report(epoch, plan.epochs);
            }

            return Status();
        }

        Status run_epochs(const PreparedRun& run, const EpochSteps& steps)
        {
            EpochPlan plan;
            plan.epochs = run.spec.epochs;
            plan.first_epoch = static_cast<std::int64_t>(run.start.epoch);
            plan.first_batch = static_cast<std::size_t>(run.start.batch);
            plan.batches = run.dataset.batch_count();

            EpochSteps scheduled = steps;
            if (run.schedule != nullptr)
            {
                scheduled.set_rate = [&run]()
                {
                    SA_ZONE("optim", "schedule");
                    Optim::apply_step_schedule(*run.schedule, run.optimizer);
                };
            }

            return run_epoch_loop(plan, scheduled);
        }

    } // namespace Apps
} // namespace SushiAI
