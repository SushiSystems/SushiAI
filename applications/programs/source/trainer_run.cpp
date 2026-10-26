/**************************************************************************/
/* trainer_run.cpp                                                        */
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
 * @file trainer_run.cpp
 * @brief Implements the run that drives Train::Trainer over a dataset, one step a batch.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The metric belongs to the objective".
 */

#include "trainer_run.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/trainer.hpp>

#include "epoch_loop.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Prints the loss of an epoch and every metric the objective measured. */
            void print_epoch(std::int64_t epoch, std::int64_t epochs,
                             const Train::StepMetrics& last)
            {
                std::cout << SushiLog::format_text("  epoch {}/{} loss {}", SushiLog::right(epoch,
                                                                                            3),
                                                     SushiLog::left(epochs, 3),
                                                     SushiLog::fixed(last.loss, 6));
                for (const Train::Metric& metric : last.metrics.all())
                    std::cout << SushiLog::format_text(
                        "   {} {}", metric.name,
                        SushiLog::right(Train::describe_metric_value(metric), 7));
                std::cout << "\n";
            }

            /**
             * @brief Trains through Train::Trainer, setting the scheduled rate before each step.
             * @return The compile count of the step and where the run stands; Incompatible
             *         for a start the data lacks; or the first failure of the trainer, of a
             *         read or of a step.
             */
            [[nodiscard]] Result<RunSummary> run_with_trainer(PreparedRun& run)
            {
                Train::TrainerOptions options;
                options.epochs = 1;

                SA_TRY_ASSIGN(const std::unique_ptr<Train::Trainer> trainer,
                              Train::Trainer::make(run.engine, run.built, run.pool,
                                                   run.objective, run.optimizer, options,
                                                   std::move(run.updates)));

                SA_TRY(check_start_batch(run));
                SA_TRY(trainer->restore_run_count(
                    static_cast<std::size_t>(run.start.step_runs)));

                Data::Batch batch;
                Train::StepMetrics last;
                EpochSteps steps;
                steps.enter_epoch = [&run](std::int64_t epoch_index)
                {
                    Data::begin_epoch(run.dataset, epoch_index);
                };
                steps.run_step = [&](std::size_t index) -> Status
                {
                    {
                        SA_ZONE("data", "read");
                        SA_TRY(run.dataset.read(index, batch));
                    }
                    SA_TRY_ASSIGN(last, trainer->run_step(batch));
                    return Status();
                };
                steps.report = [&last](std::int64_t epoch, std::int64_t epochs)
                {
                    print_epoch(epoch, epochs, last);
                };
                SA_TRY(run_epochs(run, steps));

                RunSummary summary;
                summary.compile_count = trainer->compile_count();
                summary.progress = finished_progress(run, trainer->step().run_count());
                return summary;
            }

        } // namespace

        Status register_runs(RunRegistry& registry)
        {
            return registry.set_fallback(&run_with_trainer);
        }

    } // namespace Apps
} // namespace SushiAI
