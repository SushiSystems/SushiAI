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
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/trainer.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Holds how many epochs pass between two progress lines. */
            constexpr std::int64_t K_REPORT_INTERVAL = 5;

            /** @brief Prints the loss of an epoch and every metric the objective measured. */
            void print_epoch(std::int64_t epoch, std::int64_t epochs,
                             const Train::StepMetrics& last)
            {
                std::cout << SushiBLAS::Core::format("  epoch {:>3}/{:<3} loss {:.6f}", epoch,
                                                     epochs, last.loss);
                for (const Train::Metric& metric : last.metrics.all())
                    std::cout << SushiBLAS::Core::format(
                        "   {} {:>7}", metric.name, Train::describe_metric_value(metric));
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

                const std::int64_t epochs = run.spec.epochs;
                const std::int64_t first_epoch = static_cast<std::int64_t>(run.start.epoch);
                Data::Batch batch;
                for (std::int64_t epoch = 1; epoch <= epochs; ++epoch)
                {
                    Data::begin_epoch(run.dataset, first_epoch + epoch - 1);
                    Train::StepMetrics last;

                    const std::size_t first =
                        epoch == 1 ? static_cast<std::size_t>(run.start.batch) : 0;
                    for (std::size_t index = first; index < run.dataset.batch_count(); ++index)
                    {
                        SA_TRY(run.dataset.read(index, batch));

                        if (run.schedule != nullptr)
                            Optim::apply_step_schedule(*run.schedule, run.optimizer);

                        SA_TRY_ASSIGN(last, trainer->run_step(batch));
                    }

                    if (epoch == 1 || epoch % K_REPORT_INTERVAL == 0 || epoch == epochs)
                        print_epoch(epoch, epochs, last);
                }

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
