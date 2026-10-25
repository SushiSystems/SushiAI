/**************************************************************************/
/* free_training_loop.cpp                                                 */
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
 * @file free_training_loop.cpp
 * @brief Implements the loop that drives an anchor-free training step over a dataset.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_training_loop.hpp>

#include <vector>

#include <SushiAI/data/epoch.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/detection_run_check.hpp>
#include <SushiAI/vision/training/run_progress.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Checks that @p dataset can feed @p step for the run @p options describes. */
        [[nodiscard]] Status check_run(const FreeTrainingStep& step, const Data::Dataset& dataset,
                                       const FreeLoopOptions& options)
        {
            const FreeObjective& objective = step.objective();

            DetectionRunSubject subject;
            subject.run_name = "An anchor-free run";
            subject.batch_origin = "ports were traced for";
            subject.batch = objective.batch();
            subject.classes = objective.facts().classes;

            return check_detection_run(dataset, subject, make_detection_run_plan(options));
        }
    } // namespace

    Result<FreeRunReport> run_free_training(FreeTrainingStep& step, const Data::Dataset& dataset,
                                            const FreeLoopOptions& options)
    {
        SA_TRY(check_run(step, dataset, options));

        Optim::Optimizer& optimizer = step.optimizer();
        FreeRunReport report;

        // One batch and one truth for the run, so both keep their storage between batches.
        Data::Batch batch;
        Detect::GroundTruthBatch truth;

        for (std::int64_t epoch = 1; epoch <= options.epochs; ++epoch)
        {
            Data::begin_epoch(dataset, options.first_epoch + epoch - 1);

            const std::size_t first = epoch == 1 ? options.first_batch : 0;
            for (std::size_t index = first; index < dataset.batch_count(); ++index)
            {
                SA_TRY(dataset.read(index, batch));
                SA_TRY(truth_from_roles(batch, truth));

                FreeProgress progress;
                progress.epoch = epoch;
                progress.batch = index;
                progress.rate = options.schedule != nullptr
                              ? Optim::apply_step_schedule(*options.schedule, optimizer)
                              : optimizer.learning_rate();

                SA_TRY_ASSIGN(progress.terms, step.run(batch.inputs, truth));
                progress.iteration = static_cast<std::int64_t>(optimizer.step_count());

                if (report.updates == 0)
                    report.first = progress.terms;
                report.last = progress.terms;
                report.last_rate = progress.rate;
                ++report.updates;

                SA_TRY(report_run_progress(options.on_update, progress));
            }
        }

        return report;
    }

} // namespace SushiAI::Vision::Training
