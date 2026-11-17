/**************************************************************************/
/* anchor_training_run.cpp                                                */
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
 * @file anchor_training_run.cpp
 * @brief Implements the `anchor` run row, trained through AnchorTrainingStep.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#include "vision_training_runs.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/anchor_training_loop.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>

#include "epoch_loop.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Prints the cost and the rate of the last update of an epoch. */
            void print_epoch(const Vision::Training::AnchorProgress& progress,
                             std::int64_t epochs)
            {
                std::cout << SushiLog::format_text(
                    "  epoch {}/{} loss {}   box {}   rate {}\n",
                    SushiLog::right(progress.epoch, 3), SushiLog::left(epochs, 3),
                    SushiLog::fixed(progress.cost.total, 6), SushiLog::fixed(progress.cost.box, 6),
                    SushiLog::general(progress.rate, 6));
            }

            /**
             * @brief Starts the objectness and class biases of every anchor head of @p run from
             *        the stride of its level, as Ultralytics' YOLOv5 does.
             * @return The failure of the model or task read, of the head lookup or of the write.
             */
            [[nodiscard]] Status start_anchor_biases(PreparedRun& run)
            {
                SA_TRY_ASSIGN(const Config::ModelSpec model,
                              Config::load_model(run.spec.model_path));
                SA_TRY_ASSIGN(const Vision::Detect::Anchor::AnchorTask task,
                              Vision::Detect::Anchor::read_anchor_task(run.given.task));
                SA_TRY_ASSIGN(const std::vector<Vision::Training::AnchorBiasHead> heads,
                              Vision::Training::derive_anchor_bias_heads(
                                  model, run.given.input_sample, task, run.given.outputs));
                return Vision::Training::initialize_anchor_biases(task, heads, run.state);
            }

        } // namespace

        Result<RunSummary> run_anchor(ExecutionBackend& backend, PreparedRun& run)
        {
            // The registry returns the interface; the step takes the type behind it.
            SA_TRY_ASSIGN(const Vision::Training::AnchorObjective objective,
                          Vision::Training::build_anchor_objective(
                              run.given, run.spec.objective_attributes));

            if (run.spec.resume.empty() && run.spec.weights.empty())
            {
                SA_TRY(start_anchor_biases(run));
            }

            SA_TRY_ASSIGN(
                const std::unique_ptr<Vision::Training::AnchorTrainingStep> step,
                Vision::Training::AnchorTrainingStep::make(
                    backend, objective, run.built, run.pool, run.optimizer,
                    std::move(run.updates)));

            SA_TRY(check_start_batch(run));
            SA_TRY(step->restore_run_count(static_cast<std::size_t>(run.start.step_runs)));

            const std::int64_t epochs = run.spec.epochs;
            const std::size_t batches = run.dataset.batch_count();

            Vision::Training::AnchorLoopOptions options;
            options.epochs = epochs;
            options.schedule = run.schedule;
            options.first_epoch = static_cast<std::int64_t>(run.start.epoch);
            options.first_batch = static_cast<std::size_t>(run.start.batch);
            const auto first_iteration = static_cast<std::int64_t>(run.optimizer.step_count());
            options.on_update = [epochs, batches](
                                    const Vision::Training::AnchorProgress& progress)
            {
                if (is_reported_epoch(progress.epoch, epochs)
                    && progress.batch + 1 == batches)
                    print_epoch(progress, epochs);
            };
            options.after_update = [&run, &step, first_iteration](
                                       const Vision::Training::AnchorProgress& progress)
            {
                return checkpoint_after_update(
                    run,
                    UpdateAt{progress.epoch, progress.batch,
                             progress.iteration - first_iteration},
                    step->run_count());
            };

            // The progress lines carry what the report holds, so only its status is read.
            SA_TRY(Vision::Training::run_anchor_training(*step, run.dataset, options)
                       .status());

            const std::size_t forward_compiles = step->training().lead()->compile_count();
            SA_FAIL_IF(forward_compiles != 1, StatusCode::InvalidDescription,
                       "the part of the step before its heads was compiled {} times",
                       forward_compiles);

            RunSummary summary;
            summary.compile_count = step->training().step().compile_count();
            summary.progress = finished_progress(run, step->run_count());
            return summary;
        }

    } // namespace Apps
} // namespace SushiAI
