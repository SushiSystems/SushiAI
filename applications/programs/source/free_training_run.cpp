/**************************************************************************/
/* free_training_run.cpp                                                  */
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
 * @file free_training_run.cpp
 * @brief Implements the `anchor_free` run row, trained through FreeTrainingStep.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#include "vision_training_runs.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/training/free_bias_init.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>
#include <SushiAI/vision/training/free_training_loop.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>

#include "epoch_loop.hpp"

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Prints the loss and the terms of the last update of an anchor-free epoch. */
            void print_free_epoch(const Vision::Training::FreeProgress& progress,
                                  std::int64_t epochs)
            {
                std::cout << SushiLog::format_text(
                    "  epoch {}/{} loss {}   box {}   class {}   dfl {}   rate {}\n",
                    SushiLog::right(progress.epoch, 3), SushiLog::left(epochs, 3),
                    SushiLog::fixed(progress.terms.total, 6),
                    SushiLog::fixed(progress.terms.box, 6),
                    SushiLog::fixed(progress.terms.classes, 6),
                    SushiLog::fixed(progress.terms.distribution, 6),
                    SushiLog::general(progress.rate, 6));
            }

            /**
             * @brief Returns the dotted name the detect row's state hangs under, found as the
             *        one entry that ends in the first one-to-many box bias.
             * @return The prefix, empty for a detect row at the root, or InvalidDescription
             *         when no entry or more than one ends so.
             */
            [[nodiscard]] Result<std::string> find_free_head_prefix(const NN::ModelState& state)
            {
                const std::string suffix =
                    std::string(Vision::Detect::Free::K_DETECT_BOX_BRANCH) + ".0."
                    + std::to_string(Vision::Detect::Free::K_DETECT_LAST_CONV_INDEX) + ".bias";

                std::vector<std::string> found;
                for (const NN::StateEntry& entry : state.entries())
                {
                    const std::string& name = entry.name;
                    if (name == suffix)
                        found.emplace_back();
                    else if (name.size() > suffix.size() + 1
                             && name.compare(name.size() - suffix.size(), suffix.size(), suffix)
                                    == 0
                             && name[name.size() - suffix.size() - 1] == '.')
                        found.push_back(name.substr(0, name.size() - suffix.size() - 1));
                }

                SA_FAIL_IF(found.size() != 1, StatusCode::InvalidDescription,
                           "the model state holds {} entries ending in '{}'; an anchor-free "
                           "model holds one detect row",
                           found.size(), suffix);
                return found.front();
            }

        } // namespace

        Result<RunSummary> run_anchor_free(ExecutionBackend& backend, PreparedRun& run)
        {
            // The registry returns the interface; the step takes the type behind it.
            SA_TRY_ASSIGN(const Vision::Training::FreeObjective objective,
                          Vision::Training::build_free_objective(
                              run.given, run.spec.objective_attributes));

            if (run.spec.resume.empty() && run.spec.weights.empty())
            {
                SA_TRY_ASSIGN(const std::string prefix, find_free_head_prefix(run.state));
                SA_TRY(Vision::Training::initialize_free_biases(objective.facts(), prefix,
                                                                run.state));
            }

            SA_TRY_ASSIGN(
                const std::unique_ptr<Vision::Training::FreeTrainingStep> step,
                Vision::Training::FreeTrainingStep::make(
                    backend, objective, run.built, run.pool, run.optimizer,
                    std::move(run.updates)));

            SA_TRY(check_start_batch(run));
            SA_TRY(step->restore_run_count(static_cast<std::size_t>(run.start.step_runs)));

            const std::int64_t epochs = run.spec.epochs;
            const std::size_t batches = run.dataset.batch_count();

            Vision::Training::FreeLoopOptions options;
            options.epochs = epochs;
            options.schedule = run.schedule;
            options.first_epoch = static_cast<std::int64_t>(run.start.epoch);
            options.first_batch = static_cast<std::size_t>(run.start.batch);
            const auto first_iteration = static_cast<std::int64_t>(run.optimizer.step_count());
            options.on_update = [epochs, batches](
                                    const Vision::Training::FreeProgress& progress)
            {
                if (is_reported_epoch(progress.epoch, epochs)
                    && progress.batch + 1 == batches)
                    print_free_epoch(progress, epochs);
            };
            options.after_update = [&run, &step, first_iteration](
                                       const Vision::Training::FreeProgress& progress)
            {
                return checkpoint_after_update(
                    run,
                    UpdateAt{progress.epoch, progress.batch,
                             progress.iteration - first_iteration},
                    step->run_count());
            };

            // The progress lines carry what the report holds, so only its status is read.
            SA_TRY(Vision::Training::run_free_training(*step, run.dataset, options).status());

            const std::size_t forward_compiles = step->training().lead()->compile_count();
            SA_FAIL_IF(forward_compiles != 1, StatusCode::InvalidDescription,
                       "the part of the step before its ports was compiled {} times",
                       forward_compiles);

            RunSummary summary;
            summary.compile_count = step->training().step().compile_count();
            summary.progress = finished_progress(run, step->run_count());
            return summary;
        }

    } // namespace Apps
} // namespace SushiAI
