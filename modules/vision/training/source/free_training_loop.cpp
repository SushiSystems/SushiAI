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
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the rank of the boxes role of one sample, [M, 4]. */
        constexpr std::int32_t K_BOXES_RANK = 2;

        /**
         * @brief Checks that @p dataset can feed @p step for the run @p options describes.
         * @return Success; InvalidDescription for no epoch, a negative first epoch, no batch
         *         or no [M, 4] boxes role; Incompatible for another batch size or a first
         *         batch the dataset lacks; or the failure of validate_rate_schedule or of
         *         match_roles.
         */
        [[nodiscard]] Status check_run(const FreeTrainingStep& step,
                                       const Data::Dataset& dataset,
                                       const FreeLoopOptions& options)
        {
            const FreeObjective& objective = step.objective();

            SA_FAIL_IF(options.epochs < 1, StatusCode::InvalidDescription,
                       "An anchor-free run needs at least one epoch; got {}", options.epochs);
            SA_FAIL_IF(options.first_epoch < 0, StatusCode::InvalidDescription,
                       "An anchor-free run starts at an epoch index of 0 or above; got {}",
                       options.first_epoch);
            SA_FAIL_IF(dataset.batch_count() == 0, StatusCode::InvalidDescription,
                       "The '{}' data holds no batches", dataset.name());
            SA_FAIL_IF(options.first_batch >= dataset.batch_count(), StatusCode::Incompatible,
                       "The run starts at batch {} of its first epoch, but the '{}' data holds "
                       "{} batches an epoch",
                       options.first_batch, dataset.name(), dataset.batch_count());
            SA_FAIL_IF(dataset.batch_size() != objective.batch(), StatusCode::Incompatible,
                       "The '{}' data has a batch of {}, but the ports were traced for {}",
                       dataset.name(), dataset.batch_size(), objective.batch());

            if (options.schedule != nullptr)
            {
                SA_TRY(Optim::validate_rate_schedule(*options.schedule));
            }

            const Data::TargetDescription* const boxes =
                Data::find_target(dataset.targets(), Data::K_ROLE_BOXES);
            SA_FAIL_IF(boxes == nullptr || boxes->sample_shape.rank() != K_BOXES_RANK,
                       StatusCode::InvalidDescription,
                       "The '{}' data publishes no '{}' role of [M, {}] per sample, which "
                       "states the object slots a detection step reads",
                       dataset.name(), Data::K_ROLE_BOXES, K_TRUTH_BOX_VALUES);

            SA_TRY_ASSIGN(const std::vector<Data::TargetDescription> roles,
                          describe_truth_roles(boxes->sample_shape.dim(0),
                                               objective.facts().classes));

            return Train::match_roles(roles, dataset);
        }

        /** @brief Calls the callback of @p options, when it holds one, and guards its throw. */
        [[nodiscard]] Status report_progress(const FreeLoopOptions& options,
                                             const FreeProgress& progress)
        {
            if (!options.on_update)
                return Status();

            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute, [&]()
            {
                options.on_update(progress);
                return Status();
            });
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

                SA_TRY(report_progress(options, progress));
            }
        }

        return report;
    }

} // namespace SushiAI::Vision::Training
