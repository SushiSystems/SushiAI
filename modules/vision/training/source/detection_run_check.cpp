/**************************************************************************/
/* detection_run_check.cpp                                                */
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

/**
 * @file detection_run_check.cpp
 * @brief Implements the check of a dataset and a run plan that both detection training loops run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/detection_run_check.hpp>

#include <vector>

#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the rank of the boxes role of one sample, [M, 4]. */
        constexpr std::int32_t K_BOXES_RANK = 2;
    } // namespace

    Status check_detection_run(const Data::Dataset& dataset, const DetectionRunSubject& subject,
                               const DetectionRunPlan& plan)
    {
        SA_FAIL_IF(plan.epochs < 1, StatusCode::InvalidDescription,
                   "{} needs at least one epoch; got {}", subject.run_name, plan.epochs);
        SA_FAIL_IF(plan.first_epoch < 0, StatusCode::InvalidDescription,
                   "{} starts at an epoch index of 0 or above; got {}", subject.run_name,
                   plan.first_epoch);
        SA_FAIL_IF(dataset.batch_count() == 0, StatusCode::InvalidDescription,
                   "The '{}' data holds no batches", dataset.name());
        SA_FAIL_IF(plan.first_batch >= dataset.batch_count(), StatusCode::Incompatible,
                   "The run starts at batch {} of its first epoch, but the '{}' data holds "
                   "{} batches an epoch",
                   plan.first_batch, dataset.name(), dataset.batch_count());
        SA_FAIL_IF(dataset.batch_size() != subject.batch, StatusCode::Incompatible,
                   "The '{}' data has a batch of {}, but the {} {}", dataset.name(),
                   dataset.batch_size(), subject.batch_origin, subject.batch);

        if (plan.schedule != nullptr)
        {
            SA_TRY(Optim::validate_rate_schedule(*plan.schedule));
        }

        const Data::TargetDescription* const boxes =
            Data::find_target(dataset.targets(), Data::K_ROLE_BOXES);
        SA_FAIL_IF(boxes == nullptr || boxes->sample_shape.rank() != K_BOXES_RANK,
                   StatusCode::InvalidDescription,
                   "The '{}' data publishes no '{}' role of [M, {}] per sample, which "
                   "states the object slots a detection step reads",
                   dataset.name(), Data::K_ROLE_BOXES, K_TRUTH_BOX_VALUES);

        SA_TRY_ASSIGN(const std::vector<Data::TargetDescription> roles,
                      describe_truth_roles(boxes->sample_shape.dim(0), subject.classes));

        return Train::match_roles(roles, dataset);
    }

} // namespace SushiAI::Vision::Training
