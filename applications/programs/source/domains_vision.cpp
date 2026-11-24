/**************************************************************************/
/* domains_vision.cpp                                                     */
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
 * @file domains_vision.cpp
 * @brief Adds the rows of vision to the app registries.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table".
 */

#include "domains.hpp"

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/yolox/yolox_task.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/training/free_objective.hpp>

#include "detection_reports.hpp"
#include "inference_run.hpp"
#include "training_run.hpp"
#include "vision_training_runs.hpp"

namespace SushiAI
{
    namespace Apps
    {
        Status register_vision_domain(Registries& registries)
        {
            SA_TRY(Vision::register_nodes(registries.nodes));
            SA_TRY(Vision::register_datasets(registries.datasets));
            SA_TRY(Vision::register_objectives(registries.objectives));
            SA_TRY(registries.runs.add(
                RunEntry{Vision::Detect::Anchor::K_ANCHOR_OBJECTIVE_TYPE, &run_anchor}));
            SA_TRY(registries.runs.add(
                RunEntry{Vision::Training::K_FREE_OBJECTIVE_TYPE, &run_anchor_free}));
            SA_TRY(registries.inferences.add(InferenceEntry{
                Vision::Detect::Anchor::K_ANCHOR_TASK_TYPE, &report_anchor_detections}));
            SA_TRY(registries.inferences.add(InferenceEntry{
                Vision::Detect::Free::K_FREE_TASK_TYPE, &report_free_detections}));
            SA_TRY(registries.inferences.add(InferenceEntry{
                Vision::Detect::Yolox::K_YOLOX_TASK_TYPE, &report_yolox_detections}));
            SA_TRY(Vision::register_importers(registries.importers));
            SA_TRY(Vision::register_exporters(registries.exporters));

            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
