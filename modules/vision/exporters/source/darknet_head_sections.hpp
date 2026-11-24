/**************************************************************************/
/* darknet_head_sections.hpp                                              */
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
 * @file darknet_head_sections.hpp
 * @brief Declares the reading of an anchor task into the [yolo] sections of a .cfg.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /**
             * @brief Reads the task of @p model as the heads a .cfg states.
             * @return The task; Unsupported for a model with no task, a task of another type,
             *         or one whose assignment, size form, class counts or outputs a [yolo]
             *         section cannot state; or what read_anchor_task refuses.
             */
            [[nodiscard]] Result<Detect::Anchor::AnchorTask> read_darknet_heads(
                const Config::ModelSpec& model);

            /**
             * @brief Returns the lines of the [yolo] section of head @p head, its training
             *        lines @p training last.
             * @param input The network input [H, W, C], which the anchors are fractions of.
             */
            [[nodiscard]] std::vector<DarknetOptionText> describe_head(
                const Detect::Anchor::AnchorTask& task, std::size_t head,
                const std::vector<std::int64_t>& input,
                const std::vector<DarknetOptionText>& training);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
