/**************************************************************************/
/* darknet_head_sections.cpp                                              */
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
 * @file darknet_head_sections.cpp
 * @brief Defines the reading of an anchor task into the [yolo] sections of a .cfg.
 * @author Mustafa Garip
 */

#include "darknet_head_sections.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                using Detect::Anchor::AnchorTask;

                /** @brief Holds the prefix the importer names head k's output with. */
                constexpr std::string_view K_HEAD_PREFIX = "head.";

                /** @brief Holds the one port a head publishes. */
                constexpr std::string_view K_PORT = "output";

                /** @brief Returns Unsupported unless the model's task is an anchor task. */
                Status check_task_type(const Config::Value& task)
                {
                    SA_FAIL_IF(task.is_null(), StatusCode::Unsupported,
                               "the model states no task, and the Darknet importer reads a model "
                               "only through its [yolo] sections");
                    const Config::Value absent;
                    const Config::Value& type = task.at_or("type", absent);
                    SA_TRY_ASSIGN(const std::string& name, type.as_string("task 'type'"));
                    SA_FAIL_IF(name != Detect::Anchor::K_ANCHOR_TASK_TYPE, StatusCode::Unsupported,
                               "the model's task is '{}', and a [yolo] section states an '{}' "
                               "task",
                               name, Detect::Anchor::K_ANCHOR_TASK_TYPE);
                    return Status();
                }

                /** @brief Returns Unsupported for what the heads of @p task cannot state. */
                Status check_heads(const AnchorTask& task)
                {
                    SA_FAIL_IF(Detect::Anchor::writes_assignment_section(task.assignment),
                               StatusCode::Unsupported,
                               "the task states an assignment, and a [yolo] section assigns "
                               "truths by the best shape alone");
                    for (std::size_t head = 0; head < task.heads.size(); ++head)
                    {
                        const Detect::Anchor::HeadOptions& options = task.heads[head].options;
                        SA_FAIL_IF(options.num_classes != task.num_classes,
                                   StatusCode::Unsupported,
                                   "head {} has {} classes and the task {}, and the importer "
                                   "gives every head the task's count",
                                   head, options.num_classes, task.num_classes);
                        SA_FAIL_IF(options.size != Detect::Anchor::SizeForm::Exponential,
                                   StatusCode::Unsupported,
                                   "head {} decodes its sizes as a squared sigmoid, which no "
                                   "[yolo] key states",
                                   head);
                        const std::string name =
                            std::string(K_HEAD_PREFIX) + std::to_string(head);
                        SA_FAIL_IF(task.heads[head].output != name, StatusCode::Unsupported,
                                   "head {} reads the output '{}', and the importer names it "
                                   "'{}'",
                                   head, task.heads[head].output, name);
                    }
                    return Status();
                }

                /** @brief Returns Unsupported unless the model publishes the heads, in order. */
                Status check_outputs(const Config::ModelSpec& model, const AnchorTask& task)
                {
                    SA_FAIL_IF(model.outputs.size() != task.heads.size(),
                               StatusCode::Unsupported,
                               "the model publishes {} outputs and its task has {} heads, and "
                               "the importer publishes the heads alone",
                               model.outputs.size(), task.heads.size());
                    for (std::size_t at = 0; at < model.outputs.size(); ++at)
                    {
                        const Config::OutputSpec& output = model.outputs[at];
                        SA_FAIL_IF(output.name != task.heads[at].output || output.port != K_PORT ||
                                       output.node < 0 ||
                                       static_cast<std::size_t>(output.node) >=
                                           model.layers.size(),
                                   StatusCode::Unsupported,
                                   "output {} '{}' is not the port 'output' of a row named "
                                   "after head {}",
                                   at, output.name, at);
                    }
                    return Status();
                }

                /** @brief Returns the lines that name the suppression of @p task. */
                std::vector<DarknetOptionText> describe_suppression(const AnchorTask& task)
                {
                    if (task.criterion == Detect::SuppressionCriterion::IoU)
                        return {};
                    if (task.distance_exponent == Detect::K_DEFAULT_DISTANCE_EXPONENT)
                        return {DarknetOptionText{"nms_kind", "greedynms"}};
                    return {DarknetOptionText{"nms_kind", "diounms"},
                            DarknetOptionText{"beta_nms", spell_float(task.distance_exponent)}};
                }
            } // namespace

            Result<AnchorTask> read_darknet_heads(const Config::ModelSpec& model)
            {
                SA_TRY(check_task_type(model.task));
                SA_TRY_ASSIGN(AnchorTask task, Detect::Anchor::read_anchor_task(model.task));
                SA_TRY(check_heads(task));
                SA_TRY(check_outputs(model, task));
                return task;
            }

            std::vector<DarknetOptionText> describe_head(
                const AnchorTask& task, std::size_t head, const std::vector<std::int64_t>& input,
                const std::vector<DarknetOptionText>& training)
            {
                std::string anchors;
                for (const Detect::Anchor::AnchorSize& anchor : task.layout.anchors)
                {
                    anchors.append(anchors.empty() ? "" : ",")
                        .append(spell_anchor_side(anchor.width, input[1]))
                        .append(",")
                        .append(spell_anchor_side(anchor.height, input[0]));
                }
                const Detect::Anchor::HeadOptions& options = task.heads[head].options;
                std::vector<DarknetOptionText> lines = {
                    {"mask", spell_integer_list(task.layout.head_anchors[head])},
                    {"anchors", anchors},
                    {"classes", spell_integer(task.num_classes)},
                    {"num", spell_integer(static_cast<std::int64_t>(task.layout.anchors.size()))}};
                if (options.scale_xy != 1.0f)
                    lines.push_back({"scale_x_y", spell_float(options.scale_xy)});
                if (options.coding == Detect::Anchor::HeadCoding::Activated)
                    lines.push_back({"new_coords", "1"});
                for (DarknetOptionText& line : describe_suppression(task))
                    lines.push_back(std::move(line));
                lines.insert(lines.end(), training.begin(), training.end());
                return lines;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
