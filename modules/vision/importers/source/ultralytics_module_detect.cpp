/**************************************************************************/
/* ultralytics_module_detect.cpp                                          */
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
 * @file ultralytics_module_detect.cpp
 * @brief Defines the translation of an Ultralytics or a YOLOv5 Detect entry and its task.
 * @author Mustafa Garip
 */

#include "ultralytics_module_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Holds the score threshold of Ultralytics' prediction. */
                constexpr double K_SCORE_THRESHOLD = 0.25;

                /** @brief Holds the suppression threshold of Ultralytics' prediction. */
                constexpr double K_SUPPRESSION_THRESHOLD = 0.7;

                /** @brief Holds the values of one anchor slot besides the classes: box and object. */
                constexpr std::int64_t K_ANCHOR_VALUES = 5;

                /** @brief Holds the centre stretch of the YOLOv5 decoding, 2 sigmoid - 0.5. */
                constexpr double K_SCALE_XY = 2.0;

                /** @brief Returns the absolute row @p reference of @p entry names. */
                std::int64_t absolute_row(const ModuleEntry& entry, std::int64_t reference) noexcept
                {
                    return reference < 0 ? entry.index + reference : reference;
                }

                /** @brief Returns the anchor-free task section with Ultralytics' thresholds. */
                Config::Value anchor_free_task()
                {
                    return Config::Value::object(
                        {"type", "score_threshold", "suppression"},
                        {Config::Value::string("anchor_free_detection"),
                         Config::Value::real(K_SCORE_THRESHOLD),
                         Config::Value::object({"criterion", "threshold"},
                                               {Config::Value::string("iou"),
                                                Config::Value::real(K_SUPPRESSION_THRESHOLD)})});
                }

                /** @brief Translates the anchor-free head of one detect row. */
                Status translate_anchor_free(const ModuleEntry& entry, std::int64_t classes,
                                             ModuleProgress& progress)
                {
                    const std::int64_t row = static_cast<std::int64_t>(progress.layers.size());
                    append_row(progress, entry, "detect",
                               {"classes", "reg_max", "end2end", "class_branch"},
                               {Config::Value::integer(classes),
                                Config::Value::integer(progress.reg_max),
                                Config::Value::boolean(progress.end2end),
                                Config::Value::string(progress.legacy ? "plain" : "depthwise")});
                    progress.outputs.push_back(
                        Config::Value::object({"node"}, {Config::Value::integer(row)}));
                    progress.task = anchor_free_task();
                    // The distribution ramp is fixed, so the model computes it and holds no record.
                    progress.tensor_rules.dropped.push_back(std::to_string(entry.index) + ".dfl.");
                    return Status();
                }

                /**
                 * @brief Returns the anchors of one level, stated in pixels, as fractions of the
                 *        input: each width over its width and each height over its height.
                 */
                Result<std::vector<Config::Value>> level_anchors(const Config::Value& level,
                                                                 UltralyticsInputSize input,
                                                                 const ModuleEntry& entry)
                {
                    SA_TRY_ASSIGN(const std::vector<Config::Value>& values,
                                  level.as_array("anchors"));
                    SA_FAIL_IF(values.empty() || values.size() % 2 != 0,
                               StatusCode::InvalidDescription,
                               "entry {} (Detect) states a level of {} anchor values, not pairs",
                               entry.index, values.size());
                    std::vector<Config::Value> pairs;
                    for (std::size_t i = 0; i < values.size(); i += 2)
                    {
                        SA_TRY_ASSIGN(const double width, values[i].as_real("anchors"));
                        SA_TRY_ASSIGN(const double height, values[i + 1].as_real("anchors"));
                        pairs.push_back(Config::Value::array(
                            {Config::Value::real(width / static_cast<double>(input.width)),
                             Config::Value::real(height / static_cast<double>(input.height))}));
                    }
                    return pairs;
                }

                /**
                 * @brief Translates the YOLOv5 head: one 1x1 convolution per level, published
                 *        as head_<level>, and the anchor task that decodes them.
                 */
                Status translate_anchored(const ModuleEntry& entry, std::int64_t classes,
                                          const Config::Value& levels, ModuleProgress& progress)
                {
                    SA_TRY_ASSIGN(const std::vector<Config::Value>& stated,
                                  levels.as_array("anchors"));
                    SA_FAIL_IF(stated.size() != entry.from.size(), StatusCode::InvalidDescription,
                               "entry {} (Detect) reads {} levels and states anchors for {}",
                               entry.index, entry.from.size(), stated.size());
                    std::vector<Config::Value> anchors;
                    std::vector<Config::Value> heads;
                    for (std::size_t level = 0; level < stated.size(); ++level)
                    {
                        SA_TRY_ASSIGN(std::vector<Config::Value> pairs,
                                      level_anchors(stated[level], progress.image_size, entry));
                        const std::int64_t slots = static_cast<std::int64_t>(pairs.size());
                        std::vector<Config::Value> slot_indices;
                        for (const Config::Value& pair : pairs)
                        {
                            slot_indices.push_back(Config::Value::integer(
                                static_cast<std::int64_t>(anchors.size())));
                            anchors.push_back(pair);
                        }

                        const std::int64_t row = static_cast<std::int64_t>(progress.layers.size());
                        const std::string output = "head_" + std::to_string(level);
                        progress.layers.push_back(Config::Value::object(
                            {"type", "from", "filters", "kernel", "use_bias"},
                            {Config::Value::string("conv2d"),
                             Config::Value::integer(absolute_row(entry, entry.from[level])),
                             Config::Value::integer(slots * (classes + K_ANCHOR_VALUES)),
                             Config::Value::integer(1), Config::Value::boolean(true)}));
                        progress.outputs.push_back(Config::Value::object(
                            {"node", "port", "name"},
                            {Config::Value::integer(row), Config::Value::string("output"),
                             Config::Value::string(output)}));
                        heads.push_back(Config::Value::object(
                            {"output", "anchors", "coding", "scale_xy", "size"},
                            {Config::Value::string(output),
                             Config::Value::array(std::move(slot_indices)),
                             Config::Value::string("logit"), Config::Value::real(K_SCALE_XY),
                             Config::Value::string("squared_sigmoid")}));
                        progress.tensor_rules.renames.emplace_back(
                            std::to_string(entry.index) + ".m." + std::to_string(level) + ".",
                            std::to_string(row) + ".");
                    }
                    progress.task = Config::Value::object(
                        {"type", "classes", "anchors", "heads", "suppression", "assignment"},
                        {Config::Value::string("anchor_detection"), Config::Value::integer(classes),
                         Config::Value::array(std::move(anchors)),
                         Config::Value::array(std::move(heads)),
                         Config::Value::object({"criterion"}, {Config::Value::string("iou")}),
                         Config::Value::object({"rule"}, {Config::Value::string("shape_ratio")})});
                    // The anchors buffer is the task's anchors over the strides.
                    progress.tensor_rules.dropped.push_back(std::to_string(entry.index) +
                                                            ".anchors");
                    return Status();
                }
            } // namespace

            Status translate_detect(const ModuleEntry& entry, ModuleProgress& progress)
            {
                SA_TRY(reject_extra_arguments(entry, 2));
                SA_TRY_ASSIGN(const std::int64_t classes,
                              argument_int(entry, 0, progress.classes));
                SA_FAIL_IF(classes < 1, StatusCode::InvalidDescription,
                           "entry {} (Detect) states {} classes", entry.index, classes);
                for (const std::int64_t reference : entry.from)
                {
                    SA_TRY_ASSIGN(const std::int64_t part,
                                  source_channels(entry, progress, reference));
                    static_cast<void>(part);
                }

                const bool anchored = entry.args.size() == 2 && !entry.args[1].is_null();
                const Status translated =
                    anchored ? translate_anchored(entry, classes, entry.args[1], progress)
                             : translate_anchor_free(entry, classes, progress);
                SA_TRY(translated);
                progress.channels.push_back(0);
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
