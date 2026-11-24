/**************************************************************************/
/* darknet_training_lines.cpp                                             */
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
 * @file darknet_training_lines.cpp
 * @brief Defines the writing of a training file and a data block as the lines of a .cfg.
 * @author Mustafa Garip
 */

#include "darknet_training_lines.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                using Detect::Anchor::HeadTrainingOptions;

                /** @brief Holds the one optimizer a .cfg states. */
                constexpr std::string_view K_SGD = "sgd";

                /** @brief Holds the one pipeline whose keys a .cfg states. */
                constexpr std::string_view K_DARKNET_PIPELINE = "darknet";

                /** @brief Lists the keys of the data block a .cfg states, and its pipeline. */
                constexpr std::string_view K_DATA_KEYS[] = {"pipeline", "jitter", "hue",
                                                            "saturation", "exposure"};

                /** @brief Returns whether @p key is one of @p keys. */
                template<std::size_t N>
                bool is_listed(const std::string& key, const std::string_view (&keys)[N])
                {
                    for (const std::string_view listed : keys)
                    {
                        if (listed == key)
                            return true;
                    }
                    return false;
                }

                /** @brief Appends the lines of the objective of @p options to @p lines. */
                void describe_objective(const HeadTrainingOptions& options,
                                        DarknetTrainingLines& lines)
                {
                    const bool ciou = options.box_loss == Detect::Anchor::BoxLoss::CompleteIoU;
                    const bool best = options.objectness_target ==
                                      Detect::Anchor::ObjectnessTarget::BestIoU;
                    lines.head.push_back({"iou_loss", ciou ? "ciou" : "mse"});
                    lines.head.push_back({"objectness_smooth", best ? "1" : "0"});
                    lines.head.push_back({"iou_normalizer", spell_float(options.box_weight)});
                    lines.head.push_back({"obj_normalizer", spell_float(options.object_weight)});
                    lines.head.push_back({"max_delta", spell_float(options.max_delta)});
                    lines.head.push_back(
                        {"iou_thresh", spell_float(options.thresholds.match_threshold)});
                    lines.head.push_back(
                        {"ignore_thresh", spell_float(options.thresholds.ignore_threshold)});
                    const HeadTrainingOptions plain;
                    if (options.box_gradient != Detect::BoxGradient::DarknetCompatible)
                        lines.notes.emplace_back("a .cfg trains with Darknet's box gradient; the "
                                                 "objective's box_gradient is left out");
                    if (options.class_gate != plain.class_gate ||
                        options.box_gain != plain.box_gain ||
                        options.class_gain != plain.class_gain ||
                        options.objectness_gain != plain.objectness_gain)
                        lines.notes.emplace_back("a .cfg states no class gate and no gain; the "
                                                 "objective's class_gate and gains are left out");
                }

                /** @brief Appends the momentum and decay of an sgd @p optimizer to @p lines. */
                Status describe_optimizer(const Config::Value& optimizer,
                                          DarknetTrainingLines& lines)
                {
                    const Config::Value absent;
                    const Config::Value& type = optimizer.at_or("type", absent);
                    if (type.kind() != Config::Kind::String ||
                        type.as_string("optimizer type").value() != K_SGD)
                    {
                        lines.notes.emplace_back("a .cfg states an sgd solver; the optimizer is "
                                                 "left out");
                        return Status();
                    }
                    for (const std::string& key : optimizer.keys())
                    {
                        if (key == "momentum" || key == "weight_decay")
                        {
                            SA_TRY_ASSIGN(const Config::Value& value,
                                          optimizer.at(key, "optimizer"));
                            SA_TRY_ASSIGN(const double real, value.as_real("optimizer " + key));
                            lines.network.push_back(
                                {key == "momentum" ? "momentum" : "decay", spell_real(real)});
                        }
                        else if (key != "type")
                            lines.notes.push_back("a .cfg states no optimizer '" + key +
                                                  "'; it is left out");
                    }
                    return Status();
                }

                /** @brief Appends the lines of a step @p schedule section to @p lines. */
                Status describe_schedule(const Config::Value& section,
                                         DarknetTrainingLines& lines)
                {
                    const Config::Value absent;
                    const Config::Value& type = section.at_or("type", absent);
                    if (type.kind() != Config::Kind::String ||
                        type.as_string("").value() != Optim::K_STEP_SCHEDULE_TYPE)
                    {
                        lines.notes.emplace_back("a .cfg states a step schedule; the schedule is "
                                                 "left out");
                        return Status();
                    }
                    SA_TRY_ASSIGN(const Optim::StepSchedule schedule,
                                  Optim::read_step_schedule_section(section));
                    lines.network.push_back({"learning_rate", spell_real(schedule.base_rate)});
                    lines.network.push_back(
                        {"burn_in", spell_integer(schedule.warmup_iterations)});
                    lines.network.push_back({"policy", "steps"});
                    if (!schedule.steps.empty())
                    {
                        std::string scales;
                        for (const double scale : schedule.scales)
                            scales.append(scales.empty() ? "" : ",").append(spell_real(scale));
                        lines.network.push_back({"steps", spell_integer_list(schedule.steps)});
                        lines.network.push_back({"scales", scales});
                    }
                    if (schedule.warmup_power != Optim::StepSchedule().warmup_power)
                        lines.notes.emplace_back("a .cfg ramps its warm-up with the power 4; the "
                                                 "schedule's warmup_power is left out");
                    return Status();
                }

                /** @brief Appends the training lines of the recipe @p training to @p lines. */
                Status describe_recipe(const Config::Value& training, DarknetTrainingLines& lines)
                {
                    for (const std::string& key : training.keys())
                    {
                        SA_TRY_ASSIGN(const Config::Value& section, training.at(key, "training"));
                        if (key == "objective")
                        {
                            SA_TRY_ASSIGN(const HeadTrainingOptions options,
                                          Detect::Anchor::read_head_training_section(section));
                            describe_objective(options, lines);
                        }
                        else if (key == "optimizer")
                        {
                            SA_TRY(describe_optimizer(section, lines));
                        }
                        else if (key == "schedule")
                        {
                            SA_TRY(describe_schedule(section, lines));
                        }
                        else
                        {
                            lines.notes.push_back("a .cfg states no training '" + key +
                                                  "'; it is left out");
                        }
                    }
                    return Status();
                }

                /** @brief Appends the augmentation lines of a darknet @p data block. */
                Status describe_data(const Config::Value& data, DarknetTrainingLines& lines)
                {
                    const Config::Value absent;
                    const Config::Value& pipeline = data.at_or("pipeline", absent);
                    if (pipeline.kind() != Config::Kind::String ||
                        pipeline.as_string("").value() != K_DARKNET_PIPELINE)
                    {
                        lines.notes.emplace_back("a .cfg states the darknet pipeline alone; the "
                                                 "data block is left out");
                        return Status();
                    }
                    for (const std::string& key : data.keys())
                    {
                        if (!is_listed(key, K_DATA_KEYS))
                        {
                            lines.notes.push_back("a .cfg states no data '" + key +
                                                  "'; it is left out");
                            continue;
                        }
                        if (key == "pipeline")
                            continue;
                        SA_TRY_ASSIGN(const Config::Value& value, data.at(key, "data"));
                        SA_TRY_ASSIGN(const double real, value.as_real("data " + key));
                        DarknetOptionText line{key, spell_float(static_cast<float>(real))};
                        if (key == "jitter")
                            lines.head.push_back(std::move(line));
                        else
                            lines.network.push_back(std::move(line));
                    }
                    return Status();
                }
            } // namespace

            Result<DarknetTrainingLines> describe_darknet_training(const IO::ImportedModel& model)
            {
                DarknetTrainingLines lines;
                if (model.training.kind() == Config::Kind::Object)
                {
                    SA_TRY(describe_recipe(model.training, lines));
                }
                else
                {
                    lines.notes.emplace_back("no training file was given; the .cfg holds the "
                                             "architecture alone");
                }
                if (model.data.kind() == Config::Kind::Object)
                {
                    SA_TRY(describe_data(model.data, lines));
                }
                return lines;
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
