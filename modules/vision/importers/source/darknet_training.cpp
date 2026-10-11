/**************************************************************************/
/* darknet_training.cpp                                                   */
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
 * @file darknet_training.cpp
 * @brief Implements the reader of a .cfg's solver settings and of its heads' training options.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_training.hpp>

#include <cstdint>
#include <string_view>
#include <utility>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>

#include "darknet_layer_table.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            using Detail::find_option;
            using Detail::read_integer;
            using Detail::read_real;
            using Detect::Anchor::BoxLoss;
            using Detect::Anchor::HeadTrainingOptions;
            using Detect::Anchor::ObjectnessTarget;

            /** @brief Holds the form of the complete IoU gradient every imported head takes. */
            constexpr Detect::BoxGradient K_IMPORTED_BOX_GRADIENT =
                Detect::BoxGradient::DarknetCompatible;

            /** @brief Holds the section type of one anchor head. */
            constexpr std::string_view K_YOLO_SECTION = "yolo";

            /** @brief Holds the one policy the reader accepts; an absent policy is the same. */
            constexpr std::string_view K_STEPS_POLICY = "steps";

            /** @brief Holds the one truth_thresh the reader accepts. */
            constexpr double K_ONLY_TRUTH_THRESHOLD = 1.0;

            /** @brief Holds the one label_smooth_eps the reader accepts. */
            constexpr double K_ONLY_LABEL_SMOOTHING = 0.0;

            /** @brief Holds one iou_loss name and the box loss it selects. */
            struct BoxLossRow
            {
                std::string_view name; /**< The value of the section's iou_loss key. */
                BoxLoss loss;          /**< The box term that value selects. */
            };

            /** @brief Lists every iou_loss this importer reads. */
            constexpr BoxLossRow K_BOX_LOSSES[] = {
                {"mse", BoxLoss::SquaredError},
                {"ciou", BoxLoss::CompleteIoU},
            };

            /** @brief Returns the box-loss row of @p name, or null when none holds it. */
            [[nodiscard]] const BoxLossRow* find_box_loss(std::string_view name) noexcept
            {
                for (const BoxLossRow& row : K_BOX_LOSSES)
                {
                    if (row.name == name)
                        return &row;
                }
                return nullptr;
            }

            /** @brief Reads @p key as a finite float, or returns @p fallback when absent. */
            [[nodiscard]] Result<float> read_narrowed(const DarknetSection& section,
                                                      std::string_view key, float fallback)
            {
                SA_TRY_ASSIGN(const double value,
                              read_real(section, key, static_cast<double>(fallback)));
                return static_cast<float>(value);
            }

            /** @brief Returns Unsupported naming the line of a @p key other than @p accepted. */
            [[nodiscard]] Status require_number(const DarknetSection& section,
                                                std::string_view key, double accepted)
            {
                SA_TRY_ASSIGN(const double value, read_real(section, key, accepted));
                SA_FAIL_IF(value != accepted, StatusCode::Unsupported,
                           "line {}: '{}' is {}, and this importer reads {} alone",
                           find_option(section, key)->line, key, value, accepted);
                return Status();
            }

            /**
             * @brief Reads steps and scales into @p solver; an absent key keeps its list empty.
             * @return InvalidDescription naming the line of a malformed entry, or of the
             *         section when the two lists differ in length.
             */
            [[nodiscard]] Status read_schedule(const DarknetSection& net, DarknetSolver& solver)
            {
                if (find_option(net, "steps") != nullptr)
                {
                    SA_TRY_ASSIGN(solver.steps, Detail::read_integer_list(net, "steps"));
                }
                if (find_option(net, "scales") != nullptr)
                {
                    SA_TRY_ASSIGN(solver.scales, Detail::read_real_list(net, "scales"));
                }
                SA_FAIL_IF(solver.steps.size() != solver.scales.size(),
                           StatusCode::InvalidDescription,
                           "line {}: 'steps' holds {} entries and 'scales' holds {}; each step "
                           "needs one scale",
                           net.line, solver.steps.size(), solver.scales.size());

                const DarknetOption* policy = find_option(net, "policy");
                SA_FAIL_IF(policy != nullptr && policy->value != K_STEPS_POLICY,
                           StatusCode::Unsupported,
                           "line {}: the policy '{}' is not read; only steps is", policy->line,
                           policy->value);
                return Status();
            }

            /**
             * @brief Reads the solver settings of the first section over their defaults.
             * @return InvalidDescription naming the line of a malformed value or of a batch or
             *         subdivisions below 1; Unsupported for a policy other than steps.
             */
            [[nodiscard]] Result<DarknetSolver> read_solver(const DarknetSection& net)
            {
                DarknetSolver solver;
                SA_TRY_ASSIGN(solver.learning_rate,
                              read_real(net, "learning_rate", solver.learning_rate));
                SA_TRY_ASSIGN(solver.momentum, read_real(net, "momentum", solver.momentum));
                SA_TRY_ASSIGN(solver.decay, read_real(net, "decay", solver.decay));
                SA_TRY_ASSIGN(solver.burn_in, read_integer(net, "burn_in", solver.burn_in));
                SA_TRY_ASSIGN(solver.max_batches,
                              read_integer(net, "max_batches", solver.max_batches));
                SA_TRY_ASSIGN(solver.batch, read_integer(net, "batch", solver.batch));
                SA_TRY_ASSIGN(solver.subdivisions,
                              read_integer(net, "subdivisions", solver.subdivisions));
                SA_FAIL_IF(solver.batch < 1 || solver.subdivisions < 1,
                           StatusCode::InvalidDescription,
                           "line {}: the [{}] section needs a batch and subdivisions of 1 or "
                           "more; got {} and {}",
                           net.line, net.type, solver.batch, solver.subdivisions);

                SA_TRY(read_schedule(net, solver));
                return solver;
            }

            /**
             * @brief Reads the box loss of a head from iou_loss; an absent key keeps @p loss.
             * @return Unsupported naming the line of a name outside K_BOX_LOSSES.
             */
            [[nodiscard]] Status read_box_loss(const DarknetSection& section, BoxLoss& loss)
            {
                const DarknetOption* named = find_option(section, "iou_loss");
                if (named == nullptr)
                    return Status();

                const BoxLossRow* row = find_box_loss(named->value);
                SA_FAIL_IF(row == nullptr, StatusCode::Unsupported,
                           "line {}: the iou_loss '{}' is not one this importer reads",
                           named->line, named->value);

                loss = row->loss;
                return Status();
            }

            /**
             * @brief Reads the objectness target of a head from objectness_smooth: 1 is BestIoU.
             * @return InvalidDescription naming the line of a value other than 0 and 1.
             */
            [[nodiscard]] Status read_objectness_target(const DarknetSection& section,
                                                        ObjectnessTarget& target)
            {
                SA_TRY_ASSIGN(const std::int64_t smooth,
                              read_integer(section, "objectness_smooth", 0));
                SA_FAIL_IF(smooth != 0 && smooth != 1, StatusCode::InvalidDescription,
                           "line {}: 'objectness_smooth' is {}; it is 0 or 1",
                           find_option(section, "objectness_smooth")->line, smooth);

                target = smooth == 1 ? ObjectnessTarget::BestIoU : ObjectnessTarget::One;
                return Status();
            }

            /**
             * @brief Reads the training options of one [yolo] section over their defaults.
             * @return The options; Unsupported naming the line of a value this importer does
             *         not read; InvalidDescription naming the line of a malformed value, or
             *         the section's line when validate_head_training refuses the options.
             */
            [[nodiscard]] Result<HeadTrainingOptions> read_head(const DarknetSection& section)
            {
                HeadTrainingOptions options;
                options.box_gradient = K_IMPORTED_BOX_GRADIENT;
                SA_TRY(read_box_loss(section, options.box_loss));
                SA_TRY(read_objectness_target(section, options.objectness_target));
                SA_TRY_ASSIGN(options.box_weight,
                              read_narrowed(section, "iou_normalizer", options.box_weight));
                SA_TRY_ASSIGN(options.object_weight,
                              read_narrowed(section, "obj_normalizer", options.object_weight));
                // cls_normalizer scales no delta here, so its number is checked and dropped.
                SA_TRY(read_real(section, "cls_normalizer", 1.0));
                SA_TRY_ASSIGN(options.max_delta,
                              read_narrowed(section, "max_delta", options.max_delta));
                SA_TRY_ASSIGN(options.thresholds.match_threshold,
                              read_narrowed(section, "iou_thresh",
                                            options.thresholds.match_threshold));
                SA_TRY_ASSIGN(options.thresholds.ignore_threshold,
                              read_narrowed(section, "ignore_thresh",
                                            options.thresholds.ignore_threshold));

                SA_TRY(require_number(section, "truth_thresh", K_ONLY_TRUTH_THRESHOLD));
                SA_TRY(require_number(section, "label_smooth_eps", K_ONLY_LABEL_SMOOTHING));
                SA_TRY(SA_ANNOTATE(Detect::Anchor::validate_head_training(options), "line {}",
                                   section.line));
                return options;
            }

        } // namespace

        Result<DarknetTraining> read_darknet_training(const DarknetConfig& config)
        {
            DarknetTraining training;
            SA_TRY_ASSIGN(training.solver, read_solver(config.net));
            for (const DarknetSection& section : config.layers)
            {
                if (section.type != K_YOLO_SECTION)
                    continue;

                SA_TRY_ASSIGN(HeadTrainingOptions head, read_head(section));
                training.heads.push_back(std::move(head));
            }
            return training;
        }

    } // namespace Vision::Importers
} // namespace SushiAI
