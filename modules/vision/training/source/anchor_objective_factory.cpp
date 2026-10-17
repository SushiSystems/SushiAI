/**************************************************************************/
/* anchor_objective_factory.cpp                                           */
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
 * @file anchor_objective_factory.cpp
 * @brief Implements the factory that builds an anchor objective from the files of a run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/anchor_objective_factory.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the rank of one input sample, [H, W, C]. */
        constexpr std::int32_t K_SAMPLE_RANK = 3;

        /** @brief Holds the rank of a head's published output, [B, H, W, channels]. */
        constexpr std::int32_t K_HEAD_RANK = 4;

        /** @brief Returns the names of @p outputs in quotes, or "(none)". */
        std::string list_outputs(const std::vector<Train::OutputShape>& outputs)
        {
            std::string text;
            for (const Train::OutputShape& output : outputs)
            {
                text += text.empty() ? "'" : ", '";
                text += output.name + "'";
            }
            return text.empty() ? std::string("(none)") : text;
        }

        /**
         * @brief Describes head @p index of @p task from the output the model published.
         * @return The head; InvalidDescription for an output the model does not publish;
         *         Incompatible for one that is not [given.batch, H, W, A * (5 + C)].
         */
        [[nodiscard]] Result<AnchorHead> describe_head(
            const Train::ObjectiveContext& given, const Detect::Anchor::AnchorTask& task,
            const Detect::Anchor::HeadTrainingOptions& training, std::size_t index)
        {
            const Detect::Anchor::AnchorTaskHead& head = task.heads[index];
            const std::int64_t head_index = static_cast<std::int64_t>(index);

            const Train::OutputShape* const output =
                Train::find_output_shape(given.outputs, head.output);
            SA_FAIL_IF(output == nullptr, StatusCode::InvalidDescription,
                       "the task's head {} reads the output '{}', which the model does not "
                       "publish; it publishes {}",
                       index, head.output, list_outputs(given.outputs));

            SA_TRY(Detect::Anchor::validate_head(head.options, task.layout, head_index));
            const std::int64_t channels =
                Detect::Anchor::count_head_channels(head.options, task.layout, head_index);

            const Shape& shape = output->shape;
            SA_FAIL_IF(shape.rank() != K_HEAD_RANK || shape.dim(0) != given.batch
                           || shape.dim(3) != channels,
                       StatusCode::Incompatible,
                       "the output '{}' is {}, but head {} of the task needs [{}, H, W, {}]",
                       head.output, shape.to_string(), index, given.batch, channels);

            return AnchorHead{head.output,
                              Detect::Anchor::HeadShape{shape.dim(0), shape.dim(1), shape.dim(2)},
                              head.options, training};
        }
    } // namespace

    Result<AnchorObjective> build_anchor_objective(const Train::ObjectiveContext& given,
                                                   const Config::Value& attributes)
    {
        SA_FAIL_IF(given.task.is_null(), StatusCode::InvalidDescription,
                   "the model file has no 'task' section, which states the anchors and the "
                   "heads an '{}' objective trains",
                   Detect::Anchor::K_ANCHOR_OBJECTIVE_TYPE);

        SA_TRY_ASSIGN(Detect::Anchor::AnchorTask task,
                      Detect::Anchor::read_anchor_task(given.task));
        SA_TRY_ASSIGN(const Detect::Anchor::HeadTrainingOptions training,
                      Detect::Anchor::read_head_training_section(attributes));

        SA_FAIL_IF(given.input_sample.rank() != K_SAMPLE_RANK, StatusCode::InvalidDescription,
                   "the model reads samples of {}; an anchor objective needs [H, W, C]",
                   given.input_sample.to_string());
        SA_FAIL_IF(given.classes != 0 && given.classes != task.num_classes,
                   StatusCode::Incompatible,
                   "the data spans {} classes, but the model's task states {}", given.classes,
                   task.num_classes);

        const Detect::Anchor::NetworkSize network{given.input_sample.dim(0),
                                                  given.input_sample.dim(1)};

        std::vector<AnchorHead> heads;
        SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                        [&]() -> Status
        {
            heads.reserve(task.heads.size());
            for (std::size_t index = 0; index < task.heads.size(); ++index)
            {
                SA_TRY_ASSIGN(AnchorHead head, describe_head(given, task, training, index));
                heads.push_back(std::move(head));
            }

            return Status();
        }));

        return AnchorObjective::make(network, std::move(task.layout), std::move(heads));
    }

    Result<std::unique_ptr<Train::Objective>> make_anchor_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context)
    {
        Result<AnchorObjective> built = build_anchor_objective(given, attributes);
        SA_TRY(SA_ANNOTATE(built.status(), "{}", context));

        return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<Train::Objective>(
                std::make_unique<AnchorObjective>(std::move(built).value()));
        });
    }

} // namespace SushiAI::Vision::Training
