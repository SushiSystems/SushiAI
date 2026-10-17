/**************************************************************************/
/* darknet_import.cpp                                                     */
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
 * @file darknet_import.cpp
 * @brief Implements the Darknet importer over the translation, the recipe and the tensors.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_import.hpp>

#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_recipe.hpp>
#include <SushiAI/vision/importers/darknet_tensors.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the files the importer takes at least: the .cfg. */
            constexpr std::size_t K_FEWEST_FILES = 1;

            /** @brief Holds the files the importer takes at most: the .cfg and the .weights. */
            constexpr std::size_t K_MOST_FILES = 2;

            /**
             * @brief Returns what decoding the heads of @p model needs, as a task.
             * @pre @p model has at least one head, as translate_darknet_model returns it.
             */
            [[nodiscard]] Detect::Anchor::AnchorTask describe_task(const DarknetModel& model)
            {
                Detect::Anchor::AnchorTask task;
                task.num_classes = model.heads.front().options.num_classes;
                task.layout = model.layout;
                task.criterion = model.criterion;
                task.distance_exponent = model.distance_exponent;

                task.heads.reserve(model.heads.size());
                for (const DarknetHead& head : model.heads)
                    task.heads.push_back(Detect::Anchor::AnchorTaskHead{head.output, head.options});

                return task;
            }

            /**
             * @brief Writes the task section of @p model.
             * @return The section; InvalidDescription for a model without a head or for what
             *         write_anchor_task refuses.
             */
            [[nodiscard]] Result<Config::Value> write_task(const DarknetModel& model)
            {
                SA_FAIL_IF(model.heads.empty(), StatusCode::InvalidDescription,
                           "the translated model has no head, so it has no task to state");

                SA_TRY_ASSIGN(const Detect::Anchor::AnchorTask task,
                              SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                           [&]()
                {
                    return describe_task(model);
                }));

                return Detect::Anchor::write_anchor_task(task);
            }

        } // namespace

        Result<IO::ImportedModel> import_darknet_model(const DarknetConfig& config,
                                                       const DarknetModel& model,
                                                       const DarknetWeights* weights)
        {
            SA_TRY_ASSIGN(Config::Value task, write_task(model));
            SA_TRY_ASSIGN(const DarknetTraining training, read_darknet_training(config));
            SA_TRY_ASSIGN(DarknetRecipe recipe, write_darknet_recipe(config, training));

            std::vector<IO::ImportedTensor> tensors;
            if (weights != nullptr)
            {
                SA_TRY_ASSIGN(tensors, list_darknet_tensors(*weights, model.convolutions,
                                                            model.targets));
            }

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                IO::ImportedModel imported;
                imported.model = model.spec;
                imported.model.task = std::move(task);
                imported.weights = std::move(tensors);
                imported.training = std::move(recipe.training);
                imported.notes = std::move(recipe.notes);
                return imported;
            });
        }

        Result<IO::ImportedModel> import_darknet_files(
            SushiRuntime::span<const std::filesystem::path> files)
        {
            SA_FAIL_IF(files.size() < K_FEWEST_FILES || files.size() > K_MOST_FILES,
                       StatusCode::InvalidDescription,
                       "the '{}' format takes a .cfg and optionally a .weights, but {} files "
                       "were given",
                       K_DARKNET_FORMAT, files.size());

            SA_TRY_ASSIGN(const DarknetConfig config, load_darknet_config(files[0]));
            SA_TRY_ASSIGN(const DarknetModel model,
                          translate_darknet_model(config, DarknetTranslation{}));
            if (files.size() == K_FEWEST_FILES)
                return import_darknet_model(config, model, nullptr);

            SA_TRY_ASSIGN(const DarknetWeights weights,
                          read_darknet_weights(files[1], model.convolutions));
            return import_darknet_model(config, model, &weights);
        }

    } // namespace Vision::Importers
} // namespace SushiAI
