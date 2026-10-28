/**************************************************************************/
/* registration.cpp                                                       */
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
 * @file registration.cpp
 * @brief Implements the vision registrations over the factories of the vision modules.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/registration/registration.hpp>

#include <SushiAI/config/train_config.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c2psa.hpp>
#include <SushiAI/vision/blocks/c2f.hpp>
#include <SushiAI/vision/blocks/c3.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/blocks/segment_head.hpp>
#include <SushiAI/vision/blocks/sppf.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>

namespace SushiAI::Vision
{
    Status register_nodes(NN::NodeRegistry& registry) noexcept
    {
        SA_TRY(registry.add(NN::NodeEntry{"conv", &Blocks::make_conv_node}));
        SA_TRY(registry.add(NN::NodeEntry{"bottleneck", &Blocks::make_bottleneck_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c3k", &Blocks::make_c3k_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c3k2", &Blocks::make_c3k2_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c2f", &Blocks::make_c2f_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c3", &Blocks::make_c3_node}));
        SA_TRY(registry.add(NN::NodeEntry{"sppf", &Blocks::make_sppf_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c2psa", &Blocks::make_c2psa_node}));
        SA_TRY(registry.add(NN::NodeEntry{"detect", &Blocks::make_detect_node}));
        SA_TRY(registry.add(NN::NodeEntry{"segment_head", &Blocks::make_segment_head_node}));

        return Status();
    }

    Status register_datasets(Data::DatasetRegistry& registry) noexcept
    {
        SA_TRY(registry.add(Data::DatasetEntry{"mnist", &Datasets::make_mnist}));
        SA_TRY(registry.add(Data::DatasetEntry{Config::K_READER_DATA_TYPE,
                                               &Datasets::make_reader_dataset}));

        return Status();
    }

    Status register_objectives(Train::ObjectiveRegistry& registry) noexcept
    {
        SA_TRY(registry.add(Train::ObjectiveEntry{Detect::Anchor::K_ANCHOR_OBJECTIVE_TYPE,
                                                  &Training::make_anchor_objective}));
        SA_TRY(registry.add(Train::ObjectiveEntry{Training::K_FREE_OBJECTIVE_TYPE,
                                                  &Training::make_free_objective}));
        SA_TRY(registry.add(
            Train::ObjectiveEntry{Training::K_PIXEL_CROSS_ENTROPY_TYPE,
                                  &Training::make_pixel_cross_entropy_objective}));
        SA_TRY(registry.add(Train::ObjectiveEntry{Training::K_DICE_TYPE,
                                                  &Training::make_dice_objective}));

        return Status();
    }

    Status register_importers(IO::ImporterRegistry& registry) noexcept
    {
        return registry.add(IO::ImporterEntry{Importers::K_DARKNET_FORMAT,
                                              &Importers::import_darknet_files});
    }

} // namespace SushiAI::Vision
