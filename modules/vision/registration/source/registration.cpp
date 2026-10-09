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
 * @brief Implements the vision registrations over the block and MNIST factories.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/registration.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c2psa.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/blocks/sppf.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>

namespace SushiAI::Vision
{
    Status register_nodes(NN::NodeRegistry& registry)
    {
        SA_TRY(registry.add(NN::NodeEntry{"conv", &Blocks::make_conv_node}));
        SA_TRY(registry.add(NN::NodeEntry{"bottleneck", &Blocks::make_bottleneck_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c3k", &Blocks::make_c3k_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c3k2", &Blocks::make_c3k2_node}));
        SA_TRY(registry.add(NN::NodeEntry{"sppf", &Blocks::make_sppf_node}));
        SA_TRY(registry.add(NN::NodeEntry{"c2psa", &Blocks::make_c2psa_node}));
        SA_TRY(registry.add(NN::NodeEntry{"detect", &Blocks::make_detect_node}));

        return Status();
    }

    Status register_datasets(Data::DatasetRegistry& registry)
    {
        return registry.add(Data::DatasetEntry{"mnist", &Datasets::make_mnist});
    }

} // namespace SushiAI::Vision
