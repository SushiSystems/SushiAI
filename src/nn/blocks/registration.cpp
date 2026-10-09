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
 * @brief Implements register_blocks over the factories of the five block headers.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/blocks/registration.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/blocks/bottleneck.hpp>
#include <SushiAI/nn/blocks/c3k.hpp>
#include <SushiAI/nn/blocks/c3k2.hpp>
#include <SushiAI/nn/blocks/conv_norm.hpp>
#include <SushiAI/nn/blocks/sppf.hpp>
#include <SushiAI/nn/node_registry.hpp>

namespace SushiAI
{
    namespace NN
    {
        Status register_blocks(NodeRegistry& registry)
        {
            SA_TRY(registry.add(NodeEntry{"conv", &make_conv_node}));
            SA_TRY(registry.add(NodeEntry{"bottleneck", &make_bottleneck_node}));
            SA_TRY(registry.add(NodeEntry{"c3k", &make_c3k_node}));
            SA_TRY(registry.add(NodeEntry{"c3k2", &make_c3k2_node}));
            SA_TRY(registry.add(NodeEntry{"sppf", &make_sppf_node}));

            return Status();
        }

    } // namespace NN
} // namespace SushiAI
