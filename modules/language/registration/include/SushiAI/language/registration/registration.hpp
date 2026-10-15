/**************************************************************************/
/* registration.hpp                                                       */
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
 * @file registration.hpp
 * @brief Declares the functions that add the rows of language to the registries.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        class NodeRegistry;
    } // namespace NN

    namespace Data
    {
        class DatasetRegistry;
    } // namespace Data

    namespace Train
    {
        class ObjectiveRegistry;
    } // namespace Train
} // namespace SushiAI

namespace SushiAI::Language
{
    /**
     * @brief Adds position_embedding, causal_self_attention, feed_forward, transformer_block
     *        and moe_feed_forward as node rows, then language_head as a linked node row.
     * @return InvalidDescription when @p registry already holds one of the six types.
     */
    [[nodiscard]] Status register_nodes(NN::NodeRegistry& registry);

    /**
     * @brief Adds the tokens row and the prompt row.
     * @return InvalidDescription when @p registry already holds one of the two types.
     */
    [[nodiscard]] Status register_datasets(Data::DatasetRegistry& registry);

    /**
     * @brief Adds the next_token row and the expert_balance row.
     * @return InvalidDescription when @p registry already holds one of the two types.
     */
    [[nodiscard]] Status register_objectives(Train::ObjectiveRegistry& registry);

} // namespace SushiAI::Language
