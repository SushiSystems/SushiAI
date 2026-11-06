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
 * @file registration.hpp
 * @brief Declares the functions that add the rows of language to the registries.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>

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
    [[nodiscard]] Status register_nodes(NN::NodeRegistry& registry) noexcept;

    /**
     * @brief Adds the tokens row and the prompt row.
     * @return InvalidDescription when @p registry already holds one of the two types.
     */
    [[nodiscard]] Status register_datasets(Data::DatasetRegistry& registry) noexcept;

    /**
     * @brief Adds the next_token row and the expert_balance row.
     * @return InvalidDescription when @p registry already holds one of the two types.
     */
    [[nodiscard]] Status register_objectives(Train::ObjectiveRegistry& registry) noexcept;

} // namespace SushiAI::Language
