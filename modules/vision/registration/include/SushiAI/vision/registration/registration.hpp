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
 * @brief Declares the functions that fill the registries with the rows of vision.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief The registry of node and layer rows; declared in node_registry.hpp. */
        class NodeRegistry;
    } // namespace NN

    namespace Data
    {
        /** @brief The registry of dataset rows; declared in dataset_registry.hpp. */
        class DatasetRegistry;
    } // namespace Data

    namespace Train
    {
        /** @brief The registry of objective rows; declared in objective_registry.hpp. */
        class ObjectiveRegistry;
    } // namespace Train

    namespace IO
    {
        /** @brief The registry of importer rows; declared in importer_registry.hpp. */
        class ImporterRegistry;
    } // namespace IO
} // namespace SushiAI

namespace SushiAI::Vision
{
    /**
     * @brief Adds the block node types to @p registry, in the order conv, bottleneck, c3k,
     *        c3k2, sppf, c2psa, detect and segment_head.
     * @param registry The registry that receives the eight rows.
     * @return Success, or the first failure of NodeRegistry::add().
     */
    [[nodiscard]] Status register_nodes(NN::NodeRegistry& registry) noexcept;

    /**
     * @brief Adds the "mnist" and "reader" dataset rows to @p registry, in that order.
     * @param registry The registry that receives the rows; "reader" builds the section
     *        Datasets::make_reader_dataset reads.
     * @return Success, or the first failure of DatasetRegistry::add().
     */
    [[nodiscard]] Status register_datasets(Data::DatasetRegistry& registry) noexcept;

    /**
     * @brief Adds the "anchor", "pixel_cross_entropy" and "dice" objective rows to
     *        @p registry, in that order.
     * @param registry The registry that receives the three rows.
     * @return Success, or the first failure of ObjectiveRegistry::add().
     */
    [[nodiscard]] Status register_objectives(Train::ObjectiveRegistry& registry) noexcept;

    /**
     * @brief Adds the "darknet" importer row to @p registry.
     * @param registry The registry that receives the row.
     * @return Success, or the first failure of ImporterRegistry::add().
     */
    [[nodiscard]] Status register_importers(IO::ImporterRegistry& registry) noexcept;

} // namespace SushiAI::Vision
