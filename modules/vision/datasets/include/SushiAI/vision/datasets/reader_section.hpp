/**************************************************************************/
/* reader_section.hpp                                                     */
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
 * @file reader_section.hpp
 * @brief Declares the factory that builds a reader dataset from a data section.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>

#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Builds a ReaderDataset from a section that names a reader and a task.
     * @param section Holds the keys README.md, section reader_section.hpp, lists.
     * @return The dataset, never null; InvalidDescription for a reader or a task the tables
     *         lack, each listed, or a value out of range; or the reader's failure.
     *         The failure of a Config accessor for a key absent or of another kind.
     * @pre @p engine outlives the dataset.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_reader_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context) noexcept;
} // namespace SushiAI::Vision::Datasets
