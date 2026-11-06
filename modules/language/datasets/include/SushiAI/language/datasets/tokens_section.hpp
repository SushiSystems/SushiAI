/**************************************************************************/
/* tokens_section.hpp                                                     */
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
 * @file tokens_section.hpp
 * @brief Declares the factory that builds a next-token dataset from a "tokens" data block.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI::Language::Datasets
{
    /** @brief Names the type of the data block make_tokens_dataset() reads. */
    inline constexpr std::string_view K_TOKENS_DATA_TYPE = "tokens";

    /**
     * @brief Builds a TokenDataset with a NextTokenCollator from the keys of a data block.
     * @param engine Allocates the tensors of each batch; it must outlive the dataset.
     * @param section "file" and "window" are required; "stride" is one window when absent
     *        and "batch_size" is that of TokenDatasetOptions.
     * @return The dataset, or InvalidDescription for a key that does not fit, or the failure
     *         of the file's header or of TokenDataset::make.
     */
    [[nodiscard]] Result<std::unique_ptr<Data::Dataset>> make_tokens_dataset(
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context) noexcept;

} // namespace SushiAI::Language::Datasets
