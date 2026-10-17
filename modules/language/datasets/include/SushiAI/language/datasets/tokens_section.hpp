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
        SushiBLAS::Engine& engine, const Config::Value& section, std::string_view context);

} // namespace SushiAI::Language::Datasets
