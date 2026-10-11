/**************************************************************************/
/* token_dataset.cpp                                                      */
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
 * @file token_dataset.cpp
 * @brief Implements the dataset that reads each batch of windows from a token file.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/token_dataset.hpp>

#include <fstream>
#include <ios>
#include <utility>

#include <SushiBLAS/core/format.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/language/text/token.hpp>

#include "disk.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_DATASET = "TokenDataset";
    } // namespace

    Result<std::unique_ptr<TokenDataset>> TokenDataset::make(
        SushiBLAS::Engine& engine, std::filesystem::path file,
        std::unique_ptr<const Collator> collator, TokenDatasetOptions options)
    {
        SA_FAIL_IF(collator == nullptr, StatusCode::InvalidDescription,
                   "{}: '{}' has no collator", K_DATASET, options.name);
        SA_FAIL_IF(options.batch_size < 1, StatusCode::InvalidDescription,
                   "{}: '{}' has a batch size of {}; it must be positive", K_DATASET,
                   options.name, options.batch_size);

        SA_TRY_ASSIGN(const TokenFileHeader header, read_token_file_header(file));

        return Detail::at_boundary(K_DATASET, [&]() -> Result<std::unique_ptr<TokenDataset>>
        {
            SA_FAIL_IF(header.vocabulary_size > collator->vocabulary_size(),
                       StatusCode::Incompatible,
                       "'{}': {}: the file states a vocabulary of {} tokens; '{}' reads one of {}",
                       Detail::display_of(file), K_DATASET, header.vocabulary_size, options.name,
                       collator->vocabulary_size());

            WindowLayout layout;
            layout.tokens = collator->tokens_per_window();
            layout.stride = options.stride.value_or(collator->window_length());

            const Result<std::uint64_t> counted = count_windows(header.token_count, layout);
            if (!counted.has_value())
                return SA_ANNOTATE(counted.status(), "{}: '{}'", K_DATASET, options.name);

            const std::uint64_t windows = counted.value();
            SA_FAIL_IF(windows < static_cast<std::uint64_t>(options.batch_size),
                       StatusCode::InvalidDescription,
                       "'{}': {}: {} ids hold {} windows of {}, fewer than one batch of {}",
                       Detail::display_of(file), K_DATASET, header.token_count, windows,
                       layout.tokens, options.batch_size);

            return std::unique_ptr<TokenDataset>(
                new TokenDataset(engine, std::move(file), header, std::move(collator), layout,
                                 windows, std::move(options)));
        }, file);
    }

    TokenDataset::TokenDataset(SushiBLAS::Engine& engine, std::filesystem::path file,
                               TokenFileHeader header, std::unique_ptr<const Collator> collator,
                               WindowLayout layout, std::uint64_t windows,
                               TokenDatasetOptions options) noexcept
        : engine_(&engine), file_(std::move(file)), header_(header),
          collator_(std::move(collator)), layout_(layout), windows_(windows),
          options_(std::move(options))
    {

    }

    std::int64_t TokenDataset::batch_size() const noexcept
    {
        return options_.batch_size;
    }

    Shape TokenDataset::sample_shape() const
    {
        return Shape{collator_->window_length()};
    }

    DType TokenDataset::sample_dtype() const noexcept
    {
        return DType::Int32;
    }

    const std::vector<Data::TargetDescription>& TokenDataset::targets() const noexcept
    {
        return collator_->targets();
    }

    std::size_t TokenDataset::batch_count() const noexcept
    {
        return static_cast<std::size_t>(windows_ / static_cast<std::uint64_t>(options_.batch_size));
    }

    std::string_view TokenDataset::name() const noexcept
    {
        return options_.name;
    }

    std::int32_t TokenDataset::vocabulary_size() const noexcept
    {
        return header_.vocabulary_size;
    }

    Status TokenDataset::read(std::size_t index, Data::Batch& batch) const
    {
        SA_FATAL_IF(index >= batch_count(), "{}: '{}' has {} batches; batch {} is outside them",
                    K_DATASET, options_.name, batch_count(), index);

        return Detail::at_boundary(K_DATASET, [&]() -> Status
        {
            const std::size_t rows = static_cast<std::size_t>(options_.batch_size);
            const std::size_t per_window = static_cast<std::size_t>(layout_.tokens);
            std::vector<Text::TokenID> tokens(rows * per_window);

            std::ifstream in(file_, std::ios::binary);
            SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                       "'{}': {}: the file cannot be opened for reading",
                       Detail::display_of(file_), K_DATASET);

            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::uint64_t window = static_cast<std::uint64_t>(index) * rows + row;
                const SushiRuntime::span<Text::TokenID> ids(tokens.data() + row * per_window,
                                                            per_window);

                SA_TRY(SA_ANNOTATE(
                    read_token_range(in, header_, first_token_of(window, layout_), ids), "'{}'",
                    Detail::display_of(file_)));
            }

            SA_TRY(Data::prepare_batch(*engine_, options_.batch_size,
                                       Data::describe_input(*this), collator_->targets(),
                                       batch));
            return collator_->collate(
                SushiRuntime::span<const Text::TokenID>(tokens.data(), tokens.size()), batch);
        }, file_);
    }
} // namespace SushiAI::Language::Datasets
