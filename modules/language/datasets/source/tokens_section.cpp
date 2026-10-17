/**************************************************************************/
/* tokens_section.cpp                                                     */
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
 * @file tokens_section.cpp
 * @brief Implements make_tokens_dataset over the token file a data block names.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/tokens_section.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include "section.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names the key that holds the path of the token file. */
        constexpr std::string_view K_FILE_KEY = "file";

        /** @brief Names the key that holds the ids between the starts of two windows. */
        constexpr std::string_view K_STRIDE_KEY = "stride";

        /** @brief Names the key that holds the windows of one batch. */
        constexpr std::string_view K_BATCH_SIZE_KEY = "batch_size";

        /** @brief Lists every key a "tokens" data block accepts beside its type. */
        constexpr std::string_view K_TOKENS_KEYS[] = {K_FILE_KEY, Detail::K_WINDOW_KEY,
                                                      K_STRIDE_KEY, K_BATCH_SIZE_KEY};
    } // namespace

    Result<std::unique_ptr<Data::Dataset>> make_tokens_dataset(SushiBLAS::Engine& engine,
                                                               const Config::Value& section,
                                                               std::string_view context)
    {
        const Config::SectionReader reader(section, context);
        SA_TRY(reader.check_object());
        SA_TRY(reader.check_keys(Config::KeySpan(K_TOKENS_KEYS)));
        SA_TRY_ASSIGN(const std::string file, Detail::read_text(reader, K_FILE_KEY));
        SA_TRY_ASSIGN(const std::int64_t window,
                      Detail::read_count(reader, Detail::K_WINDOW_KEY));

        TokenDatasetOptions options;
        if (reader.has(K_STRIDE_KEY))
        {
            SA_TRY_ASSIGN(const std::int64_t stride, Detail::read_count(reader, K_STRIDE_KEY));
            options.stride = stride;
        }

        if (reader.has(K_BATCH_SIZE_KEY))
        {
            SA_TRY_ASSIGN(options.batch_size, Detail::read_count(reader, K_BATCH_SIZE_KEY));
        }

        // The collator states the vocabulary, which only the file's header knows.
        const std::filesystem::path path(file);
        SA_TRY_ASSIGN(const TokenFileHeader header, read_token_file_header(path));
        SA_TRY_ASSIGN(std::unique_ptr<NextTokenCollator> collator,
                      NextTokenCollator::make(window, header.vocabulary_size));
        SA_TRY_ASSIGN(std::unique_ptr<TokenDataset> dataset,
                      TokenDataset::make(engine, path, std::move(collator),
                                         std::move(options)));

        return std::unique_ptr<Data::Dataset>(std::move(dataset));
    }

} // namespace SushiAI::Language::Datasets
