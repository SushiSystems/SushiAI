/**************************************************************************/
/* prompt_dataset.cpp                                                     */
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
 * @file prompt_dataset.cpp
 * @brief Implements PromptDataset and the factory that encodes the text of a data block.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/prompt_dataset.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>
#include "disk.hpp"
#include "section.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_DATASET = "PromptDataset";

        /** @brief Names the key that holds the text a generation starts from. */
        constexpr std::string_view K_TEXT_KEY = "text";

        /** @brief Names the key that holds the path of the vocabulary file. */
        constexpr std::string_view K_VOCABULARY_KEY = "vocabulary";

        /** @brief Lists every key a "prompt" data block accepts beside its type. */
        constexpr std::string_view K_PROMPT_KEYS[] = {K_TEXT_KEY, K_VOCABULARY_KEY,
                                                      Detail::K_WINDOW_KEY};
    } // namespace

    Result<std::unique_ptr<PromptDataset>> PromptDataset::make(
        SushiBLAS::Engine& engine, const std::vector<Text::TokenID>& ids, std::int64_t window)
    {
        SA_FAIL_IF(window < 1 || window > K_MAX_WINDOW_TOKENS, StatusCode::InvalidDescription,
                   "{}: a window of {} positions; it has 1 to {}", K_DATASET, window,
                   K_MAX_WINDOW_TOKENS);
        SA_FAIL_IF(ids.empty(), StatusCode::InvalidDescription,
                   "{}: the prompt holds no id, so there is nothing to continue", K_DATASET);

        for (const Text::TokenID id : ids)
        {
            SA_FAIL_IF(id < 0, StatusCode::InvalidDescription,
                       "{}: the prompt holds the id {}; an id is zero or above", K_DATASET, id);
        }

        return Detail::at_boundary(K_DATASET, [&]() -> Result<std::unique_ptr<PromptDataset>>
        {
            const std::size_t kept = std::min(ids.size(), static_cast<std::size_t>(window));
            std::vector<Text::TokenID> last(ids.end() - static_cast<std::ptrdiff_t>(kept),
                                            ids.end());

            return std::unique_ptr<PromptDataset>(
                new PromptDataset(engine, std::move(last), window));
        });
    }

    PromptDataset::PromptDataset(SushiBLAS::Engine& engine, std::vector<Text::TokenID> kept,
                                 std::int64_t window)
        : engine_(&engine), kept_(std::move(kept)), window_(window),
          targets_{Data::TargetDescription{std::string(K_ROLE_PROMPT_LENGTH), Shape{1},
                                           DType::Int32, 0}}
    {

    }

    std::int64_t PromptDataset::batch_size() const noexcept
    {
        return 1;
    }

    Shape PromptDataset::sample_shape() const
    {
        return Shape{window_};
    }

    DType PromptDataset::sample_dtype() const noexcept
    {
        return DType::Int32;
    }

    const std::vector<Data::TargetDescription>& PromptDataset::targets() const noexcept
    {
        return targets_;
    }

    std::size_t PromptDataset::batch_count() const noexcept
    {
        return 1;
    }

    std::string_view PromptDataset::name() const noexcept
    {
        return K_PROMPT_DATA_TYPE;
    }

    Status PromptDataset::read(std::size_t index, Data::Batch& batch) const
    {
        SA_FATAL_IF(index != 0, "{}: it has one batch; batch {} is outside it", K_DATASET, index);

        return Detail::at_boundary(K_DATASET, [&]() -> Status
        {
            SA_TRY(Data::prepare_batch(*engine_, batch_size(), Data::describe_input(*this),
                                       targets_, batch));

            Text::TokenID* const row = batch.inputs.data_as<Text::TokenID>();
            std::fill(row, row + window_, Text::TokenID{0});
            std::copy(kept_.begin(), kept_.end(), row);

            batch.targets[0].values.data_as<std::int32_t>()[0] =
                static_cast<std::int32_t>(kept_.size());
            return Status();
        });
    }

    Status PromptDataset::read_rows(std::size_t index, Data::RowRange rows,
                                    Data::Batch& batch) const
    {
        Data::check_rows(*this, index, rows);
        return read(index, batch);
    }

    Result<std::unique_ptr<Data::Dataset>> make_prompt_dataset(SushiBLAS::Engine& engine,
                                                               const Config::Value& section,
                                                               std::string_view context)
    {
        const Config::SectionReader reader(section, context);
        SA_TRY(reader.check_object());
        SA_TRY(reader.check_keys(Config::KeySpan(K_PROMPT_KEYS)));
        SA_TRY_ASSIGN(const std::string text, Detail::read_text(reader, K_TEXT_KEY));
        SA_TRY_ASSIGN(const std::string file, Detail::read_text(reader, K_VOCABULARY_KEY));
        SA_TRY_ASSIGN(const std::int64_t window,
                      Detail::read_count(reader, Detail::K_WINDOW_KEY));

        SA_TRY_ASSIGN(const Text::Vocabulary vocabulary,
                      Text::read_vocabulary_file(std::filesystem::path(file)));
        std::vector<Text::TokenID> ids;
        SA_TRY(SA_ANNOTATE(Text::encode_text(vocabulary, text, ids), "{}", context));

        SA_TRY_ASSIGN(std::unique_ptr<PromptDataset> dataset,
                      PromptDataset::make(engine, ids, window));

        return std::unique_ptr<Data::Dataset>(std::move(dataset));
    }

} // namespace SushiAI::Language::Datasets
