/**************************************************************************/
/* generation_task.cpp                                                    */
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
 * @file generation_task.cpp
 * @brief Implements the reader of the "next_token_generation" task section.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/generation_task.hpp>

#include <utility>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/sampler_section.hpp>
#include "section.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names the key that holds the task's type. */
        constexpr std::string_view K_TYPE_KEY = "type";

        /** @brief Names the key that holds the path of the vocabulary file. */
        constexpr std::string_view K_VOCABULARY_KEY = "vocabulary";

        /** @brief Names the key that holds the positions of the model's window. */
        constexpr std::string_view K_WINDOW_KEY = "window";

        /** @brief Names the key that holds the tokens to generate. */
        constexpr std::string_view K_TOKENS_KEY = "tokens";

        /** @brief Names the key that holds the seed of the generator. */
        constexpr std::string_view K_SEED_KEY = "seed";

        /** @brief Names the key that holds the sampler's section. */
        constexpr std::string_view K_SAMPLER_KEY = "sampler";

        /** @brief Names the key that says whether keys and values are cached. */
        constexpr std::string_view K_CACHE_KEY = "cache";

        /** @brief Lists every key the task section accepts. */
        constexpr std::string_view K_TASK_KEYS[] = {K_TYPE_KEY,   K_VOCABULARY_KEY, K_WINDOW_KEY,
                                                    K_TOKENS_KEY, K_SEED_KEY,       K_SAMPLER_KEY,
                                                    K_CACHE_KEY};
    } // namespace

    Result<GenerationTask> read_generation_task(const Config::Value& task,
                                                std::string_view context)
    {
        const Config::SectionReader reader(task, context);
        SA_TRY(reader.check_object());
        SA_TRY(reader.check_keys(Config::KeySpan(K_TASK_KEYS)));
        SA_TRY_ASSIGN(const std::string type, Detail::read_text(reader, K_TYPE_KEY));
        SA_FAIL_IF(type != K_GENERATION_TASK_TYPE, StatusCode::InvalidDescription,
                   "{} has the type '{}'; this reader takes '{}'", context, type,
                   K_GENERATION_TASK_TYPE);

        GenerationTask read;
        SA_TRY_ASSIGN(read.vocabulary, Detail::read_text(reader, K_VOCABULARY_KEY));
        SA_TRY_ASSIGN(read.window, Detail::read_whole(reader, K_WINDOW_KEY));
        SA_TRY_ASSIGN(read.tokens,
                      Detail::read_whole_or(reader, K_TOKENS_KEY, K_DEFAULT_GENERATED_TOKENS));
        SA_TRY_ASSIGN(const std::int64_t seed, Detail::read_whole_or(reader, K_SEED_KEY, 0));
        SA_TRY_ASSIGN(read.cache,
                      Detail::read_flag_or(reader, K_CACHE_KEY, K_DEFAULT_GENERATION_CACHE));

        SA_FAIL_IF(read.vocabulary.empty(), StatusCode::InvalidDescription,
                   "{} '{}' is an empty string; it names the vocabulary file", context,
                   K_VOCABULARY_KEY);
        SA_FAIL_IF(read.window < 1, StatusCode::InvalidDescription,
                   "{} '{}' must be positive, but it is {}", context, K_WINDOW_KEY, read.window);
        SA_FAIL_IF(read.tokens < 1, StatusCode::InvalidDescription,
                   "{} '{}' must be positive, but it is {}", context, K_TOKENS_KEY, read.tokens);
        SA_FAIL_IF(seed < 0, StatusCode::InvalidDescription,
                   "{} '{}' must be zero or above, but it is {}", context, K_SEED_KEY, seed);
        read.seed = static_cast<std::uint64_t>(seed);

        // A section without a sampler is read as one that states the greedy sampler.
        const Config::Value greedy = Config::Value::object(
            {std::string(K_TYPE_KEY)}, {Config::Value::string(std::string(K_GREEDY_SAMPLER))});
        const std::string sampler_context =
            std::string(context) + " '" + std::string(K_SAMPLER_KEY) + "'";
        SA_TRY_ASSIGN(read.sampler,
                      make_sampler(task.at_or(K_SAMPLER_KEY, greedy), sampler_context));
        return read;
    }

} // namespace SushiAI::Language::Generation
