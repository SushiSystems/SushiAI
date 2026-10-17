/**************************************************************************/
/* generation_task.hpp                                                    */
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
 * @file generation_task.hpp
 * @brief Declares GenerationTask and the reader of the "next_token_generation" task section.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/generation/sampler.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Names the task type of a model whose output is continued token by token. */
    inline constexpr std::string_view K_GENERATION_TASK_TYPE = "next_token_generation";

    /** @brief Holds the tokens a run generates when its task section states no count. */
    inline constexpr std::int64_t K_DEFAULT_GENERATED_TOKENS = 64;

    /** @brief Holds whether a run caches keys and values when its task section does not say. */
    inline constexpr bool K_DEFAULT_GENERATION_CACHE = true;

    /** @brief Holds what a "next_token_generation" task section states. */
    struct GenerationTask
    {
        /** @brief Holds the path of the vocabulary file, as the section spells it. */
        std::string vocabulary;

        /** @brief Holds T, the positions of the window the model was traced for. */
        std::int64_t window = 0;

        /** @brief Counts the tokens to generate; at least one. */
        std::int64_t tokens = K_DEFAULT_GENERATED_TOKENS;

        /** @brief Seeds the generator the sampler draws from. */
        std::uint64_t seed = 0;

        /** @brief Picks each token; never null after a read, greedy when none is stated. */
        std::unique_ptr<Sampler> sampler;

        /** @brief Asks for the key and value cache; the whole window per token when false. */
        bool cache = K_DEFAULT_GENERATION_CACHE;
    };

    /**
     * @brief Reads a task section whose "type" is K_GENERATION_TASK_TYPE.
     * @param task "vocabulary" and "window" are required; "tokens", "seed", "sampler" and
     *        "cache" are optional. Any other key is refused.
     * @param context What to call the section in a diagnostic.
     * @return The task, or InvalidDescription naming the key that does not fit.
     */
    [[nodiscard]] Result<GenerationTask> read_generation_task(const Config::Value& task,
                                                              std::string_view context);

} // namespace SushiAI::Language::Generation
