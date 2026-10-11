/**************************************************************************/
/* window_generator.cpp                                                   */
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
 * @file window_generator.cpp
 * @brief Implements WindowGenerator over an evaluation step of a fixed window.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/window_generator.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include "request.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_GENERATOR = "WindowGenerator";

        /** @brief Names the rank of the [1, T, V] logits. */
        constexpr std::int32_t K_LOGITS_RANK = 3;

    } // namespace

    Result<std::unique_ptr<WindowGenerator>> WindowGenerator::make(SushiBLAS::Engine& engine,
                                                                  Train::EvaluationStep& step,
                                                                  std::string_view output)
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, step.output(output));
        SA_FAIL_IF(logits->rank != K_LOGITS_RANK || logits->shape[0] != 1 ||
                       logits->dtype != DType::Float32,
                   StatusCode::InvalidDescription,
                   "{}: the output '{}' has rank {} and the element type {}; it reads "
                   "float32 logits of [1, T, V]",
                   K_GENERATOR, output, logits->rank, dtype_name(logits->dtype));

        Data::Batch feed;
        SA_TRY(Data::prepare_batch(engine, 1,
                                   Data::InputDescription{Shape{logits->shape[1]}, DType::Int32},
                                   {}, feed));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<WindowGenerator>(
                new WindowGenerator(step, *logits, std::move(feed)));
        });
    }

    WindowGenerator::WindowGenerator(Train::EvaluationStep& step,
                                     const SushiBLAS::Tensor& logits, Data::Batch feed) noexcept
        : step_(&step), logits_(&logits), feed_(std::move(feed))
    {

    }

    std::int64_t WindowGenerator::window() const noexcept
    {
        return logits_->shape[1];
    }

    std::int64_t WindowGenerator::vocabulary() const noexcept
    {
        return logits_->shape[2];
    }

    Status WindowGenerator::generate(SushiRuntime::span<const Text::TokenID> prompt,
                                     std::int64_t count, const Sampler& sampler,
                                     RandomGenerator& random,
                                     std::vector<Text::TokenID>& generated)
    {
        SA_TRY(Detail::check_request(prompt, count, vocabulary(), K_GENERATOR));

        const std::size_t length = static_cast<std::size_t>(window());
        const std::size_t scores = static_cast<std::size_t>(vocabulary());
        const std::size_t kept = std::min(prompt.size(), length);
        std::vector<Text::TokenID> context(prompt.end() - static_cast<std::ptrdiff_t>(kept),
                                           prompt.end());

        Text::TokenID* const fed = feed_.inputs.data_as<Text::TokenID>();
        for (std::int64_t made = 0; made < count; ++made)
        {
            // A position behind the context holds 0, which no earlier position reads.
            std::fill(fed, fed + length, Text::TokenID{0});
            std::copy(context.begin(), context.end(), fed);
            SA_TRY(step_->run(feed_.inputs));

            const std::size_t last = context.size() - 1;
            const LogitSpan row(logits_->data_as<float>() + last * scores, scores);
            SA_TRY(Detail::check_finite(row, last, K_GENERATOR));

            const Text::TokenID token = sampler.sample(row, random);
            generated.push_back(token);
            if (context.size() == length)
                context.erase(context.begin());

            context.push_back(token);
        }

        return Status();
    }

} // namespace SushiAI::Language::Generation
