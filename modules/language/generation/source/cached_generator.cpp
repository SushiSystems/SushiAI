/**************************************************************************/
/* cached_generator.cpp                                                   */
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
 * @file cached_generator.cpp
 * @brief Implements CachedGenerator over a carrying step of one position.
 * @author Mustafa Garip
 */

#include <SushiAI/language/generation/cached_generator.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include "request.hpp"
#include "step_leaf.hpp"

namespace SushiAI::Language::Generation
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_GENERATOR = Detail::K_CACHED_GENERATOR;

        /** @brief Names the rank of the [1, 1, V] logits. */
        constexpr std::int32_t K_LOGITS_RANK = 3;

        /** @brief Counts the leaves a run feeds whatever places its token: the id, the mask. */
        constexpr std::size_t K_OWN_LEAVES = 2;

    } // namespace

    Result<std::unique_ptr<CachedGenerator>> CachedGenerator::make(
        SushiBLAS::Engine& engine, Train::EvaluationStep& step, std::string_view output,
        const CachedStepLeaves& leaves)
    {
        SA_FAIL_IF(!step.carries_state(), StatusCode::InvalidDescription,
                   "{}: the step keeps nothing between runs; it reads a step traced with "
                   "Train::build_incremental_graph",
                   K_GENERATOR);

        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, step.output(output));
        SA_FAIL_IF(logits->rank != K_LOGITS_RANK || logits->shape[0] != 1 ||
                       logits->shape[1] != 1 || logits->dtype != DType::Float32,
                   StatusCode::InvalidDescription,
                   "{}: the output '{}' has rank {} and the element type {}; it reads "
                   "float32 logits of [1, 1, V]",
                   K_GENERATOR, output, logits->rank, dtype_name(logits->dtype));

        SA_TRY(Detail::fitting_leaf(step, Train::K_MODEL_INPUT_NAME, 1, DType::Int32));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const mask,
                      Detail::fitting_leaf(step, leaves.mask, Detail::K_ANY_COLUMNS,
                                           DType::Float32));
        SA_TRY_ASSIGN(std::unique_ptr<Detail::PositionFeed> positions,
                      Detail::make_position_feed(engine, step, leaves, mask->shape[1]));
        SA_FAIL_IF(step.input_names().size() != K_OWN_LEAVES + positions->feeds().size(),
                   StatusCode::InvalidDescription,
                   "{}: the step is fed {} leaves; it fills '{}', '{}' and the {} that place "
                   "a token",
                   K_GENERATOR, step.input_names().size(), Train::K_MODEL_INPUT_NAME,
                   leaves.mask, positions->feeds().size());

        Feeds feeds;
        SA_TRY(Data::prepare_batch(engine, 1, Data::InputDescription{Shape{1}, DType::Int32},
                                   {}, feeds.id));
        SA_TRY(Data::prepare_batch(
            engine, 1, Data::InputDescription{Shape{mask->shape[1]}, DType::Float32}, {},
            feeds.mask));

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<CachedGenerator>(new CachedGenerator(
                step, *logits, std::move(feeds), std::move(positions), leaves));
        });
    }

    CachedGenerator::CachedGenerator(Train::EvaluationStep& step,
                                     const SushiBLAS::Tensor& logits, Feeds feeds,
                                     std::unique_ptr<Detail::PositionFeed> positions,
                                     const CachedStepLeaves& leaves)
        : step_(&step),
          logits_(&logits),
          feeds_(std::move(feeds)),
          positions_(std::move(positions)),
          mask_name_(leaves.mask),
          masked_score_(static_cast<float>(leaves.masked_score))
    {
        fed_.reserve(K_OWN_LEAVES + positions_->feeds().size());
        fed_.push_back(Train::NamedFeed{Train::K_MODEL_INPUT_NAME, &feeds_.id.inputs});
        fed_.push_back(Train::NamedFeed{mask_name_, &feeds_.mask.inputs});
        for (const Train::NamedFeed& feed : positions_->feeds())
            fed_.push_back(feed);
    }

    CachedGenerator::~CachedGenerator() = default;

    std::int64_t CachedGenerator::window() const noexcept
    {
        return feeds_.mask.inputs.shape[1];
    }

    std::int64_t CachedGenerator::vocabulary() const noexcept
    {
        return logits_->shape[2];
    }

    Status CachedGenerator::advance(Text::TokenID id)
    {
        // A cache row is filled once its position is fed, and a ring of T rows stays full.
        const std::size_t length = static_cast<std::size_t>(window());
        const std::size_t filled =
            std::min(static_cast<std::size_t>(next_position_) + 1, length);
        float* const mask = feeds_.mask.inputs.data_as<float>();
        std::fill(mask, mask + filled, 0.0f);
        std::fill(mask + filled, mask + length, masked_score_);

        feeds_.id.inputs.data_as<Text::TokenID>()[0] = id;
        positions_->place(next_position_);
        ++next_position_;

        return step_->run_by_name(
            SushiRuntime::span<const Train::NamedFeed>(fed_.data(), fed_.size()));
    }

    Status CachedGenerator::restart(SushiRuntime::span<const Text::TokenID> context)
    {
        SA_TRY(step_->reset());
        next_position_ = 0;
        for (const Text::TokenID id : context)
            SA_TRY(advance(id));

        return Status();
    }

    Status CachedGenerator::generate(SushiRuntime::span<const Text::TokenID> prompt,
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
        SA_TRY(restart(SushiRuntime::span<const Text::TokenID>(context.data(), context.size())));

        for (std::int64_t made = 0; made < count; ++made)
        {
            // The last run fed the last id of the context, so the logits score what follows.
            const LogitSpan row(logits_->data_as<float>(), scores);
            SA_TRY(Detail::check_finite(row, context.size() - 1, K_GENERATOR));

            const Text::TokenID token = sampler.sample(row, random);
            generated.push_back(token);
            if (made + 1 == count)
                break;

            const bool slides = context.size() == length;
            if (slides)
                context.erase(context.begin());

            context.push_back(token);
            if (slides && !positions_->keeps_rows_on_slide())
            {
                SA_TRY(restart(
                    SushiRuntime::span<const Text::TokenID>(context.data(), context.size())));
                continue;
            }

            SA_TRY(advance(token));
        }

        return Status();
    }

} // namespace SushiAI::Language::Generation
