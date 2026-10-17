/**************************************************************************/
/* cached_generator.hpp                                                   */
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
 * @file cached_generator.hpp
 * @brief Declares CachedGenerator, the generation loop that runs one position per token over
 *        a step that carries the keys and values of the positions before it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation
{
    namespace Detail
    {
        class PositionFeed;
    } // namespace Detail

    /**
     * @brief Fills the cosines and the sines a step is fed for one position.
     * @param position Counted from the first token fed since the last reset of the step.
     * @param cosines One element per column of the cosine leaf; @p sines has as many.
     */
    using AngleRows = void (*)(std::int64_t position, SushiRuntime::span<float> cosines,
                               SushiRuntime::span<float> sines) noexcept;

    /** @brief Names what a step whose cache is a ring is fed in place of a position. */
    struct RingStepLeaves
    {
        /** @brief Names the [1, 1] int32 leaf that holds the position modulo T. */
        std::string_view slot;

        /** @brief Names the [1, n] float32 leaf that holds the cosines of the position. */
        std::string_view cosine;

        /** @brief Names the [1, n] float32 leaf that holds the sines of the position. */
        std::string_view sine;

        /** @brief Fills both leaves; a ring without it is refused. */
        AngleRows angles = nullptr;
    };

    /** @brief Names the leaves a one-position step is fed beside the id of its token. */
    struct CachedStepLeaves
    {
        /** @brief Names the [1, 1] int32 leaf that holds the position of the token. */
        std::string_view position;

        /** @brief Names the [1, T] float32 leaf added to the scores of the position. */
        std::string_view mask;

        /** @brief Holds what the mask is at a cache row no fed token fills; 0 at the others. */
        double masked_score = 0.0;

        /** @brief Names the leaves of a step that is fed a cache slot and no position. */
        RingStepLeaves ring{};
    };

    /**
     * @brief Continues a prompt one token at a time, with one run of one position per token.
     *
     * A step fed a position resets and feeds the T kept ids again when the window slides; a
     * step fed a cache slot does not. See modules/language/generation/README.md.
     */
    class CachedGenerator final : public Generator
    {
        public:
            /**
             * @brief Binds a carrying step traced for one [1, 1] int32 id, the mask and
             *        either the position or the ring of @p leaves, and [1, 1, V] logits.
             * @param engine Allocates the feeds; it, @p step and its pool outlive this object.
             * @param output The name @p step publishes the logits under.
             * @return The generator, or InvalidDescription when the step carries no state or
             *         a leaf or the output is absent or does not fit.
             */
            [[nodiscard]] static Result<std::unique_ptr<CachedGenerator>> make(
                SushiBLAS::Engine& engine, Train::EvaluationStep& step, std::string_view output,
                const CachedStepLeaves& leaves);

            CachedGenerator(const CachedGenerator&) = delete;
            CachedGenerator& operator=(const CachedGenerator&) = delete;
            CachedGenerator(CachedGenerator&&) = delete;
            CachedGenerator& operator=(CachedGenerator&&) = delete;

            /** @brief Releases the feeds; the step is not owned. */
            ~CachedGenerator() override;

            /** @brief Returns T, the positions of the window, which is the rows of a cache. */
            [[nodiscard]] std::int64_t window() const noexcept override;

            /** @brief Returns V, the tokens a position scores. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept override;

            /**
             * @brief Resets the caches, feeds the last T ids of @p prompt one at a time, then
             *        appends @p count sampled tokens to @p generated.
             * @param prompt At least one id, each in [0, V); the last T are the context.
             * @return Success, InvalidDescription for a prompt or a count that does not fit
             *         or a logit that is not finite, or the failure of a run of the step.
             */
            [[nodiscard]] Status generate(SushiRuntime::span<const Text::TokenID> prompt,
                                          std::int64_t count, const Sampler& sampler,
                                          RandomGenerator& random,
                                          std::vector<Text::TokenID>& generated) override;

        private:
            /** @brief Holds the two tensors every run feeds, whatever places its token. */
            struct Feeds
            {
                Data::Batch id;   /**< The [1, 1] int32 id of the token. */
                Data::Batch mask; /**< The [1, T] float32 addend of the scores. */
            };

            /** @brief Takes the parts make() built. */
            CachedGenerator(Train::EvaluationStep& step, const SushiBLAS::Tensor& logits,
                            Feeds feeds, std::unique_ptr<Detail::PositionFeed> positions,
                            const CachedStepLeaves& leaves);

            /**
             * @brief Resets the caches and runs the step once per id of @p context, in order.
             * @pre @p context holds 1 to T ids.
             */
            [[nodiscard]] Status restart(SushiRuntime::span<const Text::TokenID> context);

            /** @brief Runs the step for @p id at the position after the last one it ran. */
            [[nodiscard]] Status advance(Text::TokenID id);

            /// @brief Runs one position; not owned.
            Train::EvaluationStep* step_;

            /// @brief Views the [1, 1, V] logits the step's pool owns.
            const SushiBLAS::Tensor* logits_;

            /// @brief Holds the id and the mask each run feeds.
            Feeds feeds_;

            /// @brief Fills the leaves that place a token, and says what a slide costs.
            std::unique_ptr<Detail::PositionFeed> positions_;

            /// @brief Names the mask leaf of the step.
            std::string mask_name_;

            /// @brief Lists every leaf a run feeds; it points into this object.
            std::vector<Train::NamedFeed> fed_;

            /// @brief Holds the mask of a cache row no fed token fills.
            float masked_score_;

            /// @brief Counts the tokens fed since the last reset: the next position.
            std::int64_t next_position_ = 0;
    };

} // namespace SushiAI::Language::Generation
