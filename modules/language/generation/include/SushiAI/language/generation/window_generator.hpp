/**************************************************************************/
/* window_generator.hpp                                                   */
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
 * @file window_generator.hpp
 * @brief Declares WindowGenerator, the generation loop that runs the whole window per token.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
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
    /**
     * @brief Continues a prompt one token at a time, with one forward pass of the whole
     *        [1, T] window per token.
     *
     * It is the reference a cached path is compared with, and it is right for a model
     * whose position t reads positions 0 to t only.
     */
    class WindowGenerator final : public Generator
    {
        public:
            /**
             * @brief Binds a step traced for [1, T] int32 ids and one [1, T, V] float32 output.
             * @param engine Allocates the window; it, @p step and its pool outlive this object.
             * @param output The name @p step publishes the logits under.
             * @return The generator, or InvalidDescription when the output is absent or is
             *         not [1, T, V] float32, or the failure of the allocation.
             */
            [[nodiscard]] static Result<std::unique_ptr<WindowGenerator>> make(
                SushiBLAS::Engine& engine, Train::EvaluationStep& step, std::string_view output) noexcept;

            WindowGenerator(const WindowGenerator&) = delete;
            WindowGenerator& operator=(const WindowGenerator&) = delete;
            WindowGenerator(WindowGenerator&&) = delete;
            WindowGenerator& operator=(WindowGenerator&&) = delete;
            ~WindowGenerator() override = default;

            /** @brief Returns T, the positions of the window. */
            [[nodiscard]] std::int64_t window() const noexcept override;

            /** @brief Returns V, the tokens a position scores. */
            [[nodiscard]] std::int64_t vocabulary() const noexcept override;

            /**
             * @brief Appends @p count tokens to @p generated, each sampled from the logits at
             *        the last position the context fills.
             * @param prompt At least one id, each in [0, V); the last T are the context.
             * @return Success, InvalidDescription for a prompt or a count that does not fit
             *         or a logit that is not finite, or the failure of a run of the step.
             */
            [[nodiscard]] Status generate(SushiRuntime::span<const Text::TokenID> prompt,
                                          std::int64_t count, const Sampler& sampler,
                                          RandomGenerator& random,
                                          std::vector<Text::TokenID>& generated) override;

        private:
            /** @brief Takes the parts make() built. */
            WindowGenerator(Train::EvaluationStep& step, const SushiBLAS::Tensor& logits,
                            Data::Batch feed) noexcept;

            /// @brief Runs the forward pass; not owned.
            Train::EvaluationStep* step_;

            /// @brief Views the [1, T, V] logits the step's pool owns.
            const SushiBLAS::Tensor* logits_;

            /// @brief Holds the [1, T] int32 window each run feeds.
            Data::Batch feed_;
    };

} // namespace SushiAI::Language::Generation
