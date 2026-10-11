/**************************************************************************/
/* test_generation.cpp                                                    */
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
 * @file test_generation.cpp
 * @brief Tests WindowGenerator over a decoder built from a model file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>
#include "decoder_model.hpp"

using namespace SushiAI;
using Language::Generation::GreedySampler;
using Language::Generation::RandomGenerator;
using Language::Generation::TemperatureSampler;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds the tokens one run generates; with the prompt they pass the window. */
    constexpr std::int64_t K_TOKENS = 12;

    /** @brief Views ids as the span the generator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Builds the decoder, initialises it and compiles its [1, T] forward pass. */
    class Integration_Generation : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds every part a generator reads, in the order they depend. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();

                const Config::ModelSpec spec = SaTest::decoder_spec();
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built, SaTest::build_decoder(spec));
                model = std::make_unique<NN::ModelGraph>(std::move(built));

                Train::GraphOptions options;
                options.dtype = spec.input_dtype;
                SA_ASSERT_OK_AND_ASSIGN(
                    const Train::TracedStep traced,
                    Train::build_evaluation_graph(*model, Shape{1, SaTest::K_DECODER_WINDOW},
                                                  options));

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                SA_EXPECT_OK(NN::initialize(*engine, *state));

                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(*engine, traced, *state,
                                                                    *pool));
                step = std::move(compiled);

                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<WindowGenerator> made,
                    WindowGenerator::make(*engine, *step, traced.outputs.all().front().first));
                generator = std::move(made);
            }

            /** @brief Releases the parts before the engine they were built on. */
            void TearDown() override
            {
                generator.reset();
                step.reset();
                pool.reset();
                state.reset();
                model.reset();
                SaTest::EngineTest::TearDown();
            }

            /// @brief Holds the decoder.
            std::unique_ptr<NN::ModelGraph> model;

            /// @brief Holds the initialised parameters.
            std::unique_ptr<NN::ModelState> state;

            /// @brief Holds every value of the forward graph.
            std::unique_ptr<Graph::TensorPool> pool;

            /// @brief Holds the compiled forward pass.
            std::unique_ptr<Train::EvaluationStep> step;

            /// @brief Holds the generator under test.
            std::unique_ptr<WindowGenerator> generator;
    };
} // namespace

/** @brief Tests that the generator reads the window and the vocabulary of the step. */
TEST_F(Integration_Generation, TheGeneratorReadsTheExtentsOfTheStep)
{
    EXPECT_EQ(generator->window(), SaTest::K_DECODER_WINDOW);
    EXPECT_EQ(generator->vocabulary(), SaTest::K_DECODER_VOCABULARY);
}

/** @brief Tests that twelve greedy tokens equal twelve runs of one token over a grown prompt. */
TEST_F(Integration_Generation, ARunEqualsOneTokenAtATimeOverTheGrownPrompt)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::vector<TokenID> prompt{1, 2, 3};

    std::vector<TokenID> together;
    SA_EXPECT_OK(generator->generate(span_of(prompt), K_TOKENS, sampler, random, together));
    ASSERT_EQ(together.size(), static_cast<std::size_t>(K_TOKENS));

    std::vector<TokenID> grown = prompt;
    for (std::int64_t made = 0; made < K_TOKENS; ++made)
    {
        std::vector<TokenID> one;
        SA_EXPECT_OK(generator->generate(span_of(grown), 1, sampler, random, one));
        ASSERT_EQ(one.size(), 1u);
        EXPECT_EQ(one[0], together[static_cast<std::size_t>(made)]) << "token " << made;
        grown.push_back(one[0]);
    }

    for (const TokenID token : together)
    {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, SaTest::K_DECODER_VOCABULARY);
    }

    EXPECT_EQ(step->step().compile_count(), 1u);
}

/** @brief Tests that a drawing sampler gives one sequence for one seed. */
TEST_F(Integration_Generation, ASeedFixesADrawnContinuation)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<TemperatureSampler> sampler,
                            TemperatureSampler::make(1.0));
    const std::vector<TokenID> prompt{5, 9};

    RandomGenerator first_random(K_SEED);
    RandomGenerator again_random(K_SEED);
    std::vector<TokenID> first;
    std::vector<TokenID> again;
    SA_EXPECT_OK(generator->generate(span_of(prompt), K_TOKENS, *sampler, first_random, first));
    SA_EXPECT_OK(generator->generate(span_of(prompt), K_TOKENS, *sampler, again_random, again));

    EXPECT_EQ(first, again);
}

/** @brief Tests that an empty prompt, an id outside the vocabulary and no token are refused. */
TEST_F(Integration_Generation, APromptOrACountThatDoesNotFitIsRefused)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    std::vector<TokenID> generated;

    SA_EXPECT_FAILURE(generator->generate(span_of({}), 1, sampler, random, generated),
                      StatusCode::InvalidDescription, "no id");
    SA_EXPECT_FAILURE(generator->generate(span_of({1, 32}), 1, sampler, random, generated),
                      StatusCode::InvalidDescription, "id 1", "32", "0 to 31");
    SA_EXPECT_FAILURE(generator->generate(span_of({1}), 0, sampler, random, generated),
                      StatusCode::InvalidDescription, "0 tokens");
    EXPECT_TRUE(generated.empty());
}
