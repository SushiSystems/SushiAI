/**************************************************************************/
/* test_cached_generation.cpp                                             */
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
 * @file test_cached_generation.cpp
 * @brief Tests CachedGenerator against WindowGenerator over a decoder with a learned table.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/sampler.hpp>
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
using Language::Generation::CachedGenerator;
using Language::Generation::CachedStepLeaves;
using Language::Generation::GreedySampler;
using Language::Generation::RandomGenerator;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the seed of every generator of these tests; a greedy sampler draws nothing. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds the tokens of a run whose context stays shorter than the window. */
    constexpr std::int64_t K_FITTING_TOKENS = 4;

    /** @brief Holds the tokens of a run whose context reaches the window and slides. */
    constexpr std::int64_t K_SLIDING_TOKENS = 12;

    /** @brief Holds the absolute tolerance between a logit of the two float32 passes. */
    constexpr float K_LOGIT_TOLERANCE = 1e-4f;

    /** @brief Views ids as the span a generator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Returns the leaves the incremental forms of the blocks declare. */
    CachedStepLeaves block_leaves()
    {
        return CachedStepLeaves{Language::Blocks::K_POSITION_LEAF,
                                Language::Blocks::K_ATTENTION_MASK_LEAF,
                                Language::Blocks::K_MASKED_ATTENTION_SCORE};
    }

    /** @brief Holds one compiled one-position step, its pool and the generator over it. */
    struct CachedPath
    {
        std::unique_ptr<Graph::TensorPool> pool;     /**< Every value of the graph. */
        std::unique_ptr<Train::EvaluationStep> step; /**< The compiled position. */
        std::unique_ptr<CachedGenerator> generator;  /**< The generator over the step. */
    };

    /** @brief Builds the decoder, one state and both generators over it. */
    class Integration_CachedGeneration : public SaTest::EngineTest
    {
        protected:
            /** @brief Builds every part the two generators read, in the order they depend. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();

                const Config::ModelSpec spec = SaTest::decoder_spec();
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built, SaTest::build_decoder(spec));
                model = std::make_unique<NN::ModelGraph>(std::move(built));

                Train::GraphOptions options;
                options.dtype = spec.input_dtype;
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep whole,
                    Train::build_evaluation_graph(*model, Shape{1, SaTest::K_DECODER_WINDOW},
                                                  options));
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep sliced,
                    Train::build_incremental_graph(*model, Shape{1, 1}, options));
                incremental = std::make_unique<Train::TracedStep>(std::move(sliced));
                output = whole.outputs.all().front().first;

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                SA_EXPECT_OK(NN::initialize(*engine, *state));

                window_pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(*engine, whole, *state,
                                                                    *window_pool));
                window_step = std::move(compiled);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WindowGenerator> made,
                                        WindowGenerator::make(*engine, *window_step, output));
                window = std::move(made);

                make_cached_path(cached);
            }

            /** @brief Releases the parts before the engine they were built on. */
            void TearDown() override
            {
                cached.generator.reset();
                cached.step.reset();
                cached.pool.reset();
                window.reset();
                window_step.reset();
                window_pool.reset();
                state.reset();
                incremental.reset();
                model.reset();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Compiles the one-position graph into @p path and builds its generator. */
            void make_cached_path(CachedPath& path)
            {
                path.pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(*engine, *incremental,
                                                                    *state, *path.pool));
                path.step = std::move(compiled);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<CachedGenerator> made,
                    CachedGenerator::make(*engine, *path.step, output, block_leaves()));
                path.generator = std::move(made);
            }

            /** @brief Returns @p count greedy tokens of @p generator after @p prompt. */
            std::vector<TokenID> continue_with(Language::Generation::Generator& generator,
                                               const std::vector<TokenID>& prompt,
                                               std::int64_t count)
            {
                const GreedySampler sampler;
                RandomGenerator random(K_SEED);
                std::vector<TokenID> generated;
                SA_EXPECT_OK(generator.generate(span_of(prompt), count, sampler, random,
                                                generated));
                return generated;
            }

            /// @brief Holds the decoder both graphs were traced from.
            std::unique_ptr<NN::ModelGraph> model;

            /// @brief Holds the one-position graph, from which a test compiles a second step.
            std::unique_ptr<Train::TracedStep> incremental;

            /// @brief Holds the name both graphs publish the logits under.
            std::string output;

            /// @brief Holds the initialised parameters both steps read.
            std::unique_ptr<NN::ModelState> state;

            /// @brief Holds every value of the window graph.
            std::unique_ptr<Graph::TensorPool> window_pool;

            /// @brief Holds the compiled pass of the whole window.
            std::unique_ptr<Train::EvaluationStep> window_step;

            /// @brief Holds the reference generator.
            std::unique_ptr<WindowGenerator> window;

            /// @brief Holds the generator under test and what it runs on.
            CachedPath cached;
    };
} // namespace

/** @brief Tests that the cached generator reads the window and the vocabulary of its step. */
TEST_F(Integration_CachedGeneration, TheGeneratorReadsTheExtentsOfTheStep)
{
    EXPECT_EQ(cached.generator->window(), SaTest::K_DECODER_WINDOW);
    EXPECT_EQ(cached.generator->vocabulary(), SaTest::K_DECODER_VOCABULARY);
    EXPECT_TRUE(cached.step->carries_state());
    ASSERT_EQ(cached.step->input_names().size(), 3u);

    // A learned table places a token by its position, so the step is fed no cache slot.
    SA_EXPECT_OK(cached.step->input(Language::Blocks::K_POSITION_LEAF));
    SA_EXPECT_FAILURE(cached.step->input(Language::Blocks::K_CACHE_SLOT_LEAF),
                      StatusCode::InvalidDescription, "'cache_slot'");
    EXPECT_EQ(incremental->carried.values().size(),
              static_cast<std::size_t>(2 * SaTest::K_DECODER_BLOCKS));
}

/** @brief Tests that both traces leave the state with the parameters of the model alone. */
TEST_F(Integration_CachedGeneration, TheCachesAreNoEntryOfTheModelState)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelState again,
                            NN::ModelState::create(*engine, *model));

    EXPECT_EQ(again.entries().size(), state->entries().size());
    for (const NN::StateEntry& entry : again.entries())
        EXPECT_EQ(entry.kind, NN::StateKind::Parameter) << entry.name;
}

/** @brief Tests that a prompt that fits gives the tokens of the window generator. */
TEST_F(Integration_CachedGeneration, APromptThatFitsGivesTheTokensOfTheWindowGenerator)
{
    const std::vector<TokenID> prompt{1, 2, 3};

    const std::vector<TokenID> expected = continue_with(*window, prompt, K_FITTING_TOKENS);
    const std::vector<TokenID> generated =
        continue_with(*cached.generator, prompt, K_FITTING_TOKENS);

    ASSERT_EQ(expected.size(), static_cast<std::size_t>(K_FITTING_TOKENS));
    EXPECT_EQ(generated, expected);

    // Three runs feed the prompt and one feeds each token but the last, which nothing reads.
    EXPECT_EQ(cached.step->step().run_count(), 6u);
    EXPECT_EQ(cached.step->step().compile_count(), 1u);
}

/** @brief Tests that the logits after a prompt equal those of the window at its last position. */
TEST_F(Integration_CachedGeneration, TheLogitsEqualThoseOfTheWindowAtTheLastPosition)
{
    const std::vector<TokenID> prompt{4, 7, 1, 30, 2};
    static_cast<void>(continue_with(*window, prompt, 1));
    static_cast<void>(continue_with(*cached.generator, prompt, 1));

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const whole, window_step->output(output));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const one, cached.step->output(output));
    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    const float* const expected = whole->data_as<float>() + (prompt.size() - 1) * scores;
    const float* const computed = one->data_as<float>();

    for (std::size_t token = 0; token < scores; ++token)
        EXPECT_NEAR(computed[token], expected[token], K_LOGIT_TOLERANCE) << "token " << token;
}

/** @brief Tests that a context that reaches the window and slides gives the same tokens. */
TEST_F(Integration_CachedGeneration, AContextThatSlidesGivesTheTokensOfTheWindowGenerator)
{
    const std::vector<TokenID> prompt{1, 2, 3};

    const std::vector<TokenID> expected = continue_with(*window, prompt, K_SLIDING_TOKENS);
    const std::vector<TokenID> generated =
        continue_with(*cached.generator, prompt, K_SLIDING_TOKENS);

    ASSERT_EQ(expected.size(), static_cast<std::size_t>(K_SLIDING_TOKENS));
    EXPECT_EQ(generated, expected);

    // 3 runs feed the prompt, 5 fill the window, and each of 6 slides feeds 8 ids again.
    EXPECT_EQ(cached.step->step().run_count(), 56u);
    EXPECT_EQ(cached.step->step().compile_count(), 1u);
}

/** @brief Tests that a prompt longer than the window keeps its last T ids, as the window does. */
TEST_F(Integration_CachedGeneration, APromptLongerThanTheWindowKeepsItsLastIDs)
{
    const std::vector<TokenID> prompt{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11};

    const std::vector<TokenID> expected = continue_with(*window, prompt, K_FITTING_TOKENS);
    const std::vector<TokenID> generated =
        continue_with(*cached.generator, prompt, K_FITTING_TOKENS);

    EXPECT_EQ(generated, expected);
}

/** @brief Tests that a second prompt gives what a generator that saw no first prompt gives. */
TEST_F(Integration_CachedGeneration, ASecondPromptMatchesAFreshGenerator)
{
    const std::vector<TokenID> first{5, 9, 13, 2};
    const std::vector<TokenID> second{20, 3};
    static_cast<void>(continue_with(*cached.generator, first, K_SLIDING_TOKENS));
    const std::vector<TokenID> reused =
        continue_with(*cached.generator, second, K_FITTING_TOKENS);

    CachedPath fresh;
    make_cached_path(fresh);
    ASSERT_NE(fresh.generator, nullptr);
    const std::vector<TokenID> untouched =
        continue_with(*fresh.generator, second, K_FITTING_TOKENS);

    EXPECT_EQ(reused, untouched);
    EXPECT_EQ(reused, continue_with(*window, second, K_FITTING_TOKENS));
    EXPECT_EQ(cached.step->step().compile_count(), 1u);
    EXPECT_EQ(fresh.step->step().compile_count(), 1u);
}

/** @brief Tests that a step that carries nothing and a leaf that is absent are refused. */
TEST_F(Integration_CachedGeneration, AStepThatDoesNotFitIsRefused)
{
    SA_EXPECT_FAILURE(CachedGenerator::make(*engine, *window_step, output, block_leaves()),
                      StatusCode::InvalidDescription, "keeps nothing between runs");

    CachedStepLeaves misnamed = block_leaves();
    misnamed.mask = "scores";
    SA_EXPECT_FAILURE(CachedGenerator::make(*engine, *cached.step, output, misnamed),
                      StatusCode::InvalidDescription, "'scores'", "'attention_mask'");
}

/** @brief Tests that an empty prompt, an id outside the vocabulary and no token are refused. */
TEST_F(Integration_CachedGeneration, APromptOrACountThatDoesNotFitIsRefused)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    std::vector<TokenID> generated;

    SA_EXPECT_FAILURE(cached.generator->generate(span_of({}), 1, sampler, random, generated),
                      StatusCode::InvalidDescription, "CachedGenerator", "no id");
    SA_EXPECT_FAILURE(
        cached.generator->generate(span_of({1, 32}), 1, sampler, random, generated),
        StatusCode::InvalidDescription, "id 1", "32", "0 to 31");
    SA_EXPECT_FAILURE(cached.generator->generate(span_of({1}), 0, sampler, random, generated),
                      StatusCode::InvalidDescription, "0 tokens");
    EXPECT_TRUE(generated.empty());
    EXPECT_EQ(cached.step->step().run_count(), 0u);
}
