/**************************************************************************/
/* test_window_generator.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_window_generator.cpp
 * @brief Tests WindowGenerator over a decoder built from a model file, and over an in-memory
 *        source of window logits.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/decoder_models/decoder_model.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/in_memory_window_logits.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>

using namespace SushiAI;
using Language::Generation::GreedySampler;
using Language::Generation::InMemoryWindowLogits;
using Language::Generation::RandomGenerator;
using Language::Generation::TemperatureSampler;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;
// SaTest stays reachable inside a TEST_F body; see tests/language/generation/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the seed of every generator of these tests. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds the tokens one run generates; with the prompt they pass the window. */
    constexpr std::int64_t K_TOKENS = 12;

    /** @brief Holds T, the window of the in-memory source. */
    constexpr std::int64_t K_SOURCE_WINDOW = 8;

    /** @brief Holds V, the vocabulary of the in-memory source. */
    constexpr std::int64_t K_SOURCE_VOCABULARY = 32;

    /** @brief Views ids as the span the generator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /** @brief Reports whether every id of @p ids is a token of the decoder's vocabulary. */
    bool all_in_vocabulary(const std::vector<TokenID>& ids)
    {
        return std::all_of(ids.begin(), ids.end(), [](TokenID id)
        {
            return id >= 0 && id < SaTest::K_DECODER_VOCABULARY;
        });
    }

    /** @brief Builds the decoder, initialises it and compiles its [1, T] forward pass. */
    class Medium_Generation : public SaTest::EngineTest
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
                SA_EXPECT_OK(NN::initialize(backend(), *state));

                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::EvaluationStep> compiled,
                                        Train::EvaluationStep::make(backend(), traced, *state,
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

            /**
             * @brief Returns the tokens of @p count runs of one token, each over the prompt grown
             *        by the tokens before it; it stops at the first run that fails, so the
             *        caller's comparison fails with fewer tokens.
             */
            std::vector<TokenID> one_token_at_a_time(const std::vector<TokenID>& prompt,
                                                     std::int64_t count)
            {
                const GreedySampler sampler;
                RandomGenerator random(K_SEED);
                std::vector<TokenID> grown = prompt;
                std::vector<TokenID> made;
                for (std::int64_t run = 0; run < count; ++run)
                {
                    std::vector<TokenID> one;
                    if (!generator->generate(span_of(grown), 1, sampler, random, one).is_ok())
                        return made;

                    grown.insert(grown.end(), one.begin(), one.end());
                    made.insert(made.end(), one.begin(), one.end());
                }

                return made;
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

/** @brief Tests that twelve greedy tokens equal twelve runs of one token over a grown prompt. */
TEST_F(Medium_Generation, ARunEqualsOneTokenAtATimeOverTheGrownPrompt)
{
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::vector<TokenID> prompt{1, 2, 3};

    std::vector<TokenID> together;
    SA_EXPECT_OK(generator->generate(span_of(prompt), K_TOKENS, sampler, random, together));

    EXPECT_EQ(together, one_token_at_a_time(prompt, K_TOKENS));
    EXPECT_TRUE(all_in_vocabulary(together));
}

/** @brief Tests that a drawing sampler gives one sequence for one seed. */
TEST(Small_Generation, ASeedFixesADrawnContinuation)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<WindowGenerator> generator,
        WindowGenerator::make(
            std::make_unique<InMemoryWindowLogits>(K_SOURCE_WINDOW, K_SOURCE_VOCABULARY)));
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
