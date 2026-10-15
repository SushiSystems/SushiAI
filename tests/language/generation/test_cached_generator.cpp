/**************************************************************************/
/* test_cached_generator.cpp                                              */
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
 * @file test_cached_generator.cpp
 * @brief Tests CachedGenerator against WindowGenerator over decoders with a learned table and with
 *        rotary positions.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/language/decoder_models/decoder_model.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/layer_rows/minimal_case.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>

using namespace SushiAI;
using Language::Generation::CachedGenerator;
using Language::Generation::CachedStepLeaves;
using Language::Generation::GreedySampler;
using Language::Generation::RandomGenerator;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;
// SaTest stays reachable inside a TEST_F body; see tests/language/generation/README.md.
namespace SaTest = SushiAI::Test;
using Language::Generation::LogitSpan;
using Language::Generation::RingStepLeaves;

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

namespace
{

    /** @brief Holds T, the rows of the ring under test. */
    constexpr std::int64_t K_WINDOW = SaTest::K_DECODER_WINDOW;

    /** @brief Holds the tokens of a run that slides for more than three windows. */
    constexpr std::int64_t K_LONG_TOKENS = 3 * K_WINDOW + 2;

    /** @brief Holds the ids of a fed text that slides for more than three windows. */
    constexpr std::size_t K_FED_IDS = 30;

    /** @brief Holds the entries of the state of the two-block rotary decoder. */
    constexpr std::size_t K_DEEP_STATE_ENTRIES = 1 + 2 * 12 + 2;

    /** @brief Counts the caches a block carries: its keys and its values. */
    constexpr std::size_t K_CACHES = 2;

    /** @brief Holds a decoder with a learned position table before a rotary block. */
    constexpr std::string_view K_MIXED_DECODER_MODEL = R"({
        "format_version": 2,
        "input": {"shape": [8], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
            {"type": "position_embedding"},
            {"type": "transformer_block", "heads": 2, "positions": "rotary"},
            {"type": "language_head", "tie": 0}
        ],
        "task": {"type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8}
    })";

    /** @brief Returns the leaves the incremental forms of the blocks declare. */
    CachedStepLeaves rotary_block_leaves()
    {
        return CachedStepLeaves{
            Language::Blocks::K_POSITION_LEAF, Language::Blocks::K_ATTENTION_MASK_LEAF,
            Language::Blocks::K_MASKED_ATTENTION_SCORE,
            RingStepLeaves{Language::Blocks::K_CACHE_SLOT_LEAF,
                           Language::Blocks::K_ROTARY_COSINE_LEAF,
                           Language::Blocks::K_ROTARY_SINE_LEAF,
                           &Language::Blocks::write_rotary_row}};
    }

    /** @brief Returns @p count ids, id i being 7 i^2 + 3 i + 1 modulo the vocabulary. */
    std::vector<TokenID> fed_text(std::size_t count)
    {
        std::vector<TokenID> ids;
        ids.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            ids.push_back(static_cast<TokenID>((7 * i * i + 3 * i + 1) %
                                               static_cast<std::size_t>(
                                                   SaTest::K_DECODER_VOCABULARY)));
        }

        return ids;
    }

    /** @brief Feeds a one-position rotary step by hand, one token per call. */
    class RingDriver final
    {
        public:
            /** @brief Allocates the five feeds of @p step, whose mask has the cache's rows. */
            void bind(SushiBLAS::Engine& engine, Train::EvaluationStep& step)
            {
                step_ = &step;
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const mask,
                                        step.input(Language::Blocks::K_ATTENTION_MASK_LEAF));
                SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const cosine,
                                        step.input(Language::Blocks::K_ROTARY_COSINE_LEAF));
                rows_ = mask->shape[1];
                pairs_ = static_cast<std::size_t>(cosine->shape[1]);

                const Data::InputDescription index{Shape{1}, DType::Int32};
                const Data::InputDescription angles{Shape{cosine->shape[1]}, DType::Float32};
                SA_EXPECT_OK(Data::prepare_batch(engine, 1, index, {}, id_));
                SA_EXPECT_OK(Data::prepare_batch(engine, 1, index, {}, slot_));
                SA_EXPECT_OK(Data::prepare_batch(engine, 1, angles, {}, cosine_));
                SA_EXPECT_OK(Data::prepare_batch(engine, 1, angles, {}, sine_));
                SA_EXPECT_OK(Data::prepare_batch(
                    engine, 1, Data::InputDescription{Shape{rows_}, DType::Float32}, {}, mask_));
            }

            /**
             * @brief Runs the step for @p id at @p position, read by the positions of the last
             *        @p band tokens only.
             * @param band At most the rows of the cache, so no read row has been written over.
             * @pre bind() succeeded.
             */
            void feed(TokenID id, std::int64_t position, std::int64_t band)
            {
                ASSERT_NE(step_, nullptr);
                float* const mask = mask_.inputs.data_as<float>();
                std::fill(mask, mask + rows_,
                          static_cast<float>(Language::Blocks::K_MASKED_ATTENTION_SCORE));
                for (std::int64_t read = std::max<std::int64_t>(0, position - band + 1);
                     read <= position; ++read)
                {
                    mask[read % rows_] = 0.0f;
                }

                id_.inputs.data_as<TokenID>()[0] = id;
                slot_.inputs.data_as<std::int32_t>()[0] =
                    static_cast<std::int32_t>(position % rows_);
                Language::Blocks::write_rotary_row(
                    position, SushiRuntime::span<float>(cosine_.inputs.data_as<float>(), pairs_),
                    SushiRuntime::span<float>(sine_.inputs.data_as<float>(), pairs_));

                const Train::NamedFeed fed[] = {
                    {Train::K_MODEL_INPUT_NAME, &id_.inputs},
                    {Language::Blocks::K_CACHE_SLOT_LEAF, &slot_.inputs},
                    {Language::Blocks::K_ATTENTION_MASK_LEAF, &mask_.inputs},
                    {Language::Blocks::K_ROTARY_COSINE_LEAF, &cosine_.inputs},
                    {Language::Blocks::K_ROTARY_SINE_LEAF, &sine_.inputs},
                };
                SA_EXPECT_OK(step_->run_by_name(
                    SushiRuntime::span<const Train::NamedFeed>(fed, std::size(fed))));
            }

        private:
            /// @brief Runs one position; not owned, and null until bind().
            Train::EvaluationStep* step_ = nullptr;

            /// @brief Holds the rows of a cache of the step.
            std::int64_t rows_ = 0;

            /// @brief Holds the pairs of a group, the columns of an angle leaf.
            std::size_t pairs_ = 0;

            /// @brief Holds the [1, 1] int32 id.
            Data::Batch id_;

            /// @brief Holds the [1, 1] int32 cache slot.
            Data::Batch slot_;

            /// @brief Holds the [1, pairs] cosines.
            Data::Batch cosine_;

            /// @brief Holds the [1, pairs] sines.
            Data::Batch sine_;

            /// @brief Holds the [1, rows] mask.
            Data::Batch mask_;
    };

    /** @brief Holds one compiled one-position step, its pool and the generator over it. */
    struct RotaryCachedPath
    {
        std::unique_ptr<Graph::TensorPool> pool;     /**< Every value of the graph. */
        std::unique_ptr<Train::EvaluationStep> step; /**< The compiled position. */
        std::unique_ptr<CachedGenerator> generator;  /**< The generator over the step. */

        /** @brief Releases the parts in the order they depend on each other. */
        void release()
        {
            generator.reset();
            step.reset();
            pool.reset();
        }
    };

    /** @brief Holds one decoder, its state, its window path and its cached path. */
    struct Decoder
    {
        std::unique_ptr<NN::ModelGraph> model;            /**< The rows of the model file. */
        std::unique_ptr<Train::TracedStep> incremental;   /**< The one-position graph. */
        std::size_t window_tables = 0;                    /**< The host tables of the window. */
        std::string output;                               /**< The name of the logits. */
        std::unique_ptr<NN::ModelState> state;            /**< The initialised parameters. */
        std::unique_ptr<Graph::TensorPool> window_pool;   /**< Every value of the window. */
        std::unique_ptr<Train::EvaluationStep> window_step; /**< The compiled window. */
        std::unique_ptr<WindowGenerator> window;          /**< The reference generator. */
        RotaryCachedPath cached;                                /**< The generator under test. */

        /** @brief Releases the parts in the order they depend on each other. */
        void release()
        {
            cached.release();
            window.reset();
            window_step.reset();
            window_pool.reset();
            state.reset();
            incremental.reset();
            model.reset();
        }
    };

    /** @brief Builds the rotary decoders and both generators over each. */
    class Integration_RotaryGeneration : public SaTest::EngineTest
    {
        protected:
            /** @brief Releases every decoder before the engine they were built on. */
            void TearDown() override
            {
                wide.release();
                shallow.release();
                deep.release();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Builds the model of @p text, its state and the graph of one position. */
            void build_model(std::string_view text, Decoder& decoder)
            {
                const Config::ModelSpec spec = SaTest::model_from(text);
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built, SaTest::build_decoder(spec));
                decoder.model = std::make_unique<NN::ModelGraph>(std::move(built));

                Train::GraphOptions options;
                options.dtype = spec.input_dtype;
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep sliced,
                    Train::build_incremental_graph(*decoder.model, Shape{1, 1}, options));
                decoder.incremental = std::make_unique<Train::TracedStep>(std::move(sliced));
                decoder.output = decoder.incremental->outputs.all().front().first;

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *decoder.model));
                decoder.state = std::make_unique<NN::ModelState>(std::move(created));
                SA_EXPECT_OK(NN::initialize(*engine, *decoder.state));
            }

            /** @brief Compiles the one-position graph of @p decoder into @p path. */
            void compile_position(Decoder& decoder, RotaryCachedPath& path)
            {
                path.pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<Train::EvaluationStep> compiled,
                    Train::EvaluationStep::make(*engine, *decoder.incremental, *decoder.state,
                                                *path.pool));
                path.step = std::move(compiled);
            }

            /** @brief Compiles the one-position graph and builds the cached generator. */
            void make_cached_path(Decoder& decoder, RotaryCachedPath& path)
            {
                compile_position(decoder, path);
                ASSERT_NE(path.step, nullptr);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<CachedGenerator> made,
                    CachedGenerator::make(*engine, *path.step, decoder.output, rotary_block_leaves()));
                path.generator = std::move(made);
            }

            /** @brief Builds every part of the decoder @p text describes over a window of T. */
            void build(std::string_view text, Decoder& decoder)
            {
                build_model(text, decoder);
                ASSERT_NE(decoder.state, nullptr);

                Train::GraphOptions options;
                options.dtype = DType::Int32;
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep whole,
                    Train::build_evaluation_graph(*decoder.model, Shape{1, K_WINDOW}, options));
                decoder.window_tables = whole.tables.tables().size();

                decoder.window_pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<Train::EvaluationStep> compiled,
                    Train::EvaluationStep::make(*engine, whole, *decoder.state,
                                                *decoder.window_pool));
                decoder.window_step = std::move(compiled);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<WindowGenerator> made,
                    WindowGenerator::make(*engine, *decoder.window_step, decoder.output));
                decoder.window = std::move(made);

                make_cached_path(decoder, decoder.cached);
            }

            /** @brief Copies every parameter of @p from into the entry of @p to it is named as. */
            void copy_state(NN::ModelState& from, NN::ModelState& to)
            {
                ASSERT_EQ(from.entries().size(), to.entries().size());
                for (const NN::StateEntry& entry : from.entries())
                {
                    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const source,
                                            from.tensor(entry.name));
                    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* const target,
                                            to.tensor(entry.name));
                    ASSERT_EQ(source->num_elements, target->num_elements) << entry.name;
                    ASSERT_EQ(source->dtype, DType::Float32) << entry.name;
                    std::copy(source->data_as<float>(),
                              source->data_as<float>() + source->num_elements,
                              target->data_as<float>());
                }
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

            /** @brief Returns the [1, 1, V] logits @p step computed in its last run. */
            const float* logits_of(const Train::EvaluationStep& step, const std::string& output)
            {
                const Result<const SushiBLAS::Tensor*> tensor = step.output(output);
                EXPECT_TRUE(tensor.has_value());
                return tensor.has_value() ? tensor.value()->data_as<float>() : nullptr;
            }

            /// @brief Holds the rotary decoder of two blocks over a window of T.
            Decoder deep;

            /// @brief Holds the rotary decoder of one block over a window of T.
            Decoder shallow;

            /// @brief Holds the two-block decoder over a window no test fills.
            Decoder wide;
    };
} // namespace

/** @brief Tests that the one-position step is fed a ring and that no position is among it. */
TEST_F(Integration_RotaryGeneration, TheStepIsFedTheRingAndNoPosition)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);

    EXPECT_EQ(deep.cached.generator->window(), K_WINDOW);
    EXPECT_EQ(deep.cached.generator->vocabulary(), SaTest::K_DECODER_VOCABULARY);
    ASSERT_EQ(deep.cached.step->input_names().size(), 5u);
    SA_EXPECT_OK(deep.cached.step->input(Language::Blocks::K_CACHE_SLOT_LEAF));
    SA_EXPECT_OK(deep.cached.step->input(Language::Blocks::K_ROTARY_COSINE_LEAF));
    SA_EXPECT_OK(deep.cached.step->input(Language::Blocks::K_ROTARY_SINE_LEAF));
    SA_EXPECT_FAILURE(deep.cached.step->input(Language::Blocks::K_POSITION_LEAF),
                      StatusCode::InvalidDescription, "'position'");

    // A block carries its keys and its values, and the window form holds two tables per block.
    EXPECT_EQ(deep.incremental->carried.values().size(),
              static_cast<std::size_t>(2 * SaTest::K_DECODER_BLOCKS));
    EXPECT_TRUE(deep.incremental->tables.tables().empty());
    EXPECT_EQ(deep.window_tables, static_cast<std::size_t>(2 * SaTest::K_DECODER_BLOCKS));
}

/** @brief Tests that the state holds parameters only: no position table, no angle table. */
TEST_F(Integration_RotaryGeneration, TheStateHoldsParametersOnly)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.state, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelState again,
                            NN::ModelState::create(*engine, *deep.model));
    EXPECT_EQ(again.entries().size(), K_DEEP_STATE_ENTRIES);
    for (const NN::StateEntry& entry : again.entries())
    {
        EXPECT_EQ(entry.kind, NN::StateKind::Parameter) << entry.name;
        EXPECT_EQ(entry.name.find("rotary"), std::string::npos) << entry.name;
    }
}

/** @brief Tests that the logits after a prompt equal those of the window at its last position. */
TEST_F(Integration_RotaryGeneration, TheLogitsEqualThoseOfTheWindowAtTheLastPosition)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);

    const std::vector<TokenID> prompt{4, 7, 1, 30, 2};
    static_cast<void>(continue_with(*deep.window, prompt, 1));
    static_cast<void>(continue_with(*deep.cached.generator, prompt, 1));

    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    const float* const whole = logits_of(*deep.window_step, deep.output);
    const float* const one = logits_of(*deep.cached.step, deep.output);
    ASSERT_NE(whole, nullptr);
    ASSERT_NE(one, nullptr);

    const float* const expected = whole + (prompt.size() - 1) * scores;
    for (std::size_t token = 0; token < scores; ++token)
        EXPECT_NEAR(one[token], expected[token], K_LOGIT_TOLERANCE) << "token " << token;
}

/** @brief Tests that a prompt that fits gives the tokens of the window generator. */
TEST_F(Integration_RotaryGeneration, APromptThatFitsGivesTheTokensOfTheWindowGenerator)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);

    const std::vector<TokenID> prompt{1, 2, 3};
    EXPECT_EQ(continue_with(*deep.cached.generator, prompt, K_FITTING_TOKENS),
              continue_with(*deep.window, prompt, K_FITTING_TOKENS));
}

/** @brief Tests that a one-block ring gives the window generator's tokens over three windows. */
TEST_F(Integration_RotaryGeneration, AOneBlockRingGivesTheTokensOfTheWindowGeneratorAcrossSlides)
{
    build(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, shallow);
    ASSERT_NE(shallow.cached.generator, nullptr);

    const std::vector<TokenID> prompt{1, 2, 3};
    const std::vector<TokenID> expected = continue_with(*shallow.window, prompt, K_LONG_TOKENS);
    const std::vector<TokenID> generated =
        continue_with(*shallow.cached.generator, prompt, K_LONG_TOKENS);

    ASSERT_EQ(expected.size(), static_cast<std::size_t>(K_LONG_TOKENS));
    EXPECT_EQ(generated, expected);

    // Three runs feed the prompt and one feeds each token but the last, which nothing reads.
    EXPECT_EQ(shallow.cached.step->step().run_count(),
              static_cast<std::size_t>(3 + K_LONG_TOKENS - 1));
    EXPECT_EQ(shallow.cached.step->step().compile_count(), 1u);
}

/** @brief Tests the logits of a one-block ring against the window's at every fed position. */
TEST_F(Integration_RotaryGeneration, TheLogitsOfAOneBlockRingEqualTheWindowsAtEveryPosition)
{
    build(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, shallow);
    ASSERT_NE(shallow.window_step, nullptr);

    RotaryCachedPath driven;
    compile_position(shallow, driven);
    ASSERT_NE(driven.step, nullptr);
    RingDriver ring;
    ring.bind(*engine, *driven.step);

    Data::Batch window;
    SA_EXPECT_OK(Data::prepare_batch(
        *engine, 1, Data::InputDescription{Shape{K_WINDOW}, DType::Int32}, {}, window));

    const std::vector<TokenID> text = fed_text(K_FED_IDS);
    const std::size_t length = static_cast<std::size_t>(K_WINDOW);
    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    for (std::size_t position = 0; position < text.size(); ++position)
    {
        ring.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);

        // The window holds the last T ids at positions 0 to T - 1, as WindowGenerator feeds it.
        const std::size_t held = std::min(position + 1, length);
        TokenID* const fed = window.inputs.data_as<TokenID>();
        std::fill(fed, fed + length, TokenID{0});
        std::copy(text.begin() + static_cast<std::ptrdiff_t>(position + 1 - held),
                  text.begin() + static_cast<std::ptrdiff_t>(position + 1), fed);
        SA_EXPECT_OK(shallow.window_step->run(window.inputs));

        const float* const whole = logits_of(*shallow.window_step, shallow.output);
        const float* const one = logits_of(*driven.step, shallow.output);
        ASSERT_NE(whole, nullptr);
        ASSERT_NE(one, nullptr);
        for (std::size_t token = 0; token < scores; ++token)
        {
            EXPECT_NEAR(one[token], whole[(held - 1) * scores + token], K_LOGIT_TOLERANCE)
                << "position " << position << " token " << token;
        }
    }

    driven.release();
}

/** @brief Tests a ring of T rows against a cache of 64 rows read through the last T alone. */
TEST_F(Integration_RotaryGeneration, ARingEqualsAWideCacheReadThroughItsLastPositions)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);
    build_model(SaTest::K_WIDE_ROTARY_DECODER_MODEL, wide);
    ASSERT_NE(wide.state, nullptr);
    copy_state(*deep.state, *wide.state);
    compile_position(wide, wide.cached);
    ASSERT_NE(wide.cached.step, nullptr);

    const std::vector<TokenID> prompt{1, 2, 3};
    const std::vector<TokenID> generated =
        continue_with(*deep.cached.generator, prompt, K_LONG_TOKENS);
    ASSERT_EQ(generated.size(), static_cast<std::size_t>(K_LONG_TOKENS));
    ASSERT_LT(static_cast<std::int64_t>(prompt.size()) + K_LONG_TOKENS,
              SaTest::K_WIDE_DECODER_WINDOW);

    // No row of the wide cache is written twice, and the mask leaves the last T readable.
    RingDriver unwrapped;
    unwrapped.bind(*engine, *wide.cached.step);
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    std::vector<TokenID> text = prompt;
    std::vector<TokenID> expected;
    for (std::size_t position = 0; expected.size() < generated.size(); ++position)
    {
        unwrapped.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);
        if (position + 1 < text.size())
            continue;

        const float* const row = logits_of(*wide.cached.step, wide.output);
        ASSERT_NE(row, nullptr);
        expected.push_back(sampler.sample(LogitSpan(row, scores), random));
        text.push_back(expected.back());
    }

    EXPECT_EQ(generated, expected);
}

/** @brief Tests that a context that slides takes one run per token: 14 where a table takes 56. */
TEST_F(Integration_RotaryGeneration, AContextThatSlidesTakesOneRunPerToken)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);

    const std::vector<TokenID> generated =
        continue_with(*deep.cached.generator, {1, 2, 3}, K_SLIDING_TOKENS);

    // Three runs feed the prompt and one feeds each token but the last, which nothing reads.
    EXPECT_EQ(generated.size(), static_cast<std::size_t>(K_SLIDING_TOKENS));
    EXPECT_EQ(deep.cached.step->step().run_count(), 14u);
    EXPECT_EQ(deep.cached.step->step().compile_count(), 1u);
}

/** @brief Tests that a row written twice holds the second write and its neighbour its own. */
TEST_F(Integration_RotaryGeneration, ASlotWrittenTwiceHoldsTheSecondRow)
{
    build(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, shallow);
    ASSERT_NE(shallow.cached.step, nullptr);

    RotaryCachedPath once;
    compile_position(shallow, once);
    ASSERT_NE(once.step, nullptr);
    RingDriver twice_fed;
    twice_fed.bind(*engine, *shallow.cached.step);
    RingDriver once_fed;
    once_fed.bind(*engine, *once.step);

    const std::vector<TokenID> text = fed_text(static_cast<std::size_t>(K_WINDOW));
    for (std::size_t position = 0; position < text.size(); ++position)
        twice_fed.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);

    // The one block carries its keys, then its values; row 1 is the neighbour of row 0.
    const std::size_t dim = static_cast<std::size_t>(SaTest::K_DECODER_DIM);
    ASSERT_EQ(shallow.incremental->carried.values().size(), K_CACHES);
    std::vector<float> neighbours[K_CACHES];
    for (std::size_t cache = 0; cache < K_CACHES; ++cache)
    {
        const Graph::ValueID leaf = shallow.incremental->carried.values()[cache].leaf;
        const float* const rows = shallow.cached.pool->tensor(leaf).data_as<float>();
        neighbours[cache].assign(rows + dim, rows + 2 * dim);
    }

    // Position T wraps to row 0, which the other step writes once, into zeros.
    const TokenID last = 29;
    twice_fed.feed(last, K_WINDOW, K_WINDOW);
    once_fed.feed(last, K_WINDOW, K_WINDOW);

    // The keys and the values of the first block read their own position alone.
    for (std::size_t cache = 0; cache < K_CACHES; ++cache)
    {
        const Graph::ValueID leaf = shallow.incremental->carried.values()[cache].leaf;
        const float* const written = shallow.cached.pool->tensor(leaf).data_as<float>();
        const float* const alone = once.pool->tensor(leaf).data_as<float>();
        for (std::size_t channel = 0; channel < dim; ++channel)
        {
            EXPECT_EQ(written[channel], alone[channel])
                << "cache " << cache << " channel " << channel;
            EXPECT_EQ(written[dim + channel], neighbours[cache][channel])
                << "cache " << cache << " channel " << channel;
        }
    }

    once.release();
}

/** @brief Tests that a second prompt gives what a generator that saw no first prompt gives. */
TEST_F(Integration_RotaryGeneration, ASecondPromptMatchesAFreshGenerator)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.generator, nullptr);

    const std::vector<TokenID> second{20, 3};
    static_cast<void>(continue_with(*deep.cached.generator, {5, 9, 13, 2}, K_LONG_TOKENS));
    const std::vector<TokenID> reused =
        continue_with(*deep.cached.generator, second, K_LONG_TOKENS);

    RotaryCachedPath fresh;
    make_cached_path(deep, fresh);
    ASSERT_NE(fresh.generator, nullptr);
    EXPECT_EQ(reused, continue_with(*fresh.generator, second, K_LONG_TOKENS));

    fresh.release();
}

/** @brief Tests that a ring without its angles and a step fed both leaves are refused. */
TEST_F(Integration_RotaryGeneration, AStepOrARingThatDoesNotFitIsRefused)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, deep);
    ASSERT_NE(deep.cached.step, nullptr);

    CachedStepLeaves silent = rotary_block_leaves();
    silent.ring.angles = nullptr;
    SA_EXPECT_FAILURE(CachedGenerator::make(*engine, *deep.cached.step, deep.output, silent),
                      StatusCode::InvalidDescription, "no function", "'rotary_cosine'");

    build_model(K_MIXED_DECODER_MODEL, shallow);
    ASSERT_NE(shallow.state, nullptr);
    compile_position(shallow, shallow.cached);
    ASSERT_NE(shallow.cached.step, nullptr);
    SA_EXPECT_FAILURE(
        CachedGenerator::make(*engine, *shallow.cached.step, shallow.output, rotary_block_leaves()),
        StatusCode::InvalidDescription, "both", "'position'", "'cache_slot'");
}
