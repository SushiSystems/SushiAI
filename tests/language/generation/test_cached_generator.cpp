/**************************************************************************/
/* test_cached_generator.cpp                                             */
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
 * @file test_cached_generator.cpp
 * @brief Tests CachedGenerator against WindowGenerator over decoders with a learned table and with
 *        rotary positions, and over an in-memory one-position step.
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
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
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
#include "position_feed_test_support.hpp"
#include <support/tolerance.hpp>

using namespace SushiAI;
using Language::Generation::CachedGenerator;
using Language::Generation::CachedStepLeaves;
using Language::Generation::GreedySampler;
using Language::Generation::LogitSpan;
using Language::Generation::RandomGenerator;
using Language::Generation::RingStepLeaves;
using Language::Generation::WindowGenerator;
using Language::Text::TokenID;
// SaTest stays reachable inside a TEST_F body; see tests/language/generation/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = Language::Generation::Support;

namespace
{
    /** @brief Holds the seed of every generator of these tests; a greedy sampler draws nothing. */
    constexpr std::uint64_t K_SEED = 20261011u;

    /** @brief Holds T, the rows of the cache under test. */
    constexpr std::int64_t K_WINDOW = SaTest::K_DECODER_WINDOW;

    /** @brief Holds the tokens of a run whose context stays shorter than the window. */
    constexpr std::int64_t K_FITTING_TOKENS = 4;

    /** @brief Holds the tokens of a run whose context reaches the window and slides. */
    constexpr std::int64_t K_SLIDING_TOKENS = 12;

    /** @brief Holds the tokens of a run that slides for more than three windows. */
    constexpr std::int64_t K_LONG_TOKENS = 3 * K_WINDOW + 2;

    /** @brief Holds the ids of a fed text that slides for more than three windows. */
    constexpr std::size_t K_FED_IDS = 30;

    /**
     * @brief Holds the absolute tolerance between a logit of the two float32 passes: 2048 times the
     *        float unit roundoff, the smallest power of two not tighter than the earlier 1e-4f.
     */
    constexpr float K_LOGIT_TOLERANCE =
        static_cast<float>(2048.0 * TestSupport::unit_roundoff<float>());

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

    /** @brief Views ids as the span a generator reads. */
    SushiRuntime::span<const TokenID> span_of(const std::vector<TokenID>& ids)
    {
        return SushiRuntime::span<const TokenID>(ids.data(), ids.size());
    }

    /**
     * @brief Returns the leaves the incremental forms of the blocks declare: a position for a
     *        learned table and a cache slot with angles for a ring.
     */
    CachedStepLeaves block_leaves()
    {
        return CachedStepLeaves{
            Language::Blocks::K_POSITION_LEAF, Language::Blocks::K_ATTENTION_MASK_LEAF,
            Language::Blocks::K_MASKED_ATTENTION_SCORE,
            RingStepLeaves{Language::Blocks::K_CACHE_SLOT_LEAF,
                           Language::Blocks::K_ROTARY_COSINE_LEAF,
                           Language::Blocks::K_ROTARY_SINE_LEAF,
                           &Language::Blocks::write_rotary_row}};
    }

    /** @brief Returns the leaves of the blocks with a mask leaf the step is not fed. */
    CachedStepLeaves misnamed_mask_leaves()
    {
        CachedStepLeaves leaves = block_leaves();
        leaves.mask = "scores";
        return leaves;
    }

    /** @brief Returns the leaves of the blocks with a ring that names no angle function. */
    CachedStepLeaves silent_ring_leaves()
    {
        CachedStepLeaves leaves = block_leaves();
        leaves.ring.angles = nullptr;
        return leaves;
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

    /** @brief Expects each of @p scores logits of @p computed to lie within the tolerance of @p expected. */
    void expect_logits_near(const float* computed, const float* expected, std::size_t scores)
    {
        for (std::size_t token = 0; token < scores; ++token)
            EXPECT_NEAR(computed[token], expected[token], K_LOGIT_TOLERANCE) << "token " << token;
    }

    /** @brief Names a decoder and the run that slides its window. */
    struct DecoderCase
    {
        std::string_view model;       /**< The model file of the decoder. */
        std::int64_t sliding_tokens; /**< The tokens of a run whose context slides. */
    };

    /** @brief Names a step that does not fit and what the refusal names. */
    struct RefusalCase
    {
        std::string_view model;   /**< The model file of the decoder. */
        bool window_step;         /**< Whether the step is the whole window, which carries nothing. */
        CachedStepLeaves leaves;  /**< The leaves given to make. */
        std::string_view first;   /**< A substring the message holds. */
        std::string_view second;  /**< Another substring the message holds. */
    };

    /** @brief Returns the learned-table decoder, the two-block ring and the one-block ring. */
    std::vector<DecoderCase> decoder_cases()
    {
        return {{SaTest::K_DECODER_MODEL, K_SLIDING_TOKENS},
                {SaTest::K_ROTARY_DECODER_MODEL, K_LONG_TOKENS},
                {SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, K_LONG_TOKENS}};
    }

    /**
     * @brief Returns the decoders whose slides the window generator reproduces exactly: the table
     *        and the one-block ring.
     *
     * Past the first slide a two-block ring reads keys cached from ids the window has dropped, so
     * its oracle is the wide cache instead; test_cached_generator.md gives the reason.
     */
    std::vector<DecoderCase> window_exact_cases()
    {
        return {{SaTest::K_DECODER_MODEL, K_SLIDING_TOKENS},
                {SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, K_LONG_TOKENS}};
    }

    /** @brief Returns a step that carries nothing, a misnamed leaf, a silent ring and a step fed both. */
    std::vector<RefusalCase> refusal_cases()
    {
        return {{SaTest::K_DECODER_MODEL, true, block_leaves(), "keeps nothing between runs", ""},
                {SaTest::K_DECODER_MODEL, false, misnamed_mask_leaves(), "'scores'",
                 "'attention_mask'"},
                {SaTest::K_ROTARY_DECODER_MODEL, false, silent_ring_leaves(), "no function",
                 "'rotary_cosine'"},
                {K_MIXED_DECODER_MODEL, false, block_leaves(), "both", "'cache_slot'"}};
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
    struct CachedPath
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
        std::unique_ptr<NN::ModelGraph> model;              /**< The rows of the model file. */
        std::unique_ptr<Train::TracedStep> incremental;     /**< The one-position graph. */
        std::string output;                                 /**< The name of the logits. */
        std::unique_ptr<NN::ModelState> state;              /**< The initialised parameters. */
        std::unique_ptr<Graph::TensorPool> window_pool;     /**< Every value of the window. */
        std::unique_ptr<Train::EvaluationStep> window_step; /**< The compiled window. */
        std::unique_ptr<WindowGenerator> window;            /**< The reference generator. */
        CachedPath cached;                                  /**< The generator under test. */

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

    /** @brief Builds decoders and both generators over each. */
    class CachedGenerationTest : public SaTest::EngineTest
    {
        protected:
            /** @brief Releases every decoder before the engine they were built on. */
            void TearDown() override
            {
                wide.release();
                decoder.release();
                SaTest::EngineTest::TearDown();
            }

            /** @brief Builds the model of @p text, its state and the graph of one position. */
            void build_model(std::string_view text, Decoder& target)
            {
                const Config::ModelSpec spec = SaTest::model_from(text);
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built, SaTest::build_decoder(spec));
                target.model = std::make_unique<NN::ModelGraph>(std::move(built));

                Train::GraphOptions options;
                options.dtype = spec.input_dtype;
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep sliced,
                    Train::build_incremental_graph(*target.model, Shape{1, 1}, options));
                target.incremental = std::make_unique<Train::TracedStep>(std::move(sliced));
                target.output = target.incremental->outputs.all().front().first;

                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *target.model));
                target.state = std::make_unique<NN::ModelState>(std::move(created));
                SA_EXPECT_OK(NN::initialize(backend(), *target.state));
            }

            /** @brief Compiles the one-position graph of @p source into @p path. */
            void compile_position(Decoder& source, CachedPath& path)
            {
                path.pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<Train::EvaluationStep> compiled,
                    Train::EvaluationStep::make(backend(), *source.incremental, *source.state,
                                                *path.pool));
                path.step = std::move(compiled);
            }

            /** @brief Compiles the one-position graph and builds the cached generator. */
            void make_cached_path(Decoder& source, CachedPath& path)
            {
                compile_position(source, path);
                ASSERT_NE(path.step, nullptr);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<CachedGenerator> made,
                    CachedGenerator::make(*engine, *path.step, source.output, block_leaves()));
                path.generator = std::move(made);
            }

            /** @brief Builds every part of the decoder @p text describes over a window of T. */
            void build(std::string_view text, Decoder& target)
            {
                build_model(text, target);
                ASSERT_NE(target.state, nullptr);

                Train::GraphOptions options;
                options.dtype = DType::Int32;
                SA_ASSERT_OK_AND_ASSIGN(
                    Train::TracedStep whole,
                    Train::build_evaluation_graph(*target.model, Shape{1, K_WINDOW}, options));

                target.window_pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<Train::EvaluationStep> compiled,
                    Train::EvaluationStep::make(backend(), whole, *target.state,
                                                *target.window_pool));
                target.window_step = std::move(compiled);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<WindowGenerator> made,
                    WindowGenerator::make(*engine, *target.window_step, target.output));
                target.window = std::move(made);

                make_cached_path(target, target.cached);
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

            /** @brief Reports whether every entry of the state of @p model is a parameter named for no angle table. */
            bool holds_parameters_only(NN::ModelGraph& model)
            {
                const Result<NN::ModelState> again = NN::ModelState::create(*engine, model);
                if (!again.has_value())
                    return false;

                return std::all_of(again.value().entries().begin(), again.value().entries().end(),
                                   [](const NN::StateEntry& entry)
                                   {
                                       return entry.kind == NN::StateKind::Parameter &&
                                              entry.name.find("rotary") == std::string::npos;
                                   });
            }

            /**
             * @brief Feeds a text through a ring step by hand, and expects its logits at every
             *        position to equal those of the window generator over the same text.
             */
            void expect_ring_logits_near_window(Decoder& source)
            {
                CachedPath driven;
                compile_position(source, driven);
                ASSERT_NE(driven.step, nullptr);
                RingDriver ring;
                ring.bind(*engine, *driven.step);

                const std::vector<TokenID> text = fed_text(K_FED_IDS);
                const std::size_t length = static_cast<std::size_t>(K_WINDOW);
                const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
                for (std::size_t position = 0; position < text.size(); ++position)
                {
                    ring.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);

                    // The window generator feeds the last T ids of the text so far.
                    const std::vector<TokenID> so_far(
                        text.begin(), text.begin() + static_cast<std::ptrdiff_t>(position + 1));
                    static_cast<void>(continue_with(*source.window, so_far, 1));
                    const std::size_t held = std::min(position + 1, length);

                    const float* const whole = logits_of(*source.window_step, source.output);
                    const float* const one = logits_of(*driven.step, source.output);
                    ASSERT_NE(whole, nullptr);
                    ASSERT_NE(one, nullptr);
                    expect_logits_near(one, whole + (held - 1) * scores, scores);
                }

                driven.release();
            }

            /**
             * @brief Returns the tokens a wide cache gives, which is fed by hand and read through
             *        its last T positions only, so no row of it is written twice.
             */
            std::vector<TokenID> wide_cache_tokens(const std::vector<TokenID>& prompt,
                                                   std::size_t count)
            {
                RingDriver unwrapped;
                unwrapped.bind(*engine, *wide.cached.step);
                const GreedySampler sampler;
                RandomGenerator random(K_SEED);
                const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
                std::vector<TokenID> text = prompt;
                std::vector<TokenID> made;
                for (std::size_t position = 0; made.size() < count; ++position)
                {
                    unwrapped.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);

                    // A token is sampled once the text so far is all fed.
                    if (position + 1 >= text.size())
                    {
                        const float* const row = logits_of(*wide.cached.step, wide.output);
                        if (row == nullptr)
                            return made;

                        made.push_back(sampler.sample(LogitSpan(row, scores), random));
                        text.push_back(made.back());
                    }
                }

                return made;
            }

            /// @brief Holds the decoder under test.
            Decoder decoder;

            /// @brief Holds the two-block decoder over a window no test fills.
            Decoder wide;
    };

    /** @brief Builds the decoder of each row, with its window path and its cached path. */
    class Medium_CachedGeneration : public CachedGenerationTest,
                                    public ::testing::WithParamInterface<DecoderCase>
    {
        protected:
            /** @brief Builds the decoder of the row. */
            void SetUp() override
            {
                CachedGenerationTest::SetUp();
                build(GetParam().model, decoder);
            }
    };

    /** @brief Builds the decoders of window_exact_cases(), for the runs that slide. */
    class Medium_CachedSlide : public Medium_CachedGeneration
    {

    };

    /** @brief Builds the decoders the ring tests feed by hand. */
    class Medium_CachedRing : public CachedGenerationTest
    {

    };

    /** @brief Builds the decoder and the step of each refusal row. */
    class Medium_CachedRefusal : public CachedGenerationTest,
                                 public ::testing::WithParamInterface<RefusalCase>
    {
        protected:
            /**
             * @brief Builds the decoder of @p row and returns the step it offers to make: the
             *        whole window or the one-position step.
             */
            Train::EvaluationStep* step_of(const RefusalCase& row)
            {
                if (row.window_step)
                {
                    build(row.model, decoder);
                    return decoder.window_step.get();
                }

                build_model(row.model, decoder);
                compile_position(decoder, decoder.cached);
                return decoder.cached.step.get();
            }
    };

    /** @brief Gives each test a host engine for the in-memory step. */
    class Small_CachedGeneration : public Support::HostFeedTest
    {

    };
} // namespace

/** @brief Tests that a generator is built over the one-position step of each decoder. */
TEST_P(Medium_CachedGeneration, AStepIsAcceptedWithTheLeavesItIsFed)
{
    SA_EXPECT_OK(CachedGenerator::make(*engine, *decoder.cached.step, decoder.output,
                                       block_leaves()));
}

/** @brief Tests that the state holds parameters only: no cache, no position table, no angle table. */
TEST_P(Medium_CachedGeneration, TheStateHoldsParametersOnly)
{
    EXPECT_TRUE(holds_parameters_only(*decoder.model));
}

/** @brief Tests that a prompt that fits gives the tokens of the window generator. */
TEST_P(Medium_CachedGeneration, APromptThatFitsGivesTheTokensOfTheWindowGenerator)
{
    const std::vector<TokenID> prompt{1, 2, 3};

    const std::vector<TokenID> expected = continue_with(*decoder.window, prompt, K_FITTING_TOKENS);
    const std::vector<TokenID> generated =
        continue_with(*decoder.cached.generator, prompt, K_FITTING_TOKENS);

    ASSERT_EQ(expected.size(), static_cast<std::size_t>(K_FITTING_TOKENS));
    EXPECT_EQ(generated, expected);
}

/** @brief Tests that the logits after a prompt equal those of the window at its last position. */
TEST_P(Medium_CachedGeneration, TheLogitsEqualThoseOfTheWindowAtTheLastPosition)
{
    const std::vector<TokenID> prompt{4, 7, 1, 30, 2};
    static_cast<void>(continue_with(*decoder.window, prompt, 1));
    static_cast<void>(continue_with(*decoder.cached.generator, prompt, 1));

    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    const float* const whole = logits_of(*decoder.window_step, decoder.output);
    const float* const one = logits_of(*decoder.cached.step, decoder.output);
    ASSERT_NE(whole, nullptr);
    ASSERT_NE(one, nullptr);

    expect_logits_near(one, whole + (prompt.size() - 1) * scores, scores);
}

/** @brief Tests that a context that reaches the window and slides gives the window generator's tokens. */
TEST_P(Medium_CachedSlide, AContextThatSlidesGivesTheTokensOfTheWindowGenerator)
{
    const std::vector<TokenID> prompt{1, 2, 3};
    const std::int64_t tokens = GetParam().sliding_tokens;

    const std::vector<TokenID> expected = continue_with(*decoder.window, prompt, tokens);
    const std::vector<TokenID> generated =
        continue_with(*decoder.cached.generator, prompt, tokens);

    ASSERT_EQ(expected.size(), static_cast<std::size_t>(tokens));
    EXPECT_EQ(generated, expected);
}

/** @brief Tests that a second prompt gives what a generator that saw no first prompt gives. */
TEST_P(Medium_CachedGeneration, ASecondPromptMatchesAFreshGenerator)
{
    const std::vector<TokenID> first{5, 9, 13, 2};
    const std::vector<TokenID> second{20, 3};
    const std::int64_t tokens = GetParam().sliding_tokens;
    static_cast<void>(continue_with(*decoder.cached.generator, first, tokens));
    const std::vector<TokenID> reused = continue_with(*decoder.cached.generator, second, tokens);

    CachedPath fresh;
    make_cached_path(decoder, fresh);
    ASSERT_NE(fresh.generator, nullptr);
    const std::vector<TokenID> untouched = continue_with(*fresh.generator, second, tokens);

    EXPECT_EQ(reused, untouched);
    EXPECT_EQ(reused, continue_with(*decoder.window, second, tokens));

    fresh.release();
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_CachedGeneration, ::testing::ValuesIn(decoder_cases()));
INSTANTIATE_TEST_SUITE_P(Rows, Medium_CachedSlide, ::testing::ValuesIn(window_exact_cases()));

/** @brief Tests that a step that carries nothing, a leaf that is absent and a ring that does not fit are refused. */
TEST_P(Medium_CachedRefusal, AStepOrARingThatDoesNotFitIsRefused)
{
    const RefusalCase& row = GetParam();
    Train::EvaluationStep* const step = step_of(row);
    ASSERT_NE(step, nullptr);

    SA_EXPECT_FAILURE(CachedGenerator::make(*engine, *step, decoder.output, row.leaves),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_CachedRefusal, ::testing::ValuesIn(refusal_cases()));

/** @brief Tests the logits of a one-block ring against the window's at every fed position. */
TEST_F(Medium_CachedRing, TheLogitsOfAOneBlockRingEqualTheWindowsAtEveryPosition)
{
    build(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, decoder);
    ASSERT_NE(decoder.window_step, nullptr);

    expect_ring_logits_near_window(decoder);
}

/** @brief Tests a ring of T rows against a cache of 64 rows read through the last T alone. */
TEST_F(Medium_CachedRing, ARingEqualsAWideCacheReadThroughItsLastPositions)
{
    build(SaTest::K_ROTARY_DECODER_MODEL, decoder);
    ASSERT_NE(decoder.cached.generator, nullptr);
    build_model(SaTest::K_WIDE_ROTARY_DECODER_MODEL, wide);
    ASSERT_NE(wide.state, nullptr);
    copy_state(*decoder.state, *wide.state);
    compile_position(wide, wide.cached);
    ASSERT_NE(wide.cached.step, nullptr);

    const std::vector<TokenID> prompt{1, 2, 3};
    const std::vector<TokenID> generated =
        continue_with(*decoder.cached.generator, prompt, K_LONG_TOKENS);
    ASSERT_EQ(generated.size(), static_cast<std::size_t>(K_LONG_TOKENS));
    ASSERT_LT(static_cast<std::int64_t>(prompt.size()) + K_LONG_TOKENS,
              SaTest::K_WIDE_DECODER_WINDOW);

    EXPECT_EQ(generated, wide_cache_tokens(prompt, generated.size()));
}

/** @brief Tests that a row written twice holds the second write: its logits are those of a row written once. */
TEST_F(Medium_CachedRing, ASlotWrittenTwiceHoldsTheSecondRow)
{
    build(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL, decoder);
    ASSERT_NE(decoder.cached.step, nullptr);

    CachedPath once;
    compile_position(decoder, once);
    ASSERT_NE(once.step, nullptr);
    RingDriver twice_fed;
    twice_fed.bind(*engine, *decoder.cached.step);
    RingDriver once_fed;
    once_fed.bind(*engine, *once.step);

    const std::vector<TokenID> text = fed_text(static_cast<std::size_t>(K_WINDOW));
    for (std::size_t position = 0; position < text.size(); ++position)
        twice_fed.feed(text[position], static_cast<std::int64_t>(position), K_WINDOW);

    // Position T wraps to row 0; with a band of one the step reads that row alone.
    const TokenID last = 29;
    twice_fed.feed(last, K_WINDOW, 1);
    once_fed.feed(last, K_WINDOW, 1);

    const std::size_t scores = static_cast<std::size_t>(SaTest::K_DECODER_VOCABULARY);
    const float* const written = logits_of(*decoder.cached.step, decoder.output);
    const float* const alone = logits_of(*once.step, decoder.output);
    ASSERT_NE(written, nullptr);
    ASSERT_NE(alone, nullptr);
    expect_logits_near(written, alone, scores);

    once.release();
}

/** @brief Tests that a prompt longer than the window feeds its last T ids, after one reset. */
TEST_F(Small_CachedGeneration, APromptLongerThanTheWindowKeepsItsLastIDs)
{
    auto made = Support::table_step(Support::K_STEP_WINDOW, Support::K_STEP_VOCABULARY);
    const Language::Generation::InMemoryOnePositionStep* const step = made.get();
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CachedGenerator> generator,
        CachedGenerator::make(engine(), std::move(made), Support::K_STEP_OUTPUT,
                              Support::table_step_leaves()));
    const GreedySampler sampler;
    RandomGenerator random(K_SEED);
    const std::vector<TokenID> prompt{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11};
    std::vector<TokenID> generated;

    SA_EXPECT_OK(generator->generate(span_of(prompt), 1, sampler, random, generated));

    EXPECT_EQ(step->fed_ids(), (std::vector<std::int32_t>{6, 5, 4, 3, 2, 1, 0, 11}));
    EXPECT_EQ(step->resets(), 1u);
    EXPECT_EQ(generated, (std::vector<TokenID>{12}));
}
