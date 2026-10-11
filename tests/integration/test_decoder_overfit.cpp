/**************************************************************************/
/* test_decoder_overfit.cpp                                               */
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
 * @file test_decoder_overfit.cpp
 * @brief Tests that a two-block decoder with a tied head overfits one batch of a token file.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/training_step.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the tokens the model scores, V. */
    constexpr std::int64_t K_VOCABULARY = 64;

    /** @brief Holds the rows of the one batch, N. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the positions of a row, T. */
    constexpr std::int64_t K_LENGTH = 16;

    /** @brief Holds the width of the model. */
    constexpr std::int64_t K_DIM = 32;

    /** @brief Holds the heads of each attention. */
    constexpr std::int64_t K_HEADS = 2;

    /** @brief Holds the width between the two linear layers of a feed-forward. */
    constexpr std::int64_t K_HIDDEN = 64;

    /** @brief Holds the updates the one batch is trained for. */
    constexpr std::int64_t K_STEPS = 400;

    /** @brief Holds the fraction of the first loss the last one must be below. */
    constexpr double K_REMAINING_FRACTION = 0.1;

    /** @brief Bounds how far the first loss may lie from ln(V), the loss of a uniform guess. */
    constexpr double K_FIRST_LOSS_TOLERANCE = 0.5;

    /** @brief Holds the norm every update's gradients are clipped to. */
    constexpr double K_CLIP_NORM = 1.0;

    /** @brief Holds a decoder of two blocks whose head scores with the token table. */
    struct Decoder
    {
        /** @brief Holds the [V, dim] token table, which the head is tied to. */
        NN::Embedding tokens{};

        /** @brief Holds the learned position table. */
        Language::Blocks::PositionEmbedding positions;

        /** @brief Holds the first block. */
        Language::Blocks::TransformerBlock first;

        /** @brief Holds the second block. */
        Language::Blocks::TransformerBlock second;

        /** @brief Holds the final normalisation and the tied scores. */
        Language::Blocks::LanguageHead head;

        /** @brief Builds the five children and ties the head to the token table. */
        Decoder()
            : positions(K_LENGTH, K_DIM),
              first(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                    NN::K_LAYER_NORM_EPSILON),
              second(K_DIM, K_HEADS, K_HIDDEN, Language::Blocks::AttentionMask::Causal,
                     NN::K_LAYER_NORM_EPSILON),
              head(K_DIM, K_VOCABULARY, NN::K_LAYER_NORM_EPSILON)
        {
            tokens.num_embeddings = K_VOCABULARY;
            tokens.embedding_dim = K_DIM;
            head.tie_to(tokens.weight);
        }

        /** @brief Forbids a copy, whose head would stay tied to this object's table. */
        Decoder(const Decoder&) = delete;

        /** @brief Forbids copy assignment, for the same reason. */
        Decoder& operator=(const Decoder&) = delete;

        /** @brief Lists the five children in the order they are traced. */
        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("tokens", &Decoder::tokens),
                NN::child("positions", &Decoder::positions),
                NN::child("first", &Decoder::first),
                NN::child("second", &Decoder::second),
                NN::child("head", &Decoder::head),
            };
        }

        /** @brief Traces [N, T] ids to [N, T, V] logits. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID ids)
        {
            Graph::ValueID x = NN::trace_layer(tokens, trace, ids);
            x = NN::trace_layer(positions, trace, x);
            x = NN::trace_layer(first, trace, x);
            x = NN::trace_layer(second, trace, x);
            return NN::trace_layer(head, trace, x);
        }
    };

    /** @brief Owns one file under the system's temporary directory and removes it. */
    class ScratchFile final
    {
        public:
            /** @brief Names the file; nothing is written yet. */
            explicit ScratchFile(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {

            }

            /** @brief Removes the file when it exists. */
            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ScratchFile(const ScratchFile&) = delete;
            ScratchFile& operator=(const ScratchFile&) = delete;
            ScratchFile(ScratchFile&&) = delete;
            ScratchFile& operator=(ScratchFile&&) = delete;

            /** @brief Returns the path of the file. */
            [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

        private:
            /// @brief Holds the path this object removes.
            std::filesystem::path path_;
    };

    /** @brief Returns K_BATCH windows of K_LENGTH + 1 ids, id i being 7 i^2 + 3 i + 1 mod V. */
    std::vector<Language::Text::TokenID> make_ids()
    {
        const std::int64_t count = K_BATCH * (K_LENGTH + 1);
        std::vector<Language::Text::TokenID> ids;
        ids.reserve(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i)
            ids.push_back(static_cast<Language::Text::TokenID>((7 * i * i + 3 * i + 1)
                                                               % K_VOCABULARY));
        return ids;
    }

    /** @brief Returns whether the slot named @p name decays; fails the test when none has it. */
    bool decays(const std::vector<Optim::ParameterSlot>& slots, const std::string& name)
    {
        for (const Optim::ParameterSlot& slot : slots)
        {
            if (slot.name == name)
                return slot.decays;
        }

        ADD_FAILURE() << "no slot is named " << name;
        return false;
    }

    /** @brief Provides the engine the dataset, the state and the step are built on. */
    class Integration_DecoderOverfit : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the loss of one fixed batch falls below a tenth of its start. */
TEST_F(Integration_DecoderOverfit, ATwoBlockDecoderOverfitsOneBatch)
{
    const ScratchFile file("sushiai_decoder_overfit.satk");
    const std::vector<Language::Text::TokenID> ids = make_ids();
    SA_EXPECT_OK(Language::Datasets::write_token_file(
        file.path(),
        SushiRuntime::span<const Language::Text::TokenID>(ids.data(), ids.size()),
        static_cast<std::int32_t>(K_VOCABULARY)));

    SA_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Language::Datasets::NextTokenCollator> collator,
        Language::Datasets::NextTokenCollator::make(K_LENGTH,
                                                    static_cast<std::int32_t>(K_VOCABULARY)));
    Language::Datasets::TokenDatasetOptions dataset_options;
    dataset_options.batch_size = K_BATCH;
    dataset_options.stride = K_LENGTH + 1;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::TokenDataset> dataset,
        Language::Datasets::TokenDataset::make(*engine, file.path(), std::move(collator),
                                               dataset_options));
    ASSERT_EQ(dataset->batch_count(), 1u);

    Decoder model;
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_BATCH, K_LENGTH, K_VOCABULARY));
    Train::GraphOptions graph_options;
    graph_options.dtype = DType::Int32;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(model, *objective,
                                                        Shape{K_BATCH, K_LENGTH},
                                                        graph_options));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));

    // The head has no weight of its own, and only the linear weights decay.
    SA_EXPECT_FAILURE(state.tensor("head.linear.weight"), StatusCode::InvalidDescription,
                      "head.linear.weight");
    EXPECT_FALSE(decays(slots, "tokens.weight"));
    EXPECT_FALSE(decays(slots, "positions.weight"));
    EXPECT_FALSE(decays(slots, "head.norm.weight"));
    EXPECT_FALSE(decays(slots, "first.attn.qkv.bias"));
    EXPECT_TRUE(decays(slots, "first.attn.qkv.weight"));

    NN::InitPolicy policy;
    policy.family = NN::InitFamily::Xavier;
    policy.nonlinearity = NN::Nonlinearity::Linear;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    Optim::AdamWOptions optimizer_options;
    optimizer_options.weight_decay = 0.01;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options));
    SA_EXPECT_OK(optimizer->enable_gradient_clip(K_CLIP_NORM));

    Optim::CosineSchedule schedule;
    schedule.base_rate = 3e-3;
    schedule.warmup_iterations = 20;
    schedule.total_iterations = K_STEPS;
    schedule.final_rate = 3e-4;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::TrainingStep> step,
                            Train::TrainingStep::make(*engine, built.graph, pool, *optimizer,
                                                      {}));

    // The one batch is fed once: no task is recorded yet, so the host may write the leaves.
    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(0, batch));
    SushiBLAS::Tensor& inputs = pool.tensor(built.spec.inputs);
    SA_EXPECT_OK(Train::check_feed(batch.inputs, inputs, "inputs"));
    const std::int32_t* const fed = batch.inputs.data_as<std::int32_t>();
    std::int32_t* const leaf = inputs.data_as<std::int32_t>();
    for (std::int64_t i = 0; i < inputs.num_elements; ++i)
        leaf[i] = fed[i];

    const Data::BatchTarget* const labels =
        Data::find_target(batch.targets, Data::K_ROLE_LABELS);
    ASSERT_NE(labels, nullptr);
    SA_EXPECT_OK(Language::Training::fill_next_token_targets(
        labels->values, K_VOCABULARY, pool.tensor(built.spec.targets)));

    double first_loss = 0.0;
    double last_loss = 0.0;
    for (std::int64_t update = 0; update < K_STEPS; ++update)
    {
        Optim::apply_step_schedule(schedule, *optimizer);
        SA_EXPECT_OK(step->run());

        last_loss = static_cast<double>(pool.tensor(built.spec.loss).data_as<float>()[0]);
        ASSERT_TRUE(std::isfinite(last_loss)) << "update " << update;
        if (update == 0)
            first_loss = last_loss;
    }

    EXPECT_NEAR(first_loss, std::log(static_cast<double>(K_VOCABULARY)),
                K_FIRST_LOSS_TOLERANCE);
    EXPECT_LT(last_loss, K_REMAINING_FRACTION * first_loss);
    EXPECT_LT(Language::Training::perplexity(last_loss),
              Language::Training::perplexity(first_loss));
    EXPECT_EQ(step->step().compile_count(), 1u);
    EXPECT_EQ(optimizer->step_count(), static_cast<std::uint64_t>(K_STEPS));
}
