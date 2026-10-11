/**************************************************************************/
/* test_decoder_from_files.cpp                                            */
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
 * @file test_decoder_from_files.cpp
 * @brief Tests that a decoder written as a model file builds, ties its head and trains a step.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <common/test_status.hpp>
#include <common/test_utils.hpp>
#include "decoder_model.hpp"

using namespace SushiAI;

// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the rows of the batch the graphs are traced for. */
    constexpr std::int64_t K_BATCH = 2;

    /**
     * @brief Returns the parameter elements of a decoder with a tied head, by hand.
     *
     * A block holds two normalisations (2 D each), qkv (3 D^2 + 3 D), proj (D^2 + D) and
     * a feed-forward of width 4 D (8 D^2 + 5 D). The head adds its normalisation only.
     */
    constexpr std::int64_t decoder_elements(std::int64_t vocabulary, std::int64_t window,
                                            std::int64_t dim, std::int64_t blocks)
    {
        const std::int64_t block = 12 * dim * dim + 13 * dim;
        return vocabulary * dim + window * dim + blocks * block + 2 * dim;
    }

    /** @brief Provides the engine the state is created on. */
    class Integration_DecoderFromFiles : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the model file states an int32 input and a task this build can read. */
TEST(Integration_DecoderModelFile, TheFileStatesAnInt32Input)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();

    EXPECT_EQ(spec.input_shape, (Shape{SaTest::K_DECODER_WINDOW}));
    EXPECT_EQ(spec.input_dtype, DType::Int32);
    EXPECT_FALSE(spec.task.is_null());
}

/** @brief Tests that the tied head adds no weight and that the element count is the hand's. */
TEST_F(Integration_DecoderFromFiles, TheTiedHeadSharesTheTokenTable)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_decoder(spec));

    Train::GraphOptions options;
    options.dtype = spec.input_dtype;
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(
                                model, Shape{K_BATCH, SaTest::K_DECODER_WINDOW}, options));
    static_cast<void>(probe);

    std::vector<std::string> names;
    NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter&)
    {
        names.push_back(name);
    });

    EXPECT_NE(std::find(names.begin(), names.end(), "0.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "1.weight"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "4.norm.weight"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "4.linear.weight"), names.end());
    EXPECT_EQ(NN::parameter_elements(model),
              decoder_elements(SaTest::K_DECODER_VOCABULARY, SaTest::K_DECODER_WINDOW,
                               SaTest::K_DECODER_DIM, SaTest::K_DECODER_BLOCKS));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_FAILURE(state.tensor("4.linear.weight"), StatusCode::InvalidDescription,
                      "4.linear.weight");
}

/** @brief Tests that the next-token training graph of the file binds one slot per parameter. */
TEST_F(Integration_DecoderFromFiles, TheTrainingGraphBindsEveryParameterOnce)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_decoder(spec));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Training::NextTokenObjective> objective,
        Language::Training::NextTokenObjective::make(K_BATCH, SaTest::K_DECODER_WINDOW,
                                                     SaTest::K_DECODER_VOCABULARY));

    Train::GraphOptions options;
    options.dtype = spec.input_dtype;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_training_graph(
                                model, *objective, Shape{K_BATCH, SaTest::K_DECODER_WINDOW},
                                options));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));

    EXPECT_EQ(slots.size(), NN::parameter_count(model));
    EXPECT_EQ(std::count_if(slots.begin(), slots.end(), [](const Optim::ParameterSlot& slot)
    {
        return slot.name == "0.weight";
    }), 1);
}
