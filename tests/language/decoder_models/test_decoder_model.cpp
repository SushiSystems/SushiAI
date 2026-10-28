/**************************************************************************/
/* test_decoder_model.cpp                                                 */
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
 * @file test_decoder_model.cpp
 * @brief Tests that the decoder model files agree with their constants and build.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/decoder_models/decoder_model.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/model_graph.hpp>

using namespace SushiAI;

/** @brief Shortens the namespace of the test helpers, which Test alone hides in a case. */
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the whole number under @p key of @p value; a refusal fails the test run. */
    std::int64_t int_of(const Config::Value& value, std::string_view key)
    {
        return SaTest::require_value(SaTest::require_value(value.at(key, key)).as_int(key));
    }

    /** @brief Counts the rows of @p spec whose type is @p type. */
    std::int64_t count_rows(const Config::ModelSpec& spec, std::string_view type)
    {
        std::int64_t count = 0;
        for (const Config::LayerSpec& layer : spec.layers)
        {
            if (layer.type == type)
                ++count;
        }

        return count;
    }
} // namespace

/** @brief Tests that the token table of the decoder spans the vocabulary at the width. */
TEST(Small_DecoderModel, TheTokenTableHasTheVocabularyAndTheWidth)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();

    ASSERT_FALSE(spec.layers.empty());
    const Config::LayerSpec& table = spec.layers.front();
    EXPECT_EQ(table.type, "embedding");
    EXPECT_EQ(int_of(table.attributes, "num_embeddings"),
              
              SaTest::K_DECODER_VOCABULARY);
    EXPECT_EQ(int_of(table.attributes, "embedding_dim"),
              
              SaTest::K_DECODER_DIM);
    EXPECT_EQ(count_rows(spec, "transformer_block"), SaTest::K_DECODER_BLOCKS);
    EXPECT_EQ(count_rows(spec, "position_embedding"), 1);
}

/** @brief Tests that the rotary files hold no position table and the wide one its window. */
TEST(Small_DecoderModel, TheRotaryFilesHoldNoPositionTable)
{
    const Config::ModelSpec rotary = SaTest::model_from(SaTest::K_ROTARY_DECODER_MODEL);
    const Config::ModelSpec wide = SaTest::model_from(SaTest::K_WIDE_ROTARY_DECODER_MODEL);
    const Config::ModelSpec shallow = SaTest::model_from(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL);

    EXPECT_EQ(count_rows(rotary, "position_embedding"), 0);
    EXPECT_EQ(count_rows(rotary, "transformer_block"), SaTest::K_DECODER_BLOCKS);
    EXPECT_EQ(rotary.input_shape, (Shape{SaTest::K_DECODER_WINDOW}));
    EXPECT_EQ(wide.input_shape, (Shape{SaTest::K_WIDE_DECODER_WINDOW}));
    EXPECT_EQ(count_rows(shallow, "transformer_block"), 1);
}

/** @brief Tests that every decoder file builds from the rows of nn and of language. */
TEST(Small_DecoderModel, EveryFileBuilds)
{
    for (const std::string_view text :
         {SaTest::K_DECODER_MODEL, SaTest::K_ROTARY_DECODER_MODEL,
          SaTest::K_WIDE_ROTARY_DECODER_MODEL, SaTest::K_SHALLOW_ROTARY_DECODER_MODEL})
    {
        SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph built,
                                SaTest::build_decoder(SaTest::model_from(text)));
        static_cast<void>(built);
    }
}
