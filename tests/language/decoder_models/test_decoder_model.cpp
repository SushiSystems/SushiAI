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
 * @file test_decoder_model.cpp
 * @brief Tests that the decoder model files agree with their constants and build.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/decoder_models/decoder_model.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <support/test_names.hpp>

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

    /** @brief Names one decoder model file by the label a test row prints. */
    struct ModelFile
    {
        const char* label;      /**< The name the row prints. */
        std::string_view text;  /**< The text of the model file. */
    };

    /** @brief Provides the model files that must build. */
    class Small_DecoderModelBuild : public ::testing::TestWithParam<ModelFile>
    {

    };

    /** @brief Returns the four model files, one row each. */
    std::vector<ModelFile> model_files()
    {
        return {{"Learned", SaTest::K_DECODER_MODEL},
                {"Rotary", SaTest::K_ROTARY_DECODER_MODEL},
                {"WideRotary", SaTest::K_WIDE_ROTARY_DECODER_MODEL},
                {"ShallowRotary", SaTest::K_SHALLOW_ROTARY_DECODER_MODEL}};
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

/** @brief Tests that the first row is the token table of the vocabulary and width. */
TEST(Small_DecoderModel, TheTokenTableHasTheVocabularyAndTheWidth)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();

    ASSERT_FALSE(spec.layers.empty());
    const Config::LayerSpec& table = spec.layers.front();
    EXPECT_EQ(table.type, "embedding");
    EXPECT_EQ(int_of(table.attributes, "num_embeddings"), SaTest::K_DECODER_VOCABULARY);
    EXPECT_EQ(int_of(table.attributes, "embedding_dim"), SaTest::K_DECODER_DIM);
}

/** @brief Tests that the learned decoder holds its blocks and one position table. */
TEST(Small_DecoderModel, TheLearnedFileHoldsItsBlocksAndOnePositionTable)
{
    const Config::ModelSpec spec = SaTest::decoder_spec();

    EXPECT_EQ(count_rows(spec, "transformer_block"), SaTest::K_DECODER_BLOCKS);
    EXPECT_EQ(count_rows(spec, "position_embedding"), 1);
}

/** @brief Tests that the rotary file holds no position table, its blocks and its window. */
TEST(Small_DecoderModel, TheRotaryFileHoldsNoPositionTable)
{
    const Config::ModelSpec rotary = SaTest::model_from(SaTest::K_ROTARY_DECODER_MODEL);

    EXPECT_EQ(count_rows(rotary, "position_embedding"), 0);
    EXPECT_EQ(count_rows(rotary, "transformer_block"), SaTest::K_DECODER_BLOCKS);
    EXPECT_EQ(rotary.input_shape, (Shape{SaTest::K_DECODER_WINDOW}));
}

/** @brief Tests that the wide rotary file takes the wide window. */
TEST(Small_DecoderModel, TheWideRotaryFileHasItsWindow)
{
    const Config::ModelSpec wide = SaTest::model_from(SaTest::K_WIDE_ROTARY_DECODER_MODEL);

    EXPECT_EQ(wide.input_shape, (Shape{SaTest::K_WIDE_DECODER_WINDOW}));
}

/** @brief Tests that the shallow rotary file holds one block. */
TEST(Small_DecoderModel, TheShallowRotaryFileHoldsOneBlock)
{
    const Config::ModelSpec shallow = SaTest::model_from(SaTest::K_SHALLOW_ROTARY_DECODER_MODEL);

    EXPECT_EQ(count_rows(shallow, "transformer_block"), 1);
}

/** @brief Tests that a decoder file builds from the rows of nn and of language. */
TEST_P(Small_DecoderModelBuild, AFileBuilds)
{
    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph built,
                            SaTest::build_decoder(SaTest::model_from(GetParam().text)));
    static_cast<void>(built);
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_DecoderModelBuild, ::testing::ValuesIn(model_files()),
                         [](const ::testing::TestParamInfo<ModelFile>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.label);
                         });
