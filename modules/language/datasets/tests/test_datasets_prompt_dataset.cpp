/**************************************************************************/
/* test_datasets_prompt_dataset.cpp                                       */
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
 * @file test_datasets_prompt_dataset.cpp
 * @brief Tests PromptDataset's one batch and the "prompt" data block that encodes a text.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include "datasets_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::Datasets;
using Language::Text::TokenID;

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine the batches are allocated on. */
    class Unit_LanguageDatasetsPromptDataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the row and the stated length of the one batch. */
            void read_row(const Data::Dataset& dataset, std::vector<TokenID>& row,
                          std::int32_t& length)
            {
                Data::Batch batch;
                SA_EXPECT_OK(dataset.read(0, batch));

                const TokenID* const ids = batch.inputs.data_as<TokenID>();
                row.assign(ids, ids + batch.inputs.num_elements);

                const Data::BatchTarget* const stated =
                    Data::find_target(batch.targets, K_ROLE_PROMPT_LENGTH);
                ASSERT_NE(stated, nullptr);
                length = stated->values.data_as<std::int32_t>()[0];
            }
    };
} // namespace

/** @brief Tests that a short prompt fills the front of the row and zeros follow it. */
TEST_F(Unit_LanguageDatasetsPromptDataset, AShortPromptIsFollowedByZeros)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {3, 1, 4, 1, 5}, 8));

    EXPECT_EQ(dataset->batch_size(), 1);
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{8}));
    EXPECT_EQ(dataset->sample_dtype(), DType::Int32);
    EXPECT_EQ(dataset->name(), K_PROMPT_DATA_TYPE);
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, K_ROLE_PROMPT_LENGTH);

    std::vector<TokenID> row;
    std::int32_t length = 0;
    read_row(*dataset, row, length);

    EXPECT_EQ(row, (std::vector<TokenID>{3, 1, 4, 1, 5, 0, 0, 0}));
    EXPECT_EQ(length, 5);
}

/** @brief Tests that a prompt longer than the window keeps its last ids. */
TEST_F(Unit_LanguageDatasetsPromptDataset, ALongPromptKeepsItsLastIDs)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<PromptDataset> dataset,
                            PromptDataset::make(*engine, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 4));

    std::vector<TokenID> row;
    std::int32_t length = 0;
    read_row(*dataset, row, length);

    EXPECT_EQ(row, (std::vector<TokenID>{6, 7, 8, 9}));
    EXPECT_EQ(length, 4);
}

/** @brief Tests that no id, a negative id and a window of no position are each refused. */
TEST_F(Unit_LanguageDatasetsPromptDataset, MakeRefusesAPromptOrAWindowThatDoesNotFit)
{
    SA_EXPECT_FAILURE(PromptDataset::make(*engine, {}, 4), StatusCode::InvalidDescription,
                      "no id");
    SA_EXPECT_FAILURE(PromptDataset::make(*engine, {1, -2}, 4), StatusCode::InvalidDescription,
                      "-2");
    SA_EXPECT_FAILURE(PromptDataset::make(*engine, {1}, 0), StatusCode::InvalidDescription,
                      "window of 0");
}

/** @brief Tests that the data block encodes "text" with the vocabulary file it names. */
TEST_F(Unit_LanguageDatasetsPromptDataset, TheDataBlockEncodesItsText)
{
    const Support::ScratchDirectory scratch("sushiai_prompt_section");
    const std::filesystem::path file = scratch.path() / "bytes.savb";
    SA_ASSERT_OK_AND_ASSIGN(const Language::Text::Vocabulary vocabulary,
                            Language::Text::Vocabulary::make({}));
    SA_EXPECT_OK(Language::Text::write_vocabulary_file(file, vocabulary));

    const std::string stated = "\"vocabulary\": \"" + file.generic_string() + "\"";
    const Config::Value section =
        Config::parse_json("{\"text\": \"hi\", \"window\": 4, " + stated + "}", "<test>");
    const Config::Value without_text =
        Config::parse_json("{\"window\": 4, " + stated + "}", "<test>");
    const Config::Value with_unknown = Config::parse_json(
        "{\"text\": \"hi\", \"window\": 4, \"tokens\": 9, " + stated + "}", "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            make_prompt_dataset(*engine, section, "the data block"));
    std::vector<TokenID> row;
    std::int32_t length = 0;
    read_row(*dataset, row, length);

    EXPECT_EQ(row, (std::vector<TokenID>{'h', 'i', 0, 0}));
    EXPECT_EQ(length, 2);

    SA_EXPECT_FAILURE(make_prompt_dataset(*engine, without_text, "the data block"),
                      StatusCode::InvalidDescription, "the data block", "has no 'text'");
    SA_EXPECT_FAILURE(make_prompt_dataset(*engine, with_unknown, "the data block"),
                      StatusCode::InvalidDescription, "'tokens'");
}
