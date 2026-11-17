/**************************************************************************/
/* test_decision_model.cpp                                                */
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
 * @file test_decision_model.cpp
 * @brief Tests that the example files of the decision model state the extents and the roles the module fixes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/datasets/decision_file.hpp>
#include <SushiAI/language/decision_models/decision_model.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include "decision_fixture.hpp"

using namespace SushiAI;
using namespace SushiAI::Language::DecisionModels;
using DecisionTest::example_file;
using DecisionTest::K_EPOCHS;
using DecisionTest::K_TRAIN_SEED;
using DecisionTest::make_split;
// SaTest stays reachable inside a TEST_F body; see tests/language/decision_models/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Expects @p dataset to publish the role that @p input names, as @p input states it. */
    void expect_published(const Data::Dataset& dataset, const Config::NamedInputSpec& input)
    {
        const Data::TargetDescription* const role = Data::find_target(dataset.targets(), input.name);

        ASSERT_NE(role, nullptr) << "the dataset publishes no role '" << input.name << "'";
        EXPECT_EQ(role->sample_shape, input.shape) << input.name;
        EXPECT_EQ(role->dtype, input.dtype) << input.name;
    }

    /** @brief Returns the string under @p key of the object @p term. */
    std::string text_of(const Config::Value& term, const char* key)
    {
        return SaTest::require_value(
            SaTest::require_value(term.at(key, "the term")).as_string("the term"));
    }

    /** @brief Returns the column of the type mask that the object @p term keeps. */
    std::int64_t column_of(const Config::Value& term)
    {
        const Config::Value& mask = SaTest::require_value(term.at("type_mask", "the term"));
        return SaTest::require_value(
            SaTest::require_value(mask.at("column", "the type mask")).as_int("the type mask"));
    }

    /** @brief Provides the engine for the cases that read the example files and a dataset. */
    class Medium_DecisionExampleFiles : public SaTest::EngineTest
    {

    };
} // namespace

/** @brief Tests that the four extra inputs of the model file are the roles the dataset publishes. */
TEST_F(Medium_DecisionExampleFiles, TheModelFileNamesTheRolesTheDatasetPublishes)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec model,
                            Config::load_model(example_file("model.json")));
    const SaTest::ScratchDirectory scratch("sushiai_decision_roles");
    std::filesystem::create_directories(scratch.path());
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Language::Datasets::DecisionDataset> dataset,
        make_split(scratch.path(), "roles.sadc", K_TRAIN_SEED, K_TEST_EXTENTS.batch));

    ASSERT_EQ(model.inputs.size(), 4u);
    EXPECT_EQ(model.inputs[0].name, "lengths");
    EXPECT_EQ(model.inputs[1].name, "options");
    EXPECT_EQ(model.inputs[2].name, "option_lengths");
    EXPECT_EQ(model.inputs[3].name, "option_valid");
    expect_published(*dataset, model.inputs[0]);
    expect_published(*dataset, model.inputs[1]);
    expect_published(*dataset, model.inputs[2]);
    expect_published(*dataset, model.inputs[3]);
    EXPECT_EQ(model.input_shape, dataset->sample_shape());
    EXPECT_EQ(model.input_dtype, dataset->sample_dtype());
    ASSERT_EQ(model.layers.size(), 1u);
    EXPECT_EQ(model.layers[0].type, "decision_heads");
    ASSERT_EQ(model.layers[0].from.size(), 5u);
    EXPECT_EQ(model.layers[0].from[0].node, Config::K_MODEL_INPUT);
    EXPECT_EQ(model.layers[0].from[1].input, "lengths");
    EXPECT_EQ(model.layers[0].from[2].input, "options");
    EXPECT_EQ(model.layers[0].from[3].input, "option_lengths");
    EXPECT_EQ(model.layers[0].from[4].input, "option_valid");
}

/** @brief Tests that the training file sums one term for each type, each naming output and target. */
TEST_F(Medium_DecisionExampleFiles, TheTrainFileSumsOneTermForEachTypeOfQuestion)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::TrainSpec train,
                            Config::load_train(example_file("train.json")));
    SA_ASSERT_OK_AND_ASSIGN(const Config::Value& terms_value,
                            train.objective_attributes.at("terms", "the objective"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<Config::Value>& terms,
                            terms_value.as_array("the terms"));

    EXPECT_EQ(train.objective_type, "sum");
    ASSERT_EQ(terms.size(), 3u);
    EXPECT_EQ(text_of(terms[0], "type"), "cross_entropy");
    EXPECT_EQ(text_of(terms[0], "output"), "choice");
    EXPECT_EQ(text_of(terms[0], "target"), "choice");
    EXPECT_EQ(column_of(terms[0]), 0);
    EXPECT_EQ(text_of(terms[1], "type"), "binary_cross_entropy");
    EXPECT_EQ(text_of(terms[1], "output"), "yes_no");
    EXPECT_EQ(text_of(terms[1], "target"), "yes_no");
    EXPECT_EQ(column_of(terms[1]), 1);
    EXPECT_EQ(text_of(terms[2], "type"), "mean_squared_error");
    EXPECT_EQ(text_of(terms[2], "output"), "number");
    EXPECT_EQ(text_of(terms[2], "target"), "number");
    EXPECT_EQ(column_of(terms[2]), 2);
}

/** @brief Tests that the data block of the training file states the extents the tests read. */
TEST_F(Medium_DecisionExampleFiles, TheDataBlockStatesTheExtentsOfTheTests)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::TrainSpec train,
                            Config::load_train(example_file("train.json")));

    SA_ASSERT_OK_AND_ASSIGN(const Language::Datasets::DecisionsSection keys,
                            Language::Datasets::read_decisions_section(train.data.attributes,
                                                                       "the data block"));

    EXPECT_EQ(train.data.type, Language::Datasets::K_DECISIONS_DATA_TYPE);
    EXPECT_EQ(keys.options.shape.window, K_TEST_EXTENTS.window);
    EXPECT_EQ(keys.options.shape.options, K_TEST_EXTENTS.options);
    EXPECT_EQ(keys.options.batch_size, K_TEST_EXTENTS.batch);
    EXPECT_EQ(train.epochs, K_EPOCHS);
}
