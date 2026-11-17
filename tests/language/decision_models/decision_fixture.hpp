/**************************************************************************/
/* decision_fixture.hpp                                                   */
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
 * @file decision_fixture.hpp
 * @brief Declares the files, the splits and the trained session the decision-model tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/language/decision_models/calibrate_decision_model.hpp>
#include <SushiAI/language/decision_models/decision_calibration.hpp>
#include <SushiAI/language/decision_models/decision_model.hpp>
#include <SushiAI/language/decision_models/decision_predictions.hpp>
#include <SushiAI/language/decision_models/decision_scores.hpp>
#include <SushiAI/language/decision_models/decision_session.hpp>
#include <SushiAI/language/decision_models/decision_set.hpp>
#include <SushiAI/language/registration/registration.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/scratch_files/scratch_directory.hpp>
#include <SushiAI/train/objective_registry.hpp>

namespace DecisionTest
{
    using namespace SushiAI;
    using namespace SushiAI::Language::DecisionModels;
    namespace SaTest = SushiAI::Test;

/** @brief Holds the seed of the training split. */
inline constexpr std::uint64_t K_TRAIN_SEED = 11;

/** @brief Holds the seed of the calibration split. */
inline constexpr std::uint64_t K_CALIBRATION_SEED = 12;

/** @brief Holds the seed of the held-out split. */
inline constexpr std::uint64_t K_TEST_SEED = 13;

/** @brief Counts the records of the training split: 96 batches of 16. */
inline constexpr std::int64_t K_TRAIN_RECORDS = 1536;

/** @brief Counts the records of the calibration split: 60 batches, 320 rows of each type. */
inline constexpr std::int64_t K_CALIBRATION_RECORDS = 960;

/** @brief Counts the records of the held-out split: 75 batches, 400 rows of each type. */
inline constexpr std::int64_t K_TEST_RECORDS = 1200;

/** @brief Counts the passes over the training split, as the training file states them. */
inline constexpr std::int64_t K_EPOCHS = 12;

/** @brief Names the model file the calibration step rewrites. */
inline constexpr const char* K_MODEL_FILE_NAME = "decision_model.json";

/** @brief Returns the path of @p name among the files of applications/examples/decision_model. */
inline std::filesystem::path example_file(const char* name)
{
    return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples"
           / "decision_model" / name;
}

/** @brief Writes @p text to @p file. */
inline void write_text(const std::filesystem::path& file, std::string_view text)
{
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

/** @brief Makes the dataset of a split drawn from @p seed, over a file under @p folder. */
inline Result<std::unique_ptr<Language::Datasets::DecisionDataset>> make_split(
    const std::filesystem::path& folder, const char* name, std::uint64_t seed,
    std::int64_t records)
{
    DecisionSetOptions set;
    set.seed = seed;
    set.records = records;
    set.options = static_cast<std::int32_t>(K_TEST_EXTENTS.options);

    const std::filesystem::path file = folder / name;
    SA_TRY(write_decision_set(file, set));

    Language::Datasets::DecisionDatasetOptions options;
    options.batch_size = K_TEST_EXTENTS.batch;
    options.shape = {K_TEST_EXTENTS.window, K_TEST_EXTENTS.options};

    return Language::Datasets::DecisionDataset::make(SaTest::TestEnvironment::engine(), file,
                                                     options);
}

/** @brief Holds the rows of the registries the two example files name. */
struct Registries
{
    NN::NodeRegistry nodes;               /**< The rows of the model file. */
    Train::ObjectiveRegistry objectives;  /**< The types of the objective block. */
};

/** @brief Returns the rows language and train register, as `sushiai train` composes them. */
inline Result<std::unique_ptr<Registries>> make_registries()
{
    std::unique_ptr<Registries> registries = std::make_unique<Registries>();
    SA_TRY(Language::register_nodes(registries->nodes));
    SA_TRY(Train::register_objectives(registries->objectives));

    return registries;
}

/** @brief Copies the example model file to @p file. */
inline Status copy_example_model(const std::filesystem::path& file)
{
    std::ifstream in(example_file("model.json"), std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    SA_FAIL_IF(text.empty(), StatusCode::FileNotFound, "the example model file is absent");
    write_text(file, text);

    return Status();
}

/** @brief Holds what the trained session says of the held-out split. */
struct TrainedModel
{
    DecisionCalibration fitted;        /**< The temperatures the calibration step returned. */
    DecisionCalibration stored;        /**< The temperatures the model file holds afterwards. */
    DecisionScores calibrated;         /**< The scores of the held-out split at the fitted T. */
};

/** @brief Trains the session, calibrates it on its split and scores the held-out split. */
inline Result<TrainedModel> train_and_calibrate()
{
    SushiBLAS::Engine& engine = SaTest::TestEnvironment::engine();
    engine.set_mode(SushiBLAS::ExecutionMode::Auto);
    engine.rng().reseed(0);

    const SaTest::ScratchDirectory scratch("sushiai_decision_model_test");
    std::filesystem::create_directories(scratch.path());
    SA_TRY_ASSIGN(const std::unique_ptr<Language::Datasets::DecisionDataset> train,
                  make_split(scratch.path(), "train.sadc", K_TRAIN_SEED, K_TRAIN_RECORDS));
    SA_TRY_ASSIGN(const std::unique_ptr<Language::Datasets::DecisionDataset> calibration,
                  make_split(scratch.path(), "calibration.sadc", K_CALIBRATION_SEED,
                             K_CALIBRATION_RECORDS));
    SA_TRY_ASSIGN(const std::unique_ptr<Language::Datasets::DecisionDataset> heldout,
                  make_split(scratch.path(), "heldout.sadc", K_TEST_SEED, K_TEST_RECORDS));

    // The model and the objective are the example files', through the calls `sushiai train` makes.
    SA_TRY_ASSIGN(const std::unique_ptr<Registries> registries, make_registries());
    DecisionSessionOptions options;
    options.nodes = &registries->nodes;
    options.objectives = &registries->objectives;
    SA_TRY_ASSIGN(options.model, Config::load_model(example_file("model.json")));
    SA_TRY_ASSIGN(options.train, Config::load_train(example_file("train.json")));
    SA_TRY_ASSIGN(const std::unique_ptr<DecisionSession> session,
                  DecisionSession::make(SaTest::TestEnvironment::backend(), options, *train));

    TrainedModel trained;
    SA_TRY(session->fit(*train));

    const std::filesystem::path model_file = scratch.path() / K_MODEL_FILE_NAME;
    SA_TRY(copy_example_model(model_file));
    SA_TRY_ASSIGN(trained.fitted, calibrate_decision_model(*session, *calibration, model_file));
    SA_TRY_ASSIGN(const Config::ModelSpec written, Config::load_model(model_file));
    SA_TRY_ASSIGN(trained.stored, read_calibration(written));

    SA_TRY_ASSIGN(const DecisionPredictions answers, session->predict(*heldout));
    SA_TRY_ASSIGN(trained.calibrated, score_decisions(answers, trained.fitted));

    return trained;
}

} // namespace DecisionTest
