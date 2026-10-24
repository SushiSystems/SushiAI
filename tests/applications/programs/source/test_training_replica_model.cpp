/**************************************************************************/
/* test_training_replica_model.cpp                                        */
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
 * @file test_training_replica_model.cpp
 * @brief Tests TrainingReplicaModel and the files every replica is built from.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "training_replica_model.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_TrainingReplicaModel : public Test::EngineTest
    {
        protected:
            /** @brief Returns the files of @p files as a replica source. */
            static std::shared_ptr<const Apps::ReplicaSource> source_of(
                const ClusterTestFiles& files)
            {
                return Apps::ReplicaSource::make(Apps::compose_registries().value(), files.spec,
                                                 files.model_spec)
                    .value();
            }

            /** @brief Returns the model of replica 1 over two rows, built from @p source. */
            std::unique_ptr<Distribution::ReplicaModel> model_of(
                const std::shared_ptr<const Apps::ReplicaSource>& source)
            {
                const Distribution::ReplicaDraws draws{K_SEED, 1, 0};
                return Apps::TrainingReplicaModel::make(source, *engine, draws, 2).value();
            }
    };

    /** @brief Returns the rows @p first and @p count of batch 0 of epoch 0. */
    Distribution::ReplicaRows rows_at(std::uint64_t first, std::uint64_t count)
    {
        Distribution::ReplicaRows rows;
        rows.rows.first = first;
        rows.rows.count = count;
        return rows;
    }
} // namespace

/** @brief Tests that an architecture file that cannot be opened is refused by name. */
TEST_F(Medium_TrainingReplicaModel, ASourceWithoutItsModelFileIsRefused)
{
    const ClusterTestFiles files("sushiai_replica_model_missing.json");
    Config::TrainSpec spec = files.spec;
    spec.model_path = spec.model_path.parent_path() / "sushiai_replica_model_absent.json";

    SA_EXPECT_FAILURE(
        Apps::ReplicaSource::make(Apps::compose_registries().value(), spec, files.model_spec)
            .status(),
        StatusCode::FileNotFound);
}

/** @brief Tests that two sources of the same files report one fingerprint, as their models do. */
TEST_F(Medium_TrainingReplicaModel, TheSameFilesGiveOneFingerprint)
{
    const ClusterTestFiles files("sushiai_replica_model_fingerprint.json");
    const std::shared_ptr<const Apps::ReplicaSource> first = source_of(files);
    const std::shared_ptr<const Apps::ReplicaSource> second = source_of(files);

    EXPECT_EQ(first->fingerprint, second->fingerprint);
    EXPECT_EQ(model_of(first)->fingerprint(), first->fingerprint);
}

namespace
{
    /** @brief Parses @p text as a training file beside the architecture file of @p files. */
    Config::TrainSpec read_spec(const ClusterTestFiles& files, const std::string& text)
    {
        return Config::parse_train(Config::parse_json(text, "<test>"), "<test>",
                                   files.model_file.path().parent_path());
    }

    /** @brief Returns @p text with its one @p from replaced by @p to. */
    std::string replaced(std::string text, const std::string& from, const std::string& to)
    {
        const std::size_t at = text.find(from);
        EXPECT_NE(at, std::string::npos) << from;
        if (at != std::string::npos)
            text.replace(at, from.size(), to);
        return text;
    }

    /** @brief Returns the fingerprint of the source load() builds from @p spec. */
    std::uint64_t fingerprint_of(const Config::TrainSpec& spec)
    {
        return Apps::ReplicaSource::load(spec).value()->fingerprint;
    }
} // namespace

/** @brief Tests that the master's and a worker's reads of the same files agree. */
TEST_F(Medium_TrainingReplicaModel, RegressionTwoReadsOfTheSameFilesGiveOneFingerprint)
{
    const ClusterTestFiles files("sushiai_replica_model_two_reads.json");
    const std::string text = training_file(files.model_file.path());

    EXPECT_EQ(fingerprint_of(read_spec(files, text)), fingerprint_of(read_spec(files, text)));
}

/** @brief Tests that each part a replica's result depends on changes the fingerprint. */
TEST_F(Medium_TrainingReplicaModel, RegressionAChangedTrainingFileChangesTheFingerprint)
{
    const ClusterTestFiles files("sushiai_replica_model_changed.json");
    const std::string text = training_file(files.model_file.path());
    const std::uint64_t master = fingerprint_of(read_spec(files, text));

    const std::vector<std::pair<std::string, std::string>> changes = {
        {R"("seed": 7,)", R"("seed": 8,)"},
        {R"("seed": 3735928559)", R"("seed": 3735928558)"},
        {R"("batches": 1,)", R"("batches": 2,)"},
        {R"({"type": "cross_entropy"})", R"({"type": "cross_entropy", "smoothing": 0.1})"},
        {R"("worker": "worker", "rows": 2, "micro_batch": 2)",
         R"("worker": "worker", "rows": 2, "micro_batch": 1)"},
    };
    for (const std::pair<std::string, std::string>& change : changes)
    {
        const Config::TrainSpec worker =
            read_spec(files, replaced(text, change.first, change.second));
        EXPECT_NE(fingerprint_of(worker), master) << change.second;
    }

    std::ofstream(files.model_file.path(), std::ios::app) << " ";
    EXPECT_NE(fingerprint_of(read_spec(files, text)), master) << "the architecture file";
}

/** @brief Tests that what only the master reads, or where output goes, is left out. */
TEST_F(Medium_TrainingReplicaModel, TheOptimizerAndTheOutputPathsLeaveTheFingerprintAlone)
{
    const ClusterTestFiles files("sushiai_replica_model_unchanged.json");
    const std::string text = training_file(files.model_file.path());
    const std::uint64_t master = fingerprint_of(read_spec(files, text));

    const std::vector<std::pair<std::string, std::string>> changes = {
        {R"("learning_rate": 0.003)", R"("learning_rate": 0.004)"},
        {R"("epochs": 1,)", R"("epochs": 3, "checkpoint": "elsewhere.sushi",)"},
        {R"("worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail")",
         R"("worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "local")"},
    };
    for (const std::pair<std::string, std::string>& change : changes)
    {
        const Config::TrainSpec worker =
            read_spec(files, replaced(text, change.first, change.second));
        EXPECT_EQ(fingerprint_of(worker), master) << change.second;
    }
}

/** @brief Tests that rows inside their batch, of the traced count, are fed. */
TEST_F(Medium_TrainingReplicaModel, FeedsRowsOfItsBatch)
{
    const ClusterTestFiles files("sushiai_replica_model_feed.json");
    const std::unique_ptr<Distribution::ReplicaModel> model = model_of(source_of(files));

    SA_EXPECT_OK(model->feed(rows_at(2, 2)));
}

/** @brief Tests that a feed of another row count than the traced one is refused. */
TEST_F(Medium_TrainingReplicaModel, RefusesRowsOfAnotherCount)
{
    const ClusterTestFiles files("sushiai_replica_model_count.json");
    const std::unique_ptr<Distribution::ReplicaModel> model = model_of(source_of(files));

    SA_EXPECT_FAILURE(model->feed(rows_at(0, 3)), StatusCode::InvalidDescription);
}

/** @brief Tests that rows that run past the end of their batch are refused. */
TEST_F(Medium_TrainingReplicaModel, RefusesRowsOutsideTheirBatch)
{
    const ClusterTestFiles files("sushiai_replica_model_outside.json");
    const std::unique_ptr<Distribution::ReplicaModel> model = model_of(source_of(files));

    SA_EXPECT_FAILURE(model->feed(rows_at(3, 2)), StatusCode::InvalidDescription);
}
