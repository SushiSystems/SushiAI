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
#include <SushiAI/failure/status.hpp>
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
    using SushiAI::Test::require_value;
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

/** @brief Tests that the model a source builds reports the fingerprint of that source. */
TEST_F(Medium_TrainingReplicaModel, AModelReportsTheFingerprintOfItsSource)
{
    const ClusterTestFiles files("sushiai_replica_model_fingerprint.json");
    const std::shared_ptr<const Apps::ReplicaSource> source = source_of(files);

    EXPECT_EQ(model_of(source)->fingerprint(), source->fingerprint);
}

namespace
{
    /** @brief Parses @p text as a training file beside the architecture file of @p files. */
    Config::TrainSpec read_spec(const ClusterTestFiles& files, const std::string& text)
    {
        return require_value(Config::parse_train(require_value(Config::parse_json(text, "<test>")),
                                                 "<test>",
                                   files.model_file.path().parent_path()));
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

    /** @brief Returns the fingerprint of the source load() builds from the file @p text states. */
    std::uint64_t fingerprint_by_load(const ClusterTestFiles& files, const std::string& text)
    {
        return fingerprint_of(read_spec(files, text));
    }

    /** @brief Returns the fingerprint of the source make() builds from the file @p text states. */
    std::uint64_t fingerprint_by_make(const ClusterTestFiles& files, const std::string& text)
    {
        return Apps::ReplicaSource::make(Apps::compose_registries().value(), read_spec(files, text),
                                         files.model_spec)
            .value()
            ->fingerprint;
    }

    /** @brief Holds a way to build a source from the files and what it is called. */
    struct EntryCase
    {
        /** @brief Names the case. */
        const char* name;

        /** @brief Builds the source from the files and returns its fingerprint. */
        std::uint64_t (*fingerprint)(const ClusterTestFiles&, const std::string&);
    };

    /** @brief Holds a change to the training file or the architecture file. */
    struct ChangeCase
    {
        /** @brief Names the case. */
        const char* name;

        /** @brief The text of the training file that is replaced. */
        const char* from;

        /** @brief The text that takes its place. */
        const char* to;

        /** @brief The text appended to the architecture file. */
        const char* model_suffix;
    };

    /** @brief Runs the rows of the entry points that build a source. */
    class Medium_ReplicaFingerprintEntry : public ::testing::TestWithParam<EntryCase> {};

    /** @brief Runs the rows of the changes that move the fingerprint. */
    class Medium_ReplicaFingerprintChanged : public ::testing::TestWithParam<ChangeCase> {};

    /** @brief Runs the rows of the changes that leave the fingerprint alone. */
    class Medium_ReplicaFingerprintUnchanged : public ::testing::TestWithParam<ChangeCase> {};

    /** @brief Applies @p change to the files: the text it replaces, and the suffix it appends. */
    std::string changed_text(const ClusterTestFiles& files, const std::string& text,
                             const ChangeCase& change)
    {
        std::ofstream(files.model_file.path(), std::ios::app) << change.model_suffix;
        return replaced(text, change.from, change.to);
    }

} // namespace

/** @brief Tests that two reads of the same files give one fingerprint through each entry point. */
TEST_P(Medium_ReplicaFingerprintEntry, TwoReadsOfTheSameFilesGiveOneFingerprint)
{
    const ClusterTestFiles files("sushiai_replica_model_two_reads.json");
    const std::string text = training_file(files.model_file.path());

    EXPECT_EQ(GetParam().fingerprint(files, text), GetParam().fingerprint(files, text));
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_ReplicaFingerprintEntry,
                         ::testing::Values(EntryCase{"ThroughLoad", &fingerprint_by_load},
                                           EntryCase{"ThroughMake", &fingerprint_by_make}),
                         [](const ::testing::TestParamInfo<EntryCase>& info)
                         {
                             return std::string(info.param.name);
                         });

/** @brief Tests that each part a replica's result depends on changes the fingerprint. */
TEST_P(Medium_ReplicaFingerprintChanged, AChangedPartChangesTheFingerprint)
{
    const ClusterTestFiles files("sushiai_replica_model_changed.json");
    const std::string text = training_file(files.model_file.path());
    const std::uint64_t master = fingerprint_of(read_spec(files, text));

    const Config::TrainSpec worker = read_spec(files, changed_text(files, text, GetParam()));

    EXPECT_NE(fingerprint_of(worker), master);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_ReplicaFingerprintChanged,
    ::testing::Values(
        ChangeCase{"TheRunSeed", R"("seed": 7,)", R"("seed": 8,)", ""},
        ChangeCase{"TheDataSeed", R"("seed": 3735928559)", R"("seed": 3735928558)", ""},
        ChangeCase{"TheBatchCount", R"("batches": 1,)", R"("batches": 2,)", ""},
        ChangeCase{"TheObjective", R"({"type": "cross_entropy"})",
                   R"({"type": "cross_entropy", "smoothing": 0.1})", ""},
        ChangeCase{"TheMicroBatch", R"("worker": "worker", "rows": 2, "micro_batch": 2)",
                   R"("worker": "worker", "rows": 2, "micro_batch": 1)", ""},
        ChangeCase{"TheArchitectureFile", "", "", " "}),
    [](const ::testing::TestParamInfo<ChangeCase>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that what only the master reads, or where output goes, is left out. */
TEST_P(Medium_ReplicaFingerprintUnchanged, AnUnrelatedPartLeavesTheFingerprintAlone)
{
    const ClusterTestFiles files("sushiai_replica_model_unchanged.json");
    const std::string text = training_file(files.model_file.path());
    const std::uint64_t master = fingerprint_of(read_spec(files, text));

    const Config::TrainSpec worker = read_spec(files, changed_text(files, text, GetParam()));

    EXPECT_EQ(fingerprint_of(worker), master);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_ReplicaFingerprintUnchanged,
    ::testing::Values(
        ChangeCase{"TheOptimizerRate", R"("learning_rate": 0.003)", R"("learning_rate": 0.004)",
                   ""},
        ChangeCase{"TheEpochsAndTheCheckpointPath", R"("epochs": 1,)",
                   R"("epochs": 3, "checkpoint": "elsewhere.sushi",)", ""}),
    [](const ::testing::TestParamInfo<ChangeCase>& info)
    {
        return std::string(info.param.name);
    });

namespace
{
    /** @brief Returns the replica "worker" of the shared training file with @p if_absent. */
    std::string worker_replica(const std::string& name, const std::string& if_absent)
    {
        return R"("worker": ")" + name + R"(", "rows": 2, "micro_batch": 2, "if_absent": ")" +
               if_absent + R"(")";
    }
} // namespace

/** @brief Tests that a replica's "if_absent" leaves the fingerprint alone. */
TEST_F(Medium_TrainingReplicaModel, TheIfAbsentOfAReplicaLeavesTheFingerprintAlone)
{
    const ClusterTestFiles files("sushiai_replica_model_if_absent.json");
    const std::string text = replaced(training_file(files.model_file.path()),
                                      R"("sushiai_train": 2,)",
                                      R"("sushiai_train": 3, "deterministic": false,)");
    const std::uint64_t master = fingerprint_of(read_spec(files, text));

    const Config::TrainSpec worker = read_spec(
        files, replaced(text, worker_replica("master", "fail"), worker_replica("master", "local")));
    EXPECT_EQ(fingerprint_of(worker), master);
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
