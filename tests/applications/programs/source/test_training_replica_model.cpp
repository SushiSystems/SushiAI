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

#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

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
