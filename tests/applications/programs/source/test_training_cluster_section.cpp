/**************************************************************************/
/* test_training_cluster_section.cpp                                        */
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
 * @file test_training_cluster_section.cpp
 * @brief Tests read_training_cluster: a fallback to the master is refused under "deterministic".
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "training_cluster_section.hpp"

namespace
{
    using namespace SushiAI;

    /** @brief Holds the members of a replica that falls back to the master. */
    constexpr const char* K_LOCAL = R"("if_absent": "local")";

    /**
     * @brief Returns the training file of @p version whose second replica states the members
     *        @p absent, with @p extra before its "model" key.
     */
    std::string training_text(int version, const std::string& extra, const std::string& absent)
    {
        return R"({"sushiai_train": )" + std::to_string(version) + "," + extra + R"(
            "model": "model.json",
            "objective": {"type": "cross_entropy"},
            "data": {"type": "rings", "batch_size": 4, "batches": 1, "classes": 4},
            "optimizer": {"type": "adamw", "learning_rate": 0.003},
            "epochs": 1,
            "seed": 7,
            "cluster": {
                "listen": "127.0.0.1:0",
                "secret_file": "cluster.secret",
                "replicas": [
                    {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                    {"worker": "worker", "rows": 2, "micro_batch": 2, )" +
               absent + R"(}
                ]
            }
        })";
    }

    /** @brief Parses @p text as a training file. */
    Config::TrainSpec spec_of(const std::string& text)
    {
        return Config::parse_train(Config::parse_json(text, "<test>"), "<test>", ".");
    }
}

/** @brief Tests that "local" under the default "deterministic" is refused, naming both keys. */
TEST(Small_TrainingClusterSection, ALocalFallbackUnderTheDefaultIsRefusedNamingBothKeys)
{
    const Config::TrainSpec spec = spec_of(training_text(3, "", K_LOCAL));

    ASSERT_TRUE(spec.deterministic);
    SA_EXPECT_FAILURE(Apps::read_training_cluster(spec), StatusCode::InvalidDescription,
                      "replica 1", "'if_absent': 'local'", "'deterministic' is true");
}

/** @brief Tests that "local" under an explicit "deterministic": true is refused. */
TEST(Small_TrainingClusterSection, ALocalFallbackUnderAStatedTrueIsRefused)
{
    const Config::TrainSpec spec = spec_of(training_text(3, R"("deterministic": true,)", K_LOCAL));

    SA_EXPECT_FAILURE(Apps::read_training_cluster(spec), StatusCode::InvalidDescription,
                      "'if_absent'", "'deterministic'");
}

/** @brief Tests that a version 2 file, which cannot state "deterministic", is refused too. */
TEST(Small_TrainingClusterSection, ALocalFallbackInAVersion2FileIsRefused)
{
    const Config::TrainSpec spec = spec_of(training_text(2, "", K_LOCAL));

    SA_EXPECT_FAILURE(Apps::read_training_cluster(spec), StatusCode::InvalidDescription,
                      "'if_absent'", "'deterministic'");
}

/** @brief Tests that "local" is read when the file states "deterministic": false. */
TEST(Small_TrainingClusterSection, ALocalFallbackIsReadWhenTheRunIsNotDeterministic)
{
    const Config::TrainSpec spec = spec_of(training_text(3, R"("deterministic": false,)", K_LOCAL));

    SA_EXPECT_OK(Apps::read_training_cluster(spec));
}

/** @brief Tests that "wait" and "fail" are read under "deterministic": true. */
TEST(Small_TrainingClusterSection, WaitAndFailAreReadUnderADeterministicRun)
{
    SA_EXPECT_OK(Apps::read_training_cluster(
        spec_of(training_text(3, "", R"("if_absent": "fail")"))));
    SA_EXPECT_OK(Apps::read_training_cluster(
        spec_of(training_text(3, "", R"("if_absent": "wait", "wait_ms": 250)"))));
}
