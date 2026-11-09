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
 * @file test_training_cluster_section.cpp
 * @brief Tests read_training_cluster: a fallback to the master is refused under "deterministic".
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <support/test_names.hpp>

#include "training_cluster_section.hpp"

namespace
{
    using SushiAI::Test::require_value;
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
        return require_value(Config::parse_train(require_value(Config::parse_json(text, "<test>")),
                                                 "<test>",
                                   "."));
    }
}

namespace
{
/** @brief Holds the version and the extra members of a training file that states "local". */
struct RefusedCase
{
    /** @brief Names the case. */
    const char* name;

    /** @brief The "sushiai_train" version of the file. */
    int version;

    /** @brief The members placed before the "model" key. */
    const char* extra;
};
} // namespace

/** @brief Runs the rows of training files whose "local" fallback is refused. */
class Small_TrainingClusterSectionLocalFallback : public ::testing::TestWithParam<RefusedCase> {};

/** @brief Tests that "local" under a deterministic run is refused, naming both keys. */
TEST_P(Small_TrainingClusterSectionLocalFallback, IsRefusedNamingBothKeys)
{
    const Config::TrainSpec spec =
        spec_of(training_text(GetParam().version, GetParam().extra, K_LOCAL));

    SA_EXPECT_FAILURE(Apps::read_training_cluster(spec), StatusCode::InvalidDescription,
                      "replica 1", "'if_absent'", "'deterministic'");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_TrainingClusterSectionLocalFallback,
    ::testing::Values(RefusedCase{"UnderTheDefault", 3, ""},
                      RefusedCase{"UnderAStatedTrue", 3, R"("deterministic": true,)"},
                      RefusedCase{"InAVersion2File", 2, ""}),
    [](const ::testing::TestParamInfo<RefusedCase>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

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
