/**************************************************************************/
/* cluster_test_files.hpp                                                 */
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
 * @file cluster_test_files.hpp
 * @brief Declares ClusterTestFiles, the training file of two replicas the cluster tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>

/** @brief Holds what the tests of the cluster run of `sushiai train` share. */
namespace SushiAI::Apps::ClusterTest
{
    /** @brief Holds the name the worker reports for its machine. */
    inline constexpr const char* K_WORKER_NAME = "worker";

    /**
     * @brief Returns the runtime configuration of the replica the in-process worker serves: the
     *        CPU alone, so each machine of the one process drives hardware of its own.
     *
     * Two runtimes of one process on one CUDA GPU on Windows fault a host copy; see KI-17 in
     * docs/reference/KNOWN_ISSUES.md.
     */
    inline SushiRuntime::RuntimeConfig worker_replica_config()
    {
        SushiRuntime::RuntimeConfig config = Distribution::replica_runtime_config();
        config.devices.include_gpu = false;
        config.devices.include_accelerator = false;
        return config;
    }

    /** @brief Holds the seed the training file states. */
    inline constexpr std::uint64_t K_SEED = 7;

    /** @brief Holds the architecture file: six features, one hidden layer, four classes. */
    inline constexpr const char* K_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [
            {"type": "linear", "units": 8},
            {"type": "gelu"},
            {"type": "linear", "units": 4}
        ]
    })";

    /**
     * @brief Returns a training file over @p model of one batch of four rows, two on the
     *        master and two on the worker, listening at @p listen.
     */
    inline std::string training_file(const std::filesystem::path& model,
                                     const std::string& listen = "127.0.0.1:0")
    {
        return R"({
            "sushiai_train": 2,
            "model": ")" + model.generic_string() + R"(",
            "objective": {"type": "cross_entropy"},
            "data": {"type": "rings", "batch_size": 4, "batches": 1, "classes": 4,
                     "seed": 3735928559},
            "optimizer": {"type": "adamw", "learning_rate": 0.003},
            "epochs": 1,
            "seed": 7,
            "cluster": {
                "listen": ")" + listen + R"(",
                "secret_file": "cluster.secret",
                "replicas": [
                    {"worker": "master", "rows": 2, "micro_batch": 2, "if_absent": "fail"},
                    {"worker": "worker", "rows": 2, "micro_batch": 2, "if_absent": "fail"}
                ]
            }
        })";
    }

    /**
     * @struct ClusterTestFiles
     * @brief Holds the architecture file on disk and the parsed training file over it.
     */
    struct ClusterTestFiles
    {
        /** @brief Writes the architecture file under @p name and parses the training file. */
        explicit ClusterTestFiles(const char* name, const std::string& listen = "127.0.0.1:0")
            : model_file(name)
        {
            {
                std::ofstream written(model_file.path());
                written << K_MODEL;
            }

            spec = Config::parse_train(
                Config::parse_json(training_file(model_file.path(), listen), "<test>"),
                "<test>", model_file.path().parent_path());
            model_spec = Config::load_model(spec.model_path);
        }

        /** @brief Returns the checked cluster section of the training file. */
        Distribution::ClusterSection section() const
        {
            return Distribution::ClusterSection::read(spec.cluster, spec.base).value();
        }

        /** @brief The architecture file, removed when the test ends. */
        Test::ScratchFile model_file;

        /** @brief The training file. */
        Config::TrainSpec spec;

        /** @brief The architecture file, parsed. */
        Config::ModelSpec model_spec;
    };
} // namespace SushiAI::Apps::ClusterTest
