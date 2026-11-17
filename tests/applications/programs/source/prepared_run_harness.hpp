/**************************************************************************/
/* prepared_run_harness.hpp                                               */
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
 * @file prepared_run_harness.hpp
 * @brief Declares the builder of a PreparedRun over the tiny_yolo fixture, through prepare_run().
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/optim/optimizer.hpp>

#include "registries.hpp"
#include "run_preparation.hpp"
#include "tiny_yolo_files.hpp"
#include "training_run.hpp"
#include "training_setup.hpp"

namespace SushiAI::Apps::PreparedTest
{
    /**
     * @struct RunHarness
     * @brief Holds the training file of the tiny_yolo fixture and the RunBuild that
     *        prepare_run() made over it.
     */
    struct RunHarness
    {
        RunHarness() = default;
        RunHarness(const RunHarness&) = delete;
        RunHarness& operator=(const RunHarness&) = delete;
        RunHarness(RunHarness&&) = delete;
        RunHarness& operator=(RunHarness&&) = delete;
        ~RunHarness() = default;

        /** @brief The training file of the fixture; a test may change what a run reads from it. */
        Config::TrainSpec spec;

        /** @brief The parsed model file the training file names. */
        Config::ModelSpec model_spec;

        /** @brief The registries the setup was built from. */
        std::optional<Registries> registries;

        /** @brief What prepare_run() built over the members above. */
        std::unique_ptr<RunBuild> build;

        /** @brief The optimizer of @c build. */
        Optim::Optimizer* optimizer = nullptr;

        /** @brief The run input of @c build; a test may change where the run starts. */
        PreparedRun* prepared = nullptr;
    };

    /**
     * @brief Builds the PreparedRun of the tiny_yolo training file on @p backend.
     * @param checkpoint The path the training file names for its checkpoint; a run of one
     *        epoch with no checkpoint_every writes none.
     * @return The harness, or the first failure of a read, a build or a bind.
     */
    [[nodiscard]] inline Result<std::unique_ptr<RunHarness>> make_run_harness(
        ExecutionBackend& backend, const std::filesystem::path& checkpoint)
    {
        std::unique_ptr<RunHarness> made = std::make_unique<RunHarness>();
        made->spec = TinyYoloTest::training_spec(checkpoint);
        SA_TRY_ASSIGN(made->model_spec, Config::load_model(made->spec.model_path));
        SA_TRY_ASSIGN(made->registries, compose_registries());
        SA_TRY_ASSIGN(made->build, prepare_run(*made->registries, "<test>", made->spec,
                                               made->model_spec, backend, SetupScope::Step));
        made->optimizer = &made->build->optimizer();
        made->prepared = &made->build->prepared();
        return std::move(made);
    }
} // namespace SushiAI::Apps::PreparedTest
