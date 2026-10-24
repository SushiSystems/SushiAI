/**************************************************************************/
/* sushiai_worker.cpp                                                     */
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
 * @file sushiai_worker.cpp
 * @brief Implements `sushiai_worker`, which serves one replica of a training file's cluster.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_main.hpp>
#include <SushiRuntime/distributed/worker_options_parser.hpp>
#include <SushiRuntime/process_environment/environment.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

#include "registries.hpp"
#include "served_kernel.hpp"
#include "training_replica_model.hpp"

namespace
{
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Holds the text `--help` prints before the flags of SushiRuntime's worker. */
    constexpr std::string_view K_USAGE =
        "sushiai_worker - serve one replica of a training file's cluster section.\n"
        "\n"
        "Usage: sushiai_worker <train.json> --master host:port --name <worker> [flags]\n"
        "\n"
        "  train.json is the master's training file; the worker builds the model it names\n"
        "  and serves the replica whose 'worker' is --name. The flags below are\n"
        "  SushiRuntime's worker flags, read by SushiRuntime.\n"
        "\n";

    /** @brief Holds the value the worker returns when its own arguments or files are wrong. */
    constexpr int K_BAD_ARGUMENTS = static_cast<int>(Distributed::WorkerExitCode::BadArguments);

    /**
     * @brief Builds the kernel the worker named @p name serves for @p spec_path.
     * @return The kernel, or InvalidDescription for a file that does not read, or the refusal
     *         of the section or of the replica's step.
     */
    [[nodiscard]] SushiAI::Result<std::unique_ptr<SushiAI::Distribution::ReplicaKernel>>
    build_kernel(const char* spec_path, const std::string& name)
    {
        SushiAI::Config::TrainSpec spec;
        SushiAI::Config::ModelSpec model_spec;
        try
        {
            spec = SushiAI::Config::load_train(spec_path);
            model_spec = SushiAI::Config::load_model(spec.model_path);
        }
        catch (const std::exception& error)
        {
            SA_FAIL_IF(true, SushiAI::StatusCode::InvalidDescription, "{}", error.what());
        }

        SA_FAIL_IF(spec.cluster.is_null(), SushiAI::StatusCode::InvalidDescription,
                   "'{}' has no 'cluster' section, so it names no replica to serve",
                   spec_path);
        SA_TRY_ASSIGN(const SushiAI::Distribution::ClusterSection section,
                      SushiAI::Distribution::ClusterSection::read(spec.cluster, spec.base));

        SA_TRY_ASSIGN(SushiAI::Apps::Registries registries, SushiAI::Apps::compose_registries());
        SA_TRY_ASSIGN(const std::shared_ptr<const SushiAI::Apps::ReplicaSource> source,
                      SushiAI::Apps::ReplicaSource::make(std::move(registries), std::move(spec),
                                                         std::move(model_spec)));

        // Every run of a file is run 0; the master's exchange names the same index.
        return SushiAI::Apps::make_served_kernel(source, section, name, 0);
    }

    /**
     * @brief Serves the replica the command line names until a signal or a fatal end.
     * @return A Distributed::WorkerExitCode value.
     */
    [[nodiscard]] int serve(int argc, char** argv)
    {
        const bool asks_help =
            argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help");
        const std::size_t forwarded =
            argc > 2 ? static_cast<std::size_t>(argc - 2) : 0;
        const Distributed::WorkerOptionsResult parsed = Distributed::parse_worker_options(
            SushiRuntime::span<const char* const>(forwarded != 0 ? argv + 2 : nullptr,
                                                  forwarded),
            Distributed::read_environment);

        if (asks_help || parsed.status == Distributed::WorkerOptionsStatus::Help)
        {
            std::cout << K_USAGE;
            if (parsed.status == Distributed::WorkerOptionsStatus::Help)
                std::cout << parsed.message;
            return EXIT_SUCCESS;
        }

        if (argc < 2)
        {
            std::cerr << "usage: sushiai_worker <train.json> --master host:port --name <worker> "
                         "[flags]\n";
            return K_BAD_ARGUMENTS;
        }

        if (parsed.status == Distributed::WorkerOptionsStatus::Version)
        {
            std::cout << parsed.message;
            return EXIT_SUCCESS;
        }

        if (parsed.status == Distributed::WorkerOptionsStatus::Invalid)
        {
            std::cerr << "sushiai_worker: " << parsed.message << "\n";
            return K_BAD_ARGUMENTS;
        }

        // The name picks the replica, so the worker cannot fall back to the host's name.
        if (parsed.options.name.empty())
        {
            std::cerr << "sushiai_worker needs --name or SR_WORKER_NAME: the name of the "
                         "replica's 'worker' in the cluster section\n";
            return K_BAD_ARGUMENTS;
        }

        SushiAI::Result<std::unique_ptr<SushiAI::Distribution::ReplicaKernel>> built =
            build_kernel(argv[1], parsed.options.name);
        if (!built.has_value())
        {
            std::cerr << "sushiai_worker failed: " << built.status().message() << "\n";
            return K_BAD_ARGUMENTS;
        }

        const std::unique_ptr<SushiAI::Distribution::ReplicaKernel> kernel =
            std::move(built.value());
        SushiRuntime::sushi_ptr<Distributed::KernelRegistry> kernels =
            SushiRuntime::make_sushi<Distributed::KernelRegistry>();
        kernel->register_with(*kernels);
        return Distributed::run_worker(parsed.options, kernels);
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        return serve(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai_worker failed: " << error.what() << "\n";
        return static_cast<int>(Distributed::WorkerExitCode::InternalError);
    }
}
