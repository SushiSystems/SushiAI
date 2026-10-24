/**************************************************************************/
/* test_replica_tensors.cpp                                               */
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
 * @file test_replica_tensors.cpp
 * @brief Tests ReplicaTensors::list(), the order of the tensors of a replica's packs.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "replica_tensors.hpp"
#include "training_setup.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_ReplicaTensors : public Test::EngineTest
    {

    };
} // namespace

/** @brief Tests that the results are the gradients in slot order, the loss, then each source. */
TEST_F(Medium_ReplicaTensors, ListsTheGradientsThenTheLossThenTheStatistics)
{
    const ClusterTestFiles files("sushiai_replica_tensors_model.json");
    const Apps::Registries registries = Apps::compose_registries().value();
    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, *engine, K_SEED)
            .value();

    const Apps::ReplicaTensors tensors =
        Apps::ReplicaTensors::list(
            SushiRuntime::span<const Optim::ParameterSlot>(setup->parameters().data(),
                                                           setup->parameters().size()),
            setup->built(), setup->pool(),
            SushiRuntime::span<const Train::BufferUpdateSlot>(setup->updates().data(),
                                                              setup->updates().size()))
            .value();

    const std::size_t parameters = setup->parameters().size();
    ASSERT_EQ(tensors.parameters.size(), parameters);
    ASSERT_EQ(tensors.results.size(), parameters + 1 + setup->updates().size());
    for (std::size_t index = 0; index < parameters; ++index)
    {
        EXPECT_EQ(tensors.parameters[index].data(), setup->parameters()[index].parameter.data());
        EXPECT_EQ(tensors.results[index].data(), setup->parameters()[index].gradient.data());
    }
    EXPECT_EQ(tensors.results[parameters].data(),
              setup->pool().tensor(setup->built().spec.loss).data());
    for (std::size_t index = 0; index < setup->updates().size(); ++index)
        EXPECT_EQ(tensors.results[parameters + 1 + index].data(),
                  setup->updates()[index].source.data());
}
