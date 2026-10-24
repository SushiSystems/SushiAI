/**************************************************************************/
/* test_replica_model.cpp                                                 */
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
 * @file test_replica_model.cpp
 * @brief Tests what a ReplicaModelFactory is called with, and what its model must hold.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;

    /** @brief Holds what a factory was called with. */
    struct FactoryCall
    {
        ReplicaDraws draws{};       /**< @brief The draws the step passed. */
        std::int64_t rows = 0;      /**< @brief The micro-batch rows the step passed. */
        int calls = 0;              /**< @brief How many times the factory ran. */
    };

    /** @brief Returns a factory that records its call into @p call and builds LinearModel. */
    ReplicaModelFactory recording_factory(FactoryCall& call)
    {
        return [&call](SushiBLAS::Engine& engine, const ReplicaDraws& draws, std::int64_t rows)
        {
            call.draws = draws;
            call.rows = rows;
            ++call.calls;
            return LinearModel::make(engine, rows);
        };
    }

    /** @brief Wraps LinearModel and hides every result, which a step refuses. */
    class ResultlessModel final : public ReplicaModel
    {
        public:
            /** @brief Takes the model it forwards to. */
            explicit ResultlessModel(std::unique_ptr<ReplicaModel> inner)
                : inner_(std::move(inner))
            {

            }

            const Graph::Graph& graph() const noexcept override
            {
                return inner_->graph();
            }

            Graph::TensorPool& pool() noexcept override
            {
                return inner_->pool();
            }

            SushiRuntime::span<const SushiBLAS::Tensor> parameters() const noexcept override
            {
                return inner_->parameters();
            }

            SushiRuntime::span<const SushiBLAS::Tensor> results() const noexcept override
            {
                return SushiRuntime::span<const SushiBLAS::Tensor>();
            }

            std::uint64_t fingerprint() const noexcept override
            {
                return inner_->fingerprint();
            }

            Status feed(const ReplicaRows& rows) override
            {
                return inner_->feed(rows);
            }

        private:
            /// @brief The model every call but results() reaches.
            std::unique_ptr<ReplicaModel> inner_;
    };
} // namespace

/** @brief Tests that the factory runs once, with the step's draws and micro-batch rows. */
TEST(Medium_ReplicaModel, TheFactoryReceivesTheDrawsAndTheMicroBatchRows)
{
    FactoryCall call;
    const ReplicaDraws draws{K_SEED, 0, K_RUN};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(recording_factory(call), draws, 3));

    EXPECT_EQ(call.calls, 1);
    EXPECT_EQ(call.draws, draws);
    EXPECT_EQ(call.rows, 3);
    EXPECT_EQ(step->draws(), draws);
}

/** @brief Tests that a step refuses a model that packs no result. */
TEST(Medium_ReplicaModel, AStepRefusesAModelWithNoResult)
{
    const ReplicaModelFactory factory = [](SushiBLAS::Engine& engine, const ReplicaDraws&,
                                           std::int64_t rows)
        -> Result<std::unique_ptr<ReplicaModel>>
    {
        SA_TRY_ASSIGN(std::unique_ptr<ReplicaModel> inner, LinearModel::make(engine, rows));
        return Result<std::unique_ptr<ReplicaModel>>(
            std::make_unique<ResultlessModel>(std::move(inner)));
    };

    SA_EXPECT_FAILURE(ReplicaStep::make(factory, ReplicaDraws{K_SEED, 0, K_RUN}, 2),
                      StatusCode::InvalidDescription, "no result to pack");
}
