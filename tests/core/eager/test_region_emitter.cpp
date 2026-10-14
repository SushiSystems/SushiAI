/**************************************************************************/
/* test_region_emitter.cpp                                                */
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
 * @file test_region_emitter.cpp
 * @brief Tests that RegionEmitter records a region's program on the recorder it is handed.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include "eager_run_support.hpp"
#include "eager_test_support.hpp"
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_emitter.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
using Eager::RegionEmitter;
using Eager::RegionProgram;
using Graph::TensorPool;
using Graph::ValueID;
// SaTest stays reachable inside a TEST_F body; see tests/integration/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case records on. */
    class Unit_RegionEmitter : public SaTest::EngineTest
    {

    };

    /** @brief Bounds a FLOAT32 result against the same sum computed in double. */
    constexpr double K_TOLERANCE = 1e-5;

    /** @brief Holds the program of relu(x * weight + bias) and a pool with its tensors bound. */
    struct Recorded
    {
        /** @brief Builds the pool on @p engine; make_recorded() fills the rest. */
        explicit Recorded(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        Support::TracedStep step; /**< The traced step, one region. */
        RegionProgram program;    /**< Its one region, compiled. */
        TensorPool pool;          /**< Holds the tensor of each value of program. */
    };

    /**
     * @brief Compiles relu(x * weight + bias) and binds a filled tensor to each shared value.
     * @return The program and its pool, ready for an emitter.
     */
    [[nodiscard]] Result<std::unique_ptr<Recorded>> make_recorded(SushiBLAS::Engine& engine)
    {
        std::unique_ptr<Recorded> made = std::make_unique<Recorded>(engine);
        made->step = Support::linear_then_relu(false);

        const Eager::BoundaryMap boundaries = Support::boundaries_of(made->step);
        made->program = Eager::compile_region(made->step.graph, made->step.partition,
                                              made->step.partition.regions[0].key,
                                              boundaries.regions[0]);
        SA_TRY(made->pool.adopt(made->program.plan));

        std::vector<ValueID> shared = made->program.imports;
        shared.insert(shared.end(), made->program.exports.begin(), made->program.exports.end());
        for (std::size_t i = 0; i < shared.size(); ++i)
        {
            const Graph::Value& value = made->program.graph.value(shared[i]);
            SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                          allocate_tensor(engine, value.shape, value.dtype));
            Support::fill_wave(tensor, static_cast<double>(i), 0.5);
            SA_TRY(made->pool.bind(shared[i], tensor));
        }

        return std::move(made);
    }

    /** @brief Returns relu(x * weight + bias) computed on the host from the pool's tensors. */
    [[nodiscard]] std::vector<double> expected_result(const Recorded& recorded)
    {
        // The imports are read in the order MATMUL(x, weight), then BIAS_ADD(., bias).
        const std::vector<double> x = SaTest::read_tensor(
            recorded.pool.tensor(recorded.program.imports[0]));
        const std::vector<double> weight = SaTest::read_tensor(
            recorded.pool.tensor(recorded.program.imports[1]));
        const std::vector<double> bias = SaTest::read_tensor(
            recorded.pool.tensor(recorded.program.imports[2]));

        return Support::linear_relu(x, weight, bias);
    }
} // namespace

TEST_F(Unit_RegionEmitter, EmitRecordsTheProgramOnTheGraphItIsHanded)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Recorded> recorded, make_recorded(*engine));
    const std::vector<double> expected = expected_result(*recorded);

    SushiRuntime::Graph graph = engine->runtime().graph();
    RegionEmitter emitter(*engine, recorded->program, recorded->pool, nullptr);
    SA_EXPECT_OK(emitter.emit(graph.recorder()));
    SA_EXPECT_OK(recorded->pool.flush_constants());

    SushiRuntime::RunReport report;
    SA_EXPECT_OK(Detail::run_step(graph, report));

    const std::vector<double> result =
        SaTest::read_tensor(recorded->pool.tensor(recorded->step.hidden));
    ASSERT_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], K_TOLERANCE) << "element " << i;
}

TEST_F(Unit_RegionEmitter, EmitCountsItsCallsAndItsTasks)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Recorded> recorded, make_recorded(*engine));
    RegionEmitter emitter(*engine, recorded->program, recorded->pool, nullptr);
    EXPECT_EQ(emitter.emit_count(), 0u);

    SushiRuntime::Graph first = engine->runtime().graph();
    SA_EXPECT_OK(emitter.emit(first.recorder()));
    const std::size_t tasks = emitter.operation_count();

    // A second emit replays the same program, as a region recorded under a new key does.
    SushiRuntime::Graph second = engine->runtime().graph();
    SA_EXPECT_OK(emitter.emit(second.recorder()));

    EXPECT_EQ(emitter.emit_count(), 2u);
    EXPECT_GT(tasks, 0u);
    EXPECT_EQ(emitter.operation_count(), tasks);
    EXPECT_EQ(first.recorder().node_count(), second.recorder().node_count());
    EXPECT_TRUE(emitter.preludes().empty());
}
