/**************************************************************************/
/* test_session.cpp                                                       */
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
 * @file test_session.cpp
 * @brief Tests Session: running a model eagerly, region by region, and what a settled loop keeps.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include "eager_run_support.hpp"
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/buffer_update_recorder.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/in_memory_descent_recorder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using Eager::Direction;
using Graph::ValueID;
// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;
using Eager::StepCost;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_EagerSession : public SaTest::EngineTest
    {

    };

    /** @brief Provides the Engine every death case runs on. */
    class Medium_EagerSessionFatal : public SaTest::EngineTest
    {

    };

    /** @brief Holds the step size of every loop of this file. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};

    /** @brief Holds the widths of the three-layer model. */
    const std::vector<int64_t> K_THREE_LAYERS{6, 5, 4, 3};

    /**
     * @brief Bounds the loss of a scoped loop against the unscoped one, in FLOAT32; 256 unit
     * roundoffs of float, the smallest power of two that covers 1e-5.
     */
    constexpr double K_LOSS_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Holds the same model and update compiled once through Graph::CompiledStep. */
    struct Compiled
    {
        /** @brief Builds the pool on @p engine; make_compiled() fills the rest. */
        explicit Compiled(SushiBLAS::Engine& engine) : pool(engine), update(K_RATE)
        {

        }

        Graph::Graph graph;                        /**< The differentiated, fused graph. */
        Graph::TensorPool pool;                    /**< Holds the tensor of each value. */
        Graph::InMemoryDescentRecorder update;     /**< The update recorded behind the graph. */
        std::vector<Support::LinearState> model;   /**< The layers; their tensors are bound. */
        SushiBLAS::Tensor inputs;                  /**< The batch, bound as the input leaf. */
        SushiBLAS::Tensor targets;                 /**< The targets, bound as their leaf. */
        ValueID loss = Graph::K_NO_VALUE;          /**< The value differentiated. */
        std::unique_ptr<Graph::CompiledStep> step; /**< The compiled step. */
    };

    /**
     * @brief Traces the model of K_TWO_LAYERS as trace_model() does and compiles it once.
     * @return The compiled step with the same initial numbers a Support::Loop starts from.
     */
    [[nodiscard]] Result<std::unique_ptr<Compiled>> make_compiled(RuntimeExecutionBackend& backend)
    {
        SushiBLAS::Engine& engine = backend.engine();
        std::unique_ptr<Compiled> made = std::make_unique<Compiled>(engine);
        SA_TRY_ASSIGN(made->model, Support::make_model(engine, K_TWO_LAYERS));
        SA_TRY_ASSIGN(made->inputs, Support::make_batch(engine, K_TWO_LAYERS.front(), 0.25));
        SA_TRY_ASSIGN(made->targets, Support::make_batch(engine, K_TWO_LAYERS.back(), 1.75));

        const Graph::GraphBuilder builder(made->graph);
        std::vector<ValueID> parameters;
        ValueID hidden = builder.input(Shape{Support::K_ROWS, K_TWO_LAYERS.front()}, "inputs");
        const ValueID inputs = hidden;
        for (std::size_t i = 0; i + 1 < K_TWO_LAYERS.size(); ++i)
        {
            const ValueID weight =
                builder.parameter(Shape{K_TWO_LAYERS[i], K_TWO_LAYERS[i + 1]}, "state");
            const ValueID bias = builder.parameter(Shape{K_TWO_LAYERS[i + 1]}, "state");
            parameters.push_back(weight);
            parameters.push_back(bias);

            SA_TRY_ASSIGN(const ValueID projected,
                          builder.bias_add(builder.matmul(hidden, weight), bias));
            const bool hidden_layer = i + 2 < K_TWO_LAYERS.size();
            SA_TRY_ASSIGN(hidden,
                          hidden_layer ? builder.relu(projected) : builder.tanh(projected));
        }
        const ValueID targets =
            builder.input(Shape{Support::K_ROWS, K_TWO_LAYERS.back()}, "targets");
        SA_TRY_ASSIGN(made->loss, builder.nll_loss(hidden, targets));

        SA_TRY_ASSIGN(const std::vector<ValueID> gradients,
                      Autograd::differentiate(
                          made->graph, made->loss,
                          SushiRuntime::span<const ValueID>(parameters.data(), parameters.size())));

        std::vector<ValueID> observed{inputs, targets, made->loss};
        observed.insert(observed.end(), parameters.begin(), parameters.end());
        observed.insert(observed.end(), gradients.begin(), gradients.end());
        std::vector<ValueID> pinned{made->loss};
        pinned.insert(pinned.end(), gradients.begin(), gradients.end());

        Graph::CastOptions casts;
        casts.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
        SA_TRY(Graph::insert_casts(made->graph, casts));
        Graph::FusionOptions fusion;
        fusion.observed = casts.observed;
        (void)Graph::fuse(made->graph, fusion);
        Graph::PlanOptions plan;
        plan.pinned = SushiRuntime::span<const ValueID>(pinned.data(), pinned.size());
        SA_TRY_ASSIGN(Graph::MemoryPlan memory, Graph::MemoryPlan::build(made->graph, plan));
        SA_TRY(made->pool.adopt(std::move(memory)));

        SA_TRY(made->pool.bind(inputs, made->inputs));
        SA_TRY(made->pool.bind(targets, made->targets));
        for (std::size_t i = 0; i < made->model.size(); ++i)
        {
            const Support::LinearState& layer = made->model[i];
            SA_TRY(made->pool.bind(parameters[2 * i], layer.weight));
            SA_TRY(made->pool.bind(parameters[2 * i + 1], layer.bias));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& weight_gradient,
                          made->pool.materialise(gradients[2 * i],
                                                 made->graph.value(gradients[2 * i])));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& bias_gradient,
                          made->pool.materialise(gradients[2 * i + 1],
                                                 made->graph.value(gradients[2 * i + 1])));
            made->update.add(layer.weight, weight_gradient);
            made->update.add(layer.bias, bias_gradient);
        }

        SA_TRY_ASSIGN(made->step, Graph::CompiledStep::compile(backend, made->graph, made->pool,
                                                              &made->update));
        return std::move(made);
    }

    /** @brief Returns whether every element of @p tensor is zero. */
    [[nodiscard]] bool is_all_zero(const SushiBLAS::Tensor& tensor)
    {
        for (double element : SaTest::read_tensor(tensor))
        {
            if (element != 0.0)
                return false;
        }

        return true;
    }

    /**
     * @brief Expects the first @p steps steps of @p loop to give the loss bytes @p compiled gives.
     * @return Nothing; a mismatch or a refusal fails the calling test.
     */
    void expect_loss_bytes_match(Compiled& compiled, Support::Loop& loop, int steps)
    {
        for (int step = 0; step < steps; ++step)
        {
            SA_EXPECT_OK(compiled.step->run());
            const std::vector<unsigned char> expected =
                SaTest::read_tensor_bytes(compiled.pool.tensor(compiled.loss));

            SA_ASSERT_OK_AND_ASSIGN(const std::vector<unsigned char> bytes, loop.step_bytes());
            EXPECT_EQ(SaTest::first_difference(bytes, expected), expected.size())
                << "step " << step;
            EXPECT_EQ(bytes.size(), expected.size());
        }
    }

    /** @brief Runs @p steps more steps of @p loop with @p options. */
    void run_steps(Support::Loop& loop, const Support::TraceOptions& options, int steps)
    {
        for (int step = 0; step < steps; ++step)
            SA_EXPECT_OK(loop.step(options));
    }

    /** @brief Holds what a settled loop held after its first step. */
    struct Baseline
    {
        std::size_t regions = 0;     /**< The regions the executor held. */
        std::size_t emits = 0;       /**< The regions lowered into a recorder. */
        std::size_t pool_bytes = 0;  /**< The pool bytes the executor held. */
        std::size_t allocations = 0; /**< The allocations the storage bank counted. */
        std::size_t tensors = 0;     /**< The tensors the storage bank held. */
    };

    /** @brief Prepares the three-layer scoped @p loop, runs its first step and reads its state. */
    Baseline settle_first_step(SushiBLAS::Engine& engine, Support::Loop& loop)
    {
        SA_EXPECT_OK(loop.prepare(engine, K_THREE_LAYERS));
        Support::TraceOptions options;
        options.scoping = Support::Scoping::PerLayer;
        SA_EXPECT_OK(loop.step(options));

        Baseline baseline;
        baseline.regions = loop.executor.region_count();
        baseline.emits = loop.executor.emit_count();
        baseline.pool_bytes = loop.executor.allocated_bytes();
        baseline.allocations = loop.session.storage().allocation_count();
        baseline.tensors = loop.session.storage().size();
        return baseline;
    }

    /** @brief Expects the scoped losses of @p steps steps to follow the unscoped ones. */
    void expect_scoped_losses_follow_plain(Support::Loop& plain, Support::Loop& scoped,
                                           int steps)
    {
        Support::TraceOptions split;
        split.scoping = Support::Scoping::SplitActivation;
        for (int step = 0; step < steps; ++step)
        {
            SA_ASSERT_OK_AND_ASSIGN(const double expected, plain.step());
            SA_ASSERT_OK_AND_ASSIGN(const double loss, scoped.step(split));
            EXPECT_NEAR(loss, expected, K_LOSS_TOLERANCE) << "step " << step;
            EXPECT_TRUE(std::isfinite(loss));
        }
    }

    /** @brief Expects no region of the last step of @p loop to fuse a layer with its ReLU. */
    void expect_no_region_fuses_layer_and_activation(const Support::Loop& loop)
    {
        for (const Eager::RegionOrder& order : loop.session.last_plan().regions)
        {
            const Eager::RegionProgram* const program = loop.executor.program_of(order.key);
            ASSERT_NE(program, nullptr) << "key " << order.key;
            EXPECT_EQ(program->fusion.count(Graph::FusedForm::GEMMBiasReLU), 0u)
                << "key " << order.key;
        }
    }

    /** @brief Traces, differentiates, updates and finishes @p step; stores its loss in @p loss. */
    void trace_and_run(Support::Loop& loop, Eager::Step& step, ValueID& loss)
    {
        SA_ASSERT_OK_AND_ASSIGN(loss,
                                Support::trace_model(step, loop.model, loop.inputs, loop.targets));
        SA_EXPECT_OK(step.backward(loss));
        SA_EXPECT_OK(step.apply(loop.update));
    }

    /** @brief Holds the weight every blend of this file gives the buffer's current content. */
    constexpr double K_MOMENTUM = 0.1;

    /** @brief Holds the factor every blend of this file scales its source by. */
    constexpr double K_SOURCE_SCALE = 2.0;

    /** @brief Holds the width of the running statistic and of the states that share a gradient. */
    constexpr int64_t K_STAT_WIDTH = 3;

    /** @brief Holds how many steps the buffer and shared-gradient cases run. */
    constexpr int K_BUFFER_STEPS = 3;

    /** @brief Holds the [K_ROWS, width] batch statistic's graph, compiled with its blend. */
    struct BlendCompiled
    {
        /** @brief Builds the pool on @p engine; make_blend_compiled() fills the rest. */
        explicit BlendCompiled(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        Graph::Graph graph;                                 /**< Row sums of the batch. */
        Graph::TensorPool pool;                             /**< Holds the tensor of each value. */
        std::unique_ptr<Graph::BufferUpdateRecorder> blend; /**< Blends the row sums. */
        std::unique_ptr<Graph::CompiledStep> step;          /**< The compiled step. */
    };

    /**
     * @brief Compiles the row sums of @p batch with a blend of them into @p running behind it.
     * @return The compiled step, which reads @p batch in place on every run.
     */
    [[nodiscard]] Result<std::unique_ptr<BlendCompiled>> make_blend_compiled(
        RuntimeExecutionBackend& backend, const SushiBLAS::Tensor& batch,
        const SushiBLAS::Tensor& running)
    {
        std::unique_ptr<BlendCompiled> made = std::make_unique<BlendCompiled>(backend.engine());
        const Graph::GraphBuilder builder(made->graph);
        const ValueID x = builder.input(Shape{Support::K_ROWS, K_STAT_WIDTH}, "batch");
        SA_TRY_ASSIGN(const ValueID statistic, builder.sum_rows(x));

        Graph::PlanOptions plan;
        plan.pinned = SushiRuntime::span<const ValueID>(&statistic, 1);
        SA_TRY_ASSIGN(Graph::MemoryPlan memory, Graph::MemoryPlan::build(made->graph, plan));
        SA_TRY(made->pool.adopt(std::move(memory)));
        SA_TRY(made->pool.bind(x, batch));
        SA_TRY_ASSIGN(SushiBLAS::Tensor& source,
                      made->pool.materialise(statistic, made->graph.value(statistic)));

        std::vector<Graph::BufferUpdateSlot> slots;
        slots.push_back(Graph::BufferUpdateSlot{running, source, K_MOMENTUM, K_SOURCE_SCALE});
        made->blend = std::make_unique<Graph::BufferUpdateRecorder>(std::move(slots));
        SA_TRY_ASSIGN(made->step, Graph::CompiledStep::compile(backend, made->graph, made->pool,
                                                              made->blend.get()));
        return std::move(made);
    }

    /** @brief Runs @p steps eager steps that blend the row sums of @p batch into @p buffer. */
    [[nodiscard]] Status run_eager_blends(Eager::Session& session, Graph::StateID buffer,
                                          const SushiBLAS::Tensor& batch, int steps)
    {
        for (int i = 0; i < steps; ++i)
        {
            Eager::Step step = session.begin_step();
            const Graph::Traced x = step.input(batch, "batch");
            SA_TRY(step.blend_buffer(buffer, step.builder().sum_rows(x), K_MOMENTUM,
                                     K_SOURCE_SCALE));
            SA_TRY(step.finish());
        }

        return Status();
    }

    /** @brief Runs @p compiled @p steps times. */
    [[nodiscard]] Status run_compiled_steps(Graph::CompiledStep& compiled, int steps)
    {
        for (int i = 0; i < steps; ++i)
            SA_TRY(compiled.run());

        return Status();
    }

    /** @brief Holds the two parameters whose gradients are one value. */
    struct SharedPair
    {
        SushiBLAS::Tensor first;  /**< The first parameter, [K_ROWS, K_STAT_WIDTH]. */
        SushiBLAS::Tensor second; /**< The second parameter, [K_ROWS, K_STAT_WIDTH]. */
    };

    /** @brief Allocates a pair filled the same way on every call. */
    [[nodiscard]] Result<SharedPair> make_shared_pair(SushiBLAS::Engine& engine)
    {
        SharedPair pair;
        SA_TRY_ASSIGN(pair.first, allocate_tensor(engine, Shape{Support::K_ROWS, K_STAT_WIDTH}));
        SA_TRY_ASSIGN(pair.second, allocate_tensor(engine, Shape{Support::K_ROWS, K_STAT_WIDTH}));
        Support::fill_wave(pair.first, 0.2, 0.4);
        Support::fill_wave(pair.second, 0.9, 0.4);
        return pair;
    }

    /** @brief Holds the graph of loss = nll(first + second, targets) compiled with its update. */
    struct SharedCompiled
    {
        /** @brief Builds the pool on @p engine and the update with @p rate. */
        SharedCompiled(SushiBLAS::Engine& engine, float rate) : pool(engine), update(rate)
        {

        }

        Graph::Graph graph;                    /**< The differentiated graph. */
        Graph::TensorPool pool;                /**< Holds the tensor of each value. */
        Graph::InMemoryDescentRecorder update; /**< The update recorded behind the graph. */
        ValueID first_gradient = Graph::K_NO_VALUE;  /**< The gradient of the first parameter. */
        ValueID second_gradient = Graph::K_NO_VALUE; /**< The gradient of the second parameter. */
        std::unique_ptr<Graph::CompiledStep> step;   /**< The compiled step. */
    };

    /**
     * @brief Compiles loss = nll(first + second, targets) over the tensors of @p pair.
     * @return The compiled step; both parameters read one gradient tensor behind it.
     */
    [[nodiscard]] Result<std::unique_ptr<SharedCompiled>> make_shared_compiled(
        RuntimeExecutionBackend& backend, const SharedPair& pair, const SushiBLAS::Tensor& targets)
    {
        std::unique_ptr<SharedCompiled> made =
            std::make_unique<SharedCompiled>(backend.engine(), K_RATE);
        const Graph::GraphBuilder builder(made->graph);
        const Shape shape{Support::K_ROWS, K_STAT_WIDTH};
        const ValueID first = builder.parameter(shape, "state");
        const ValueID second = builder.parameter(shape, "state");
        const ValueID labels = builder.input(shape, "targets");
        SA_TRY_ASSIGN(const ValueID total, builder.add(first, second));
        SA_TRY_ASSIGN(const ValueID loss, builder.nll_loss(total, labels));

        const std::vector<ValueID> parameters{first, second};
        SA_TRY_ASSIGN(const std::vector<ValueID> gradients,
                      Autograd::differentiate(
                          made->graph, loss,
                          SushiRuntime::span<const ValueID>(parameters.data(), parameters.size())));
        made->first_gradient = gradients[0];
        made->second_gradient = gradients[1];

        std::vector<ValueID> observed{labels, loss, first, second};
        observed.insert(observed.end(), gradients.begin(), gradients.end());
        std::vector<ValueID> pinned{loss};
        pinned.insert(pinned.end(), gradients.begin(), gradients.end());

        Graph::CastOptions casts;
        casts.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
        SA_TRY(Graph::insert_casts(made->graph, casts));
        Graph::FusionOptions fusion;
        fusion.observed = casts.observed;
        (void)Graph::fuse(made->graph, fusion);
        Graph::PlanOptions plan;
        plan.pinned = SushiRuntime::span<const ValueID>(pinned.data(), pinned.size());
        SA_TRY_ASSIGN(Graph::MemoryPlan memory, Graph::MemoryPlan::build(made->graph, plan));
        SA_TRY(made->pool.adopt(std::move(memory)));

        SA_TRY(made->pool.bind(labels, targets));
        SA_TRY(made->pool.bind(first, pair.first));
        SA_TRY(made->pool.bind(second, pair.second));
        SA_TRY_ASSIGN(SushiBLAS::Tensor& shared,
                      made->pool.materialise(gradients[0], made->graph.value(gradients[0])));
        made->update.add(pair.first, shared);
        made->update.add(pair.second, made->pool.tensor(gradients[1]));

        SA_TRY_ASSIGN(made->step, Graph::CompiledStep::compile(backend, made->graph, made->pool,
                                                              &made->update));
        return std::move(made);
    }

    /** @brief Binds @p pair as trainable states @p first and @p second and pairs each with its gradient. */
    [[nodiscard]] Status bind_shared_pair(Eager::Session& session, const SharedPair& pair,
                                          Graph::StateID first, Graph::StateID second,
                                          Graph::InMemoryDescentRecorder& update)
    {
        SA_TRY(session.bind_state(first, pair.first, true));
        SA_TRY(session.bind_state(second, pair.second, true));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor first_gradient, session.gradient(first));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor second_gradient, session.gradient(second));
        update.add(pair.first, first_gradient);
        update.add(pair.second, second_gradient);
        return Status();
    }

    /** @brief Runs @p steps eager steps of loss = nll(first + second, targets) with @p update. */
    [[nodiscard]] Status run_shared_steps(Eager::Session& session, Graph::StateID first,
                                          Graph::StateID second, const SushiBLAS::Tensor& targets,
                                          Graph::InMemoryDescentRecorder& update, int steps)
    {
        for (int i = 0; i < steps; ++i)
        {
            Eager::Step step = session.begin_step();
            SA_TRY_ASSIGN(const ValueID a, step.state(first));
            SA_TRY_ASSIGN(const ValueID b, step.state(second));
            SA_TRY_ASSIGN(const ValueID loss,
                          step.builder().nll_loss(step.builder().add(a, b),
                                                  step.input(targets, "targets")));
            SA_TRY(step.backward(loss));
            SA_TRY(step.apply(update));
        }

        return Status();
    }
} // namespace

/** @brief Tests that an unscoped step gives the loss bytes the compiled step gives. */
TEST_F(Medium_EagerSession, AnUnscopedStepGivesTheLossBytesOfTheCompiledStep)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Compiled> compiled, make_compiled(backend()));
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    expect_loss_bytes_match(*compiled, loop, 4);

    // With no scope the forward and backward nodes are one region.
    EXPECT_EQ(loop.executor.region_count(), 1u);
    EXPECT_EQ(loop.session.step_count(), 4u);
}

/** @brief Tests that the first step of a scoped loop builds every region it holds. */
TEST_F(Medium_EagerSession, TheFirstStepOfAScopedLoopBuildsEveryRegion)
{
    Support::Loop loop(backend(), K_RATE);

    const Baseline baseline = settle_first_step(*engine, loop);

    EXPECT_GT(baseline.regions, 1u);
    EXPECT_EQ(loop.session.last_cost().built, baseline.regions);
}

/** @brief Tests that a settled loop composes once and lowers no region again. */
TEST_F(Medium_EagerSession, ASettledLoopComposesOnce)
{
    Support::Loop loop(backend(), K_RATE);
    const Baseline baseline = settle_first_step(*engine, loop);
    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;

    run_steps(loop, options, 4);

    EXPECT_EQ(loop.executor.compose_count(), 1u);
    EXPECT_EQ(loop.executor.emit_count(), baseline.emits);
}

/** @brief Tests that a settled loop allocates no pool byte, no tensor and no storage more. */
TEST_F(Medium_EagerSession, ASettledLoopAllocatesNothingMore)
{
    Support::Loop loop(backend(), K_RATE);
    const Baseline baseline = settle_first_step(*engine, loop);
    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;

    run_steps(loop, options, 4);

    EXPECT_EQ(loop.executor.allocated_bytes(), baseline.pool_bytes);
    EXPECT_EQ(loop.session.storage().allocation_count(), baseline.allocations);
    EXPECT_EQ(loop.session.storage().size(), baseline.tensors);
}

/** @brief Tests that the last step of a settled loop keeps every region and builds none. */
TEST_F(Medium_EagerSession, ASettledLoopKeepsEveryRegion)
{
    Support::Loop loop(backend(), K_RATE);
    const Baseline baseline = settle_first_step(*engine, loop);
    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;

    run_steps(loop, options, 4);

    EXPECT_EQ(loop.session.last_cost().kept, baseline.regions);
    EXPECT_EQ(loop.session.last_cost().built, 0u);
}

/** @brief Tests that the update of a settled loop is recorded once and replayed every step. */
TEST_F(Medium_EagerSession, ASettledLoopReplaysTheUpdateItRecordedOnce)
{
    Support::Loop loop(backend(), K_RATE);
    (void)settle_first_step(*engine, loop);
    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;

    run_steps(loop, options, 4);

    EXPECT_EQ(loop.update.record_count(), 1u);
    EXPECT_EQ(loop.update.replay_count(), 5u);
}

/** @brief Tests that a scope between a layer and its activation trains to the same loss. */
TEST_F(Medium_EagerSession, AScopeBetweenALinearLayerAndItsActivationTrainsToTheSameLoss)
{
    Support::Loop plain(backend(), K_RATE);
    Support::Loop scoped(backend(), K_RATE);
    SA_EXPECT_OK(plain.prepare(*engine, K_TWO_LAYERS));
    SA_EXPECT_OK(scoped.prepare(*engine, K_TWO_LAYERS));

    expect_scoped_losses_follow_plain(plain, scoped, 5);

    // No region of the scoped loop holds the linear layer and its activation together.
    expect_no_region_fuses_layer_and_activation(scoped);
    EXPECT_GT(scoped.executor.region_count(), plain.executor.region_count());
}

TEST_F(Medium_EagerSession, AParameterNoStepTracesKeepsAZeroGradientAndItsValue)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    SA_ASSERT_OK_AND_ASSIGN(std::vector<Support::LinearState> unused,
                            Support::make_model(*engine, {4, 3}));
    SA_EXPECT_OK(Support::bind_model(loop.session, unused, loop.update));
    const std::vector<unsigned char> before = SaTest::read_tensor_bytes(unused[0].weight);

    SA_EXPECT_OK(loop.step());
    SA_EXPECT_OK(loop.step());

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor gradient,
                            loop.session.gradient(unused[0].weight_state));
    EXPECT_TRUE(is_all_zero(gradient));
    EXPECT_EQ(SaTest::read_tensor_bytes(unused[0].weight), before);
}

TEST_F(Medium_EagerSession, AStepWithoutBackwardRunsItsForwardNodesWhenAValueIsRead)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    EXPECT_FALSE(step.has_run());

    SA_ASSERT_OK_AND_ASSIGN(const double value, step.read_scalar(loss));

    EXPECT_TRUE(step.has_run());
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_EQ(loop.session.step_count(), 1u);
    EXPECT_EQ(loop.update.record_count(), 0u);
}

/** @brief Tests that a value the step did not keep cannot be read after the run. */
TEST_F(Medium_EagerSession, AValueTheStepDidNotKeepCannotBeReadAfterTheRun)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));
    Eager::Step step = loop.session.begin_step();
    ValueID loss = Graph::K_NO_VALUE;
    trace_and_run(loop, step, loss);

    // The loss reads the model's result inside the step's one region; keep() never named it.
    const ValueID interior = step.graph().node(step.graph().value(loss).producer).inputs[0];
    SA_EXPECT_FAILURE(step.read(interior), StatusCode::InvalidDescription, "not kept");
}

/** @brief Tests that keep() after the run is refused with Unsupported. */
TEST_F(Medium_EagerSession, KeepingAValueAfterTheRunIsRefused)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));
    Eager::Step step = loop.session.begin_step();
    ValueID loss = Graph::K_NO_VALUE;
    trace_and_run(loop, step, loss);
    SA_EXPECT_OK(step.read_scalar(loss));

    const ValueID interior = step.graph().node(step.graph().value(loss).producer).inputs[0];
    SA_EXPECT_FAILURE(step.keep(interior), StatusCode::Unsupported, "has run");
}

TEST_F(Medium_EagerSession, TracingAfterTheStepRanIsRefused)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.finish());

    SA_EXPECT_OK(step.builder().scale(loss, 2.0));
    SA_EXPECT_FAILURE(step.read_scalar(loss), StatusCode::Unsupported, "after the step");
}

TEST_F(Medium_EagerSession, AStateTheSessionDoesNotHoldIsRefused)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Graph::StateIdentity stranger;
    Eager::Step step = loop.session.begin_step();
    SA_EXPECT_FAILURE(step.state(stranger.get()), StatusCode::InvalidDescription,
                      "not bound");
    SA_EXPECT_FAILURE(loop.session.gradient(stranger.get()), StatusCode::InvalidDescription,
                      "not bound as trainable");
}

TEST_F(Medium_EagerSession, BindingAStateTwiceIsRefused)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    SA_EXPECT_FAILURE(
        loop.session.bind_state(loop.model[0].weight_state, loop.model[0].weight, true),
        StatusCode::InvalidDescription, "already bound");
}

/** @brief Tests that a running statistic ends where the compiled step's blend leaves it. */
TEST_F(Medium_EagerSession, ARunningStatisticIsBlendedAcrossStepsAsTheCompiledStepBlendsIt)
{
    Eager::ScopePartitioner partitioner;
    Eager::RegionExecutor executor(backend());
    Eager::Session session(backend(), executor, partitioner);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor batch,
                            Support::make_batch(*engine, K_STAT_WIDTH, 0.25));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor eager_running,
                            allocate_tensor(*engine, Shape{K_STAT_WIDTH}));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor compiled_running,
                            allocate_tensor(*engine, Shape{K_STAT_WIDTH}));
    Support::fill_wave(eager_running, 0.5, 0.3);
    Support::fill_wave(compiled_running, 0.5, 0.3);
    const std::vector<unsigned char> initial = SaTest::read_tensor_bytes(eager_running);
    Graph::StateIdentity identity;
    const Graph::StateID buffer = identity.get();
    SA_EXPECT_OK(session.bind_state(buffer, eager_running, false));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<BlendCompiled> compiled,
                            make_blend_compiled(backend(), batch, compiled_running));

    SA_EXPECT_OK(run_eager_blends(session, buffer, batch, K_BUFFER_STEPS));
    SA_EXPECT_OK(run_compiled_steps(*compiled->step, K_BUFFER_STEPS));

    const std::vector<unsigned char> expected = SaTest::read_tensor_bytes(compiled_running);
    const std::vector<unsigned char> bytes = SaTest::read_tensor_bytes(eager_running);
    EXPECT_EQ(SaTest::first_difference(bytes, expected), expected.size());
    EXPECT_EQ(bytes.size(), expected.size());
    EXPECT_NE(bytes, initial);
}

/** @brief Tests that the blend a settled loop records is recorded once and replayed every step. */
TEST_F(Medium_EagerSession, ABlendOfASettledLoopIsRecordedOnce)
{
    Eager::ScopePartitioner partitioner;
    Eager::RegionExecutor executor(backend());
    Eager::Session session(backend(), executor, partitioner);
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor batch,
                            Support::make_batch(*engine, K_STAT_WIDTH, 0.25));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor running,
                            allocate_tensor(*engine, Shape{K_STAT_WIDTH}));
    Graph::StateIdentity identity;
    const Graph::StateID buffer = identity.get();
    SA_EXPECT_OK(session.bind_state(buffer, running, false));

    SA_EXPECT_OK(run_eager_blends(session, buffer, batch, K_BUFFER_STEPS));

    EXPECT_EQ(executor.trailing_record_count(), 1u);
}

/** @brief Tests that two states with one gradient value run and update as the compiled step does. */
TEST_F(Medium_EagerSession, TwoStatesThatShareOneGradientValueGetTheCompiledUpdate)
{
    Eager::ScopePartitioner partitioner;
    Eager::RegionExecutor executor(backend());
    Eager::Session session(backend(), executor, partitioner);
    SA_ASSERT_OK_AND_ASSIGN(const SharedPair eager, make_shared_pair(*engine));
    SA_ASSERT_OK_AND_ASSIGN(const SharedPair mirrored, make_shared_pair(*engine));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor targets,
                            Support::make_batch(*engine, K_STAT_WIDTH, 1.75));
    Graph::StateIdentity first_identity;
    Graph::StateIdentity second_identity;
    const Graph::StateID first = first_identity.get();
    const Graph::StateID second = second_identity.get();
    Graph::InMemoryDescentRecorder update(K_RATE);
    SA_EXPECT_OK(bind_shared_pair(session, eager, first, second, update));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<SharedCompiled> compiled,
                            make_shared_compiled(backend(), mirrored, targets));
    // The graph gives both parameters one gradient value, which is what the case needs.
    ASSERT_EQ(compiled->first_gradient, compiled->second_gradient);

    SA_EXPECT_OK(run_shared_steps(session, first, second, targets, update, K_BUFFER_STEPS));
    SA_EXPECT_OK(run_compiled_steps(*compiled->step, K_BUFFER_STEPS));

    const std::vector<unsigned char> first_expected = SaTest::read_tensor_bytes(mirrored.first);
    const std::vector<unsigned char> second_expected = SaTest::read_tensor_bytes(mirrored.second);
    EXPECT_EQ(SaTest::first_difference(SaTest::read_tensor_bytes(eager.first), first_expected),
              first_expected.size());
    EXPECT_EQ(SaTest::first_difference(SaTest::read_tensor_bytes(eager.second), second_expected),
              second_expected.size());
}

/** @brief Tests that a second live step stops the process. */
TEST_F(Medium_EagerSessionFatal, ASecondLiveStepStopsTheProcess)
{
    Support::Loop loop(backend(), K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_EXPECT_FATAL((void)loop.session.begin_step(), "still alive");
}

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_EagerStepCost : public SaTest::EngineTest
    {

    };

    /** @brief Holds a step size small enough that fifty layers stay finite. */
    constexpr float K_COST_RATE = 1e-6f;

    /** @brief Holds how many steps run before the measured ones, so every region is recorded. */
    constexpr int K_WARM_STEPS = 3;

    /** @brief Holds how many steps each measurement averages over. */
    constexpr int K_MEASURED_STEPS = 20;

    /** @brief Holds the layer count of the deep model; the design asks for at least 50. */
    constexpr std::size_t K_DEEP_LAYERS = 50;


    /** @brief Returns the widths of a model of @p layers layers, 16 wide inside, 4 results. */
    [[nodiscard]] std::vector<int64_t> widths_of(std::size_t layers)
    {
        std::vector<int64_t> widths(layers + 1, 16);
        widths.back() = 4;
        return widths;
    }

    /**
     * @brief Runs a settled loop and checks that its unchanged steps neither build nor record.
     * @return The failure of a step, or success once every measured step kept every region.
     */
    [[nodiscard]] Status run_settled(RuntimeExecutionBackend& backend, const std::string& label,
                                     const std::vector<int64_t>& widths,
                                     Support::Scoping scoping)
    {
        Support::Loop loop(backend, K_COST_RATE);
        SA_TRY(loop.prepare(backend.engine(), widths));

        Support::TraceOptions options;
        options.scoping = scoping;
        for (int step = 0; step < K_WARM_STEPS; ++step)
            SA_TRY(loop.step(options));

        for (int step = 0; step < K_MEASURED_STEPS; ++step)
        {
            SA_TRY(loop.step(options));
            SA_FAIL_IF(loop.session.last_cost().built != 0
                           || loop.session.last_cost().recorded != 0,
                       StatusCode::Incompatible,
                       "{}: a measured step built {} regions and recorded {} again; the loop "
                       "had not settled",
                       label, loop.session.last_cost().built,
                       loop.session.last_cost().recorded);
        }

        return Status();
    }

    /**
     * @brief Runs @p rounds rounds of one step with an extra scope and one without.
     * @return The number of regions the steps recorded again.
     */
    [[nodiscard]] std::size_t recorded_over_rounds(Support::Loop& loop,
                                                   Support::TraceOptions options, int rounds)
    {
        std::size_t recorded = 0;
        for (int round = 0; round < rounds; ++round)
        {
            options.extra_scope = true;
            SA_EXPECT_OK(loop.step(options));
            recorded += loop.session.last_cost().recorded;

            options.extra_scope = false;
            SA_EXPECT_OK(loop.step(options));
            recorded += loop.session.last_cost().recorded;
        }
        return recorded;
    }
} // namespace

TEST_F(Medium_EagerStepCost, AnUnchangedStepOfTheTwoLayerModel)
{
    SA_EXPECT_OK(run_settled(backend(), "two layers, no scope", {6, 4, 3},
                             Support::Scoping::None));
    SA_EXPECT_OK(run_settled(backend(), "two layers, one scope per layer", {6, 4, 3},
                             Support::Scoping::PerLayer));
}

TEST_F(Medium_EagerStepCost, AnUnchangedStepOfAFiftyLayerModel)
{
    SA_EXPECT_OK(run_settled(backend(), "fifty layers, no scope", widths_of(K_DEEP_LAYERS),
                             Support::Scoping::None));
    SA_EXPECT_OK(run_settled(backend(), "fifty layers, one scope per layer",
                             widths_of(K_DEEP_LAYERS), Support::Scoping::PerLayer));
}

/** @brief Tests that a scope appearing and disappearing records regions again. */
TEST_F(Medium_EagerStepCost, AScopeThatAppearsAndDisappears)
{
    Support::Loop loop(backend(), K_COST_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, widths_of(K_DEEP_LAYERS)));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;
    run_steps(loop, options, K_WARM_STEPS);

    const std::size_t recorded = recorded_over_rounds(loop, options, K_MEASURED_STEPS / 2);

    // A renumbering records regions again without compiling them; none means it was not one.
    EXPECT_GT(recorded, 0u);
}
