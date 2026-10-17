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
 * @file test_session.cpp
 * @brief Tests Session: running a model eagerly, region by region, and what a step costs.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "eager_run_support.hpp"
#include <SushiAI/SushiAI.hpp>
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
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

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

    /** @brief Holds the step size of every loop of this file. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};

    /** @brief Holds the widths of the three-layer model. */
    const std::vector<int64_t> K_THREE_LAYERS{6, 5, 4, 3};

    /** @brief Bounds the loss of a scoped loop against the unscoped one, in FLOAT32. */
    constexpr double K_LOSS_TOLERANCE = 1e-5;

    /** @brief Holds the same model and update compiled once through Graph::CompiledStep. */
    struct Compiled
    {
        /** @brief Builds the pool on @p engine; make_compiled() fills the rest. */
        explicit Compiled(SushiBLAS::Engine& engine) : pool(engine), update(K_RATE)
        {

        }

        Graph::Graph graph;                        /**< The differentiated, fused graph. */
        Graph::TensorPool pool;                    /**< Holds the tensor of each value. */
        Support::PlainDescent update;              /**< The update recorded behind the graph. */
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
    [[nodiscard]] Result<std::unique_ptr<Compiled>> make_compiled(SushiBLAS::Engine& engine)
    {
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

            const ValueID projected = builder.bias_add(builder.matmul(hidden, weight), bias);
            hidden = i + 2 < K_TWO_LAYERS.size() ? builder.relu(projected)
                                                 : builder.tanh(projected);
        }
        const ValueID targets =
            builder.input(Shape{Support::K_ROWS, K_TWO_LAYERS.back()}, "targets");
        made->loss = builder.nll_loss(hidden, targets);

        const std::vector<ValueID> gradients = Autograd::differentiate(
            made->graph, made->loss,
            SushiRuntime::span<const ValueID>(parameters.data(), parameters.size()));

        std::vector<ValueID> observed{inputs, targets, made->loss};
        observed.insert(observed.end(), parameters.begin(), parameters.end());
        observed.insert(observed.end(), gradients.begin(), gradients.end());
        std::vector<ValueID> pinned{made->loss};
        pinned.insert(pinned.end(), gradients.begin(), gradients.end());

        Graph::CastOptions casts;
        casts.observed = SushiRuntime::span<const ValueID>(observed.data(), observed.size());
        (void)Graph::insert_casts(made->graph, casts);
        Graph::FusionOptions fusion;
        fusion.observed = casts.observed;
        (void)Graph::fuse(made->graph, fusion);
        Graph::PlanOptions plan;
        plan.pinned = SushiRuntime::span<const ValueID>(pinned.data(), pinned.size());
        SA_TRY(made->pool.adopt(Graph::MemoryPlan::build(made->graph, plan)));

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

        SA_TRY_ASSIGN(made->step, Graph::CompiledStep::compile(engine, made->graph, made->pool,
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
} // namespace

TEST_F(Medium_EagerSession, AnUnscopedStepGivesTheLossBytesOfTheCompiledStep)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Compiled> compiled, make_compiled(*engine));
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    for (int step = 0; step < 4; ++step)
    {
        SA_EXPECT_OK(compiled->step->run());
        const std::vector<unsigned char> expected =
            SaTest::read_tensor_bytes(compiled->pool.tensor(compiled->loss));

        SA_ASSERT_OK_AND_ASSIGN(const std::vector<unsigned char> bytes, loop.step_bytes());
        EXPECT_EQ(SaTest::first_difference(bytes, expected), expected.size()) << "step " << step;
        EXPECT_EQ(bytes.size(), expected.size());
    }

    // With no scope the forward and backward nodes are one region.
    EXPECT_EQ(loop.executor.region_count(), 1u);
    EXPECT_EQ(loop.session.step_count(), 4u);
}

TEST_F(Medium_EagerSession, ASettledLoopComposesOnceAndAllocatesNothingMore)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_THREE_LAYERS));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;
    SA_EXPECT_OK(loop.step(options));

    const std::size_t regions = loop.executor.region_count();
    const std::size_t emits = loop.executor.emit_count();
    const std::size_t pool_bytes = loop.executor.allocated_bytes();
    const std::size_t allocations = loop.session.storage().allocation_count();
    const std::size_t tensors = loop.session.storage().size();
    EXPECT_GT(regions, 1u);
    EXPECT_EQ(loop.session.last_cost().built, regions);

    for (int step = 1; step < 5; ++step)
        SA_EXPECT_OK(loop.step(options));

    EXPECT_EQ(loop.executor.compose_count(), 1u);
    EXPECT_EQ(loop.executor.emit_count(), emits);
    EXPECT_EQ(loop.executor.allocated_bytes(), pool_bytes);
    EXPECT_EQ(loop.session.storage().allocation_count(), allocations);
    EXPECT_EQ(loop.session.storage().size(), tensors);
    EXPECT_EQ(loop.session.last_cost().kept, regions);
    EXPECT_EQ(loop.session.last_cost().built, 0u);
    EXPECT_EQ(loop.update.record_count(), 1u);
    EXPECT_EQ(loop.update.replay_count(), 5u);
}

TEST_F(Medium_EagerSession, FlippingABranchRebuildsThatScopesTwoRegionsAndNoOther)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::EncoderHead;
    SA_EXPECT_OK(loop.step(options));
    SA_EXPECT_OK(loop.step(options));

    options.head_is_relu = true;
    SA_EXPECT_OK(loop.step(options));

    EXPECT_EQ(loop.emit_count("head#0", Direction::Forward), 2u);
    EXPECT_EQ(loop.emit_count("head#0", Direction::Backward), 2u);
    EXPECT_EQ(loop.emit_count("encoder#0", Direction::Forward), 1u);
    EXPECT_EQ(loop.emit_count("encoder#0", Direction::Backward), 1u);

    // The loss is traced after the head closes, in the second unnamed segment of the step.
    EXPECT_EQ(loop.emit_count("#1", Direction::Forward), 1u);
    EXPECT_EQ(loop.session.last_cost().built, 2u);
    EXPECT_EQ(loop.session.last_cost().kept, 3u);
    EXPECT_EQ(loop.update.record_count(), 1u);
}

TEST_F(Medium_EagerSession, AScopeBetweenALinearLayerAndItsActivationTrainsToTheSameLoss)
{
    Support::Loop plain(*engine, K_RATE);
    Support::Loop scoped(*engine, K_RATE);
    SA_EXPECT_OK(plain.prepare(*engine, K_TWO_LAYERS));
    SA_EXPECT_OK(scoped.prepare(*engine, K_TWO_LAYERS));

    Support::TraceOptions split;
    split.scoping = Support::Scoping::SplitActivation;
    for (int step = 0; step < 5; ++step)
    {
        SA_ASSERT_OK_AND_ASSIGN(const double expected, plain.step());
        SA_ASSERT_OK_AND_ASSIGN(const double loss, scoped.step(split));
        EXPECT_NEAR(loss, expected, K_LOSS_TOLERANCE) << "step " << step;
        EXPECT_TRUE(std::isfinite(loss));
    }

    // No region of the scoped loop holds the linear layer and its activation together.
    for (const Eager::RegionOrder& order : scoped.session.last_plan().regions)
    {
        const Eager::RegionProgram* const program = scoped.executor.program_of(order.key);
        ASSERT_NE(program, nullptr) << "key " << order.key;
        EXPECT_EQ(program->fusion.count(Graph::FusedForm::GEMMBiasReLU), 0u)
            << "key " << order.key;
    }
    EXPECT_GT(scoped.executor.region_count(), plain.executor.region_count());
}

TEST_F(Medium_EagerSession, AScopeThatAppearsAndDisappearsRecordsTheRegionsItRenumbers)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::EncoderHead;
    SA_EXPECT_OK(loop.step(options));

    options.extra_scope = true;
    SA_EXPECT_OK(loop.step(options));
    EXPECT_GE(loop.session.last_cost().recorded, 1u);
    EXPECT_EQ(loop.update.record_count(), 2u);

    options.extra_scope = false;
    SA_ASSERT_OK_AND_ASSIGN(const double loss, loop.step(options));
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_EQ(loop.session.last_cost().dropped, 2u);
    EXPECT_EQ(loop.update.record_count(), 3u);

    // The encoder's forward key does not move; its backward key moves twice.
    EXPECT_EQ(loop.emit_count("encoder#0", Direction::Forward), 1u);
    EXPECT_EQ(loop.emit_count("encoder#0", Direction::Backward), 3u);
}

TEST_F(Medium_EagerSession, AParameterNoStepTracesKeepsAZeroGradientAndItsValue)
{
    Support::Loop loop(*engine, K_RATE);
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
    Support::Loop loop(*engine, K_RATE);
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

TEST_F(Medium_EagerSession, AValueTheStepDidNotKeepCannotBeReadAfterTheRun)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.backward(loss));
    SA_EXPECT_OK(step.apply(loop.update));

    // The loss reads the model's result inside the step's one region; keep() never named it.
    const ValueID interior = step.graph().node(step.graph().value(loss).producer).inputs[0];
    SA_EXPECT_FAILURE(step.read(interior), StatusCode::InvalidDescription, "not kept");
    SA_EXPECT_OK(step.read_scalar(loss));
    SA_EXPECT_FAILURE(step.keep(interior), StatusCode::Unsupported, "has run");
}

TEST_F(Medium_EagerSession, TracingAfterTheStepRanIsRefused)
{
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    Eager::Step step = loop.session.begin_step();
    SA_ASSERT_OK_AND_ASSIGN(const ValueID loss,
                            Support::trace_model(step, loop.model, loop.inputs, loop.targets));
    SA_EXPECT_OK(step.finish());

    (void)step.builder().scale(loss, 2.0);
    SA_EXPECT_FAILURE(step.read_scalar(loss), StatusCode::Unsupported, "after the step");
}

TEST_F(Medium_EagerSession, AStateTheSessionDoesNotHoldIsRefused)
{
    Support::Loop loop(*engine, K_RATE);
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
    Support::Loop loop(*engine, K_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, K_TWO_LAYERS));

    SA_EXPECT_FAILURE(
        loop.session.bind_state(loop.model[0].weight_state, loop.model[0].weight, true),
        StatusCode::InvalidDescription, "already bound");
}

TEST_F(Medium_EagerSession, ASecondLiveStepStopsTheProcess)
{
    Support::Loop loop(*engine, K_RATE);
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

    /** @brief Sums the costs of several steps. */
    struct CostSum
    {
        std::uint64_t partition_ns = 0; /**< Assigning region keys. */
        std::uint64_t plan_ns = 0;      /**< Finding boundaries and comparing signatures. */
        std::uint64_t compile_ns = 0;   /**< Running the passes over built regions. */
        std::uint64_t total_ns = 0;     /**< The whole flush. */
        std::size_t steps = 0;          /**< How many steps were added. */

        /** @brief Adds the cost of one step. */
        void add(const StepCost& cost) noexcept
        {
            partition_ns += cost.partition_ns;
            plan_ns += cost.plan_ns;
            compile_ns += cost.compile_ns;
            total_ns += cost.total_ns;
            ++steps;
        }
    };

    /** @brief Returns @p nanoseconds over @p steps, in microseconds. */
    [[nodiscard]] double mean_us(std::uint64_t nanoseconds, std::size_t steps)
    {
        return steps == 0 ? 0.0 : static_cast<double>(nanoseconds) / 1000.0
                                      / static_cast<double>(steps);
    }

    /** @brief Prints one row: the three host stages of a step and the whole step beside them. */
    void report(const std::string& label, const CostSum& sum, std::size_t regions)
    {
        const std::uint64_t host = sum.partition_ns + sum.plan_ns + sum.compile_ns;
        std::cout << "[eager step cost] " << label << ": regions " << regions
                  << ", partition " << mean_us(sum.partition_ns, sum.steps) << " us"
                  << ", plan " << mean_us(sum.plan_ns, sum.steps) << " us"
                  << ", compile " << mean_us(sum.compile_ns, sum.steps) << " us"
                  << ", host stages " << mean_us(host, sum.steps) << " us"
                  << ", whole step " << mean_us(sum.total_ns, sum.steps) << " us"
                  << " (mean of " << sum.steps << " steps)" << std::endl;
    }

    /** @brief Returns the widths of a model of @p layers layers, 16 wide inside, 4 results. */
    [[nodiscard]] std::vector<int64_t> widths_of(std::size_t layers)
    {
        std::vector<int64_t> widths(layers + 1, 16);
        widths.back() = 4;
        return widths;
    }

    /**
     * @brief Runs a settled loop and prints the mean host cost of its unchanged steps.
     * @return The failure of a step, or success once every measured step kept every region.
     */
    [[nodiscard]] Status measure_settled(SushiBLAS::Engine& engine, const std::string& label,
                                         const std::vector<int64_t>& widths,
                                         Support::Scoping scoping)
    {
        Support::Loop loop(engine, K_COST_RATE);
        SA_TRY(loop.prepare(engine, widths));

        Support::TraceOptions options;
        options.scoping = scoping;
        for (int step = 0; step < K_WARM_STEPS; ++step)
            SA_TRY(loop.step(options));

        CostSum sum;
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
            sum.add(loop.session.last_cost());
        }

        report(label, sum, loop.executor.region_count());
        return Status();
    }
} // namespace

TEST_F(Medium_EagerStepCost, AnUnchangedStepOfTheTwoLayerModel)
{
    SA_EXPECT_OK(measure_settled(*engine, "two layers, no scope", {6, 4, 3},
                                 Support::Scoping::None));
    SA_EXPECT_OK(measure_settled(*engine, "two layers, one scope per layer", {6, 4, 3},
                                 Support::Scoping::PerLayer));
}

TEST_F(Medium_EagerStepCost, AnUnchangedStepOfAFiftyLayerModel)
{
    SA_EXPECT_OK(measure_settled(*engine, "fifty layers, no scope", widths_of(K_DEEP_LAYERS),
                                 Support::Scoping::None));
    SA_EXPECT_OK(measure_settled(*engine, "fifty layers, one scope per layer",
                                 widths_of(K_DEEP_LAYERS), Support::Scoping::PerLayer));
}

TEST_F(Medium_EagerStepCost, AScopeThatAppearsAndDisappears)
{
    Support::Loop loop(*engine, K_COST_RATE);
    SA_EXPECT_OK(loop.prepare(*engine, widths_of(K_DEEP_LAYERS)));

    Support::TraceOptions options;
    options.scoping = Support::Scoping::PerLayer;
    for (int step = 0; step < K_WARM_STEPS; ++step)
        SA_EXPECT_OK(loop.step(options));

    CostSum appears;
    CostSum disappears;
    std::size_t recorded = 0;
    for (int round = 0; round < K_MEASURED_STEPS / 2; ++round)
    {
        options.extra_scope = true;
        SA_EXPECT_OK(loop.step(options));
        appears.add(loop.session.last_cost());
        recorded += loop.session.last_cost().recorded;

        options.extra_scope = false;
        SA_EXPECT_OK(loop.step(options));
        disappears.add(loop.session.last_cost());
        recorded += loop.session.last_cost().recorded;
    }

    report("fifty layers, a scope appears", appears, loop.executor.region_count());
    report("fifty layers, a scope disappears", disappears, loop.executor.region_count());

    // A renumbering records regions again without compiling them; none means it was not one.
    EXPECT_GT(recorded, 0u);
}
