/**************************************************************************/
/* test_compiled_step.cpp                                                 */
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
 * @file test_compiled_step.cpp
 * @brief Tests a compiled step's run index and the label and node origins it gives its graph.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/report/run_observer.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/node_range.hpp>
#include <SushiAI/graph/run_counter.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/core/graph/README.md.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueID;

namespace
{
    /** @brief Provides the Engine every case compiles on. */
    class Medium_CompiledStep : public SaTest::EngineTest
    {

    };

    /** @brief Provides a host backend, so a step is compiled and run with no device. */
    class Small_CompiledStep : public ::testing::Test
    {
        protected:
            /** @brief Holds the host adapter the pool allocates from and the step runs on. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the element count of the masked value. */
    constexpr std::size_t K_ELEMENTS = 256;

    /** @brief Holds the drop rate; a kept element is scaled by 4/3. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds the probability that an element is kept. */
    constexpr double K_KEEP = 1.0 - K_RATE;

    /** @brief Holds the site seed of the one dropout of every case. */
    constexpr std::uint64_t K_SEED = 0x5EEDC0DE20261011ull;

    /** @brief Holds the value a kept element of the mask carries, as FLOAT32 stores it. */
    constexpr double K_KEPT_VALUE = static_cast<double>(static_cast<float>(1.0 / K_KEEP));

    /** @brief Holds a compiled step that masks a leaf of ones, so its output is the mask. */
    struct MaskedOnes
    {
        /** @brief Builds the pool on @p engine; make_masked_ones() fills the rest. */
        explicit MaskedOnes(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief Holds the leaf and the dropout node. */
        Graph::Graph graph;

        /** @brief Holds the tensor of each value. */
        TensorPool pool;

        /** @brief Names the masked value. */
        ValueID masked = Graph::K_NO_VALUE;

        /** @brief Holds the compiled step. */
        std::unique_ptr<CompiledStep> step;
    };

    /** @brief Compiles a dropout over K_ELEMENTS ones with the site seed K_SEED. */
    Result<std::unique_ptr<MaskedOnes>> make_masked_ones(SushiAI::ExecutionBackend& backend)
    {
        std::unique_ptr<MaskedOnes> made = std::make_unique<MaskedOnes>(backend.engine());
        const GraphBuilder builder(made->graph);

        const ValueID ones =
            builder.parameter(Shape{static_cast<int64_t>(K_ELEMENTS)}, "ones");
        SA_TRY_ASSIGN(const Graph::DropoutResult dropped, builder.dropout(ones, K_RATE, K_SEED));
        made->masked = dropped.output;

        SA_TRY(made->pool.materialise_leaves(made->graph));
        SA_TRY_ASSIGN(made->step, CompiledStep::compile(backend, made->graph, made->pool));
        SaTest::write_tensor(made->pool.tensor(ones), std::vector<double>(K_ELEMENTS, 1.0));
        return std::move(made);
    }

    /** @brief Expects every element of @p mask to be 0 or K_KEPT_VALUE. */
    void expect_zero_or_scale(const std::vector<double>& mask)
    {
        EXPECT_TRUE(std::all_of(mask.begin(), mask.end(),
                                [](double value) { return value == 0.0 || value == K_KEPT_VALUE; }));
    }

    /** @brief Runs @p made once and returns the masked value, which is the mask of that run. */
    std::vector<double> run_and_read(MaskedOnes& made)
    {
        SA_EXPECT_OK(made.step->run());
        return SaTest::read_tensor(made.pool.tensor(made.masked));
    }

    /** @brief Runs @p made at run index @p run and returns the mask of that run. */
    std::vector<double> run_at_and_read(MaskedOnes& made, std::uint64_t run)
    {
        SA_EXPECT_OK(made.step->run_at(run));
        return SaTest::read_tensor(made.pool.tensor(made.masked));
    }
} // namespace

/** @brief Tests that a step restored to a count draws the mask of that run next. */
TEST_F(Medium_CompiledStep, ARestoredRunCountIsTheIndexOfTheNextRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> whole, make_masked_ones(backend()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> resumed,
                            make_masked_ones(backend()));
    ASSERT_EQ(whole->step->replay_prelude_count(), 1u);

    SA_EXPECT_OK(whole->step->run());
    SA_EXPECT_OK(whole->step->run());
    SA_EXPECT_OK(whole->step->run());
    const std::vector<double> third = SaTest::read_tensor(whole->pool.tensor(whole->masked));

    EXPECT_EQ(resumed->step->run_count(), 0u);
    SA_EXPECT_OK(resumed->step->restore_run_count(2));
    EXPECT_EQ(resumed->step->run_count(), 2u);
    SA_EXPECT_OK(resumed->step->run());

    EXPECT_EQ(resumed->step->run_count(), 3u);
    EXPECT_EQ(SaTest::read_tensor(resumed->pool.tensor(resumed->masked)), third);
    expect_zero_or_scale(third);
}

/** @brief Tests that three runs of one compiled step draw three different masks. */
TEST_F(Medium_CompiledStep, RunDrawsADifferentMaskAtEachRunCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(backend()));

    const std::vector<double> first = run_and_read(*made);
    const std::vector<double> second = run_and_read(*made);
    const std::vector<double> third = run_and_read(*made);

    expect_zero_or_scale(first);
    expect_zero_or_scale(second);
    expect_zero_or_scale(third);
    EXPECT_NE(first, second);
    EXPECT_NE(second, third);
    EXPECT_NE(first, third);
    EXPECT_EQ(made->step->compile_count(), 1u);
}

/** @brief Tests that run_at() draws the mask of the index it is given and counts the run. */
TEST_F(Medium_CompiledStep, RunAtDrawsTheMaskOfItsIndexAndCountsTheRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(backend()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> counted,
                            make_masked_ones(backend()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> indexed,
                            make_masked_ones(backend()));

    const std::vector<double> seventh = run_at_and_read(*made, 7);
    EXPECT_EQ(made->step->run_count(), 1u);
    const std::vector<double> after = run_and_read(*made);

    // A fresh step reaches run index 1 by counting, and index 7 by being told.
    static_cast<void>(run_and_read(*counted));
    EXPECT_EQ(run_and_read(*counted), after);
    EXPECT_EQ(run_at_and_read(*indexed, 7), seventh);
    EXPECT_NE(seventh, after);
}

/** @brief Tests that a step that has run refuses a restore and keeps its count. */
TEST_F(Small_CompiledStep, ARestoreAfterARunIsRefusedWithTheCount)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    static_cast<void>(builder.scale(leaf, 2.0));
    TensorPool pool(backend.engine());
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend, graph, pool));

    SA_EXPECT_OK(step->run());
    SA_EXPECT_FAILURE(step->restore_run_count(4), StatusCode::InvalidDescription,
                      "before its first run", "run 1");
    EXPECT_EQ(step->run_count(), 1u);
}

/** @brief Tests that a graph whose refused emitter result was dropped may not be compiled. */
TEST(Small_CompiledStepChecks, AGraphWithADroppedRefusalIsNotCompilable)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, builder.scale(leaf, 2.0));
    SA_EXPECT_OK(Graph::check_compilable(graph));

    static_cast<void>(builder.reshape(doubled, Shape{static_cast<int64_t>(5)}));

    SA_EXPECT_FAILURE(Graph::check_compilable(graph), StatusCode::InvalidDescription,
                      "refused operation");
}

/** @brief Tests that two ranges of a graph run as two steps over the tensors of one pool. */
TEST_F(Medium_CompiledStep, TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, builder.scale(leaf, 2.0));
    SA_ASSERT_OK_AND_ASSIGN(const ValueID tripled, builder.scale(doubled, 3.0));
    const Graph::NodeID cut = graph.value(doubled).producer + 1;
    ASSERT_LT(cut, graph.node_count());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> first,
        CompiledStep::compile(backend(), graph, pool, nullptr, nullptr,
                              Graph::NodeRange{0, cut}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> second,
        CompiledStep::compile(backend(), graph, pool, nullptr, nullptr,
                              Graph::NodeRange{cut, Graph::K_NO_NODE}));
    EXPECT_GE(first->operation_count(), 1u);
    EXPECT_GE(second->operation_count(), 1u);

    SaTest::write_tensor(pool.tensor(leaf), std::vector<double>{1.0, 2.0, 3.0, 4.0});
    SA_EXPECT_OK(first->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(doubled)),
              (std::vector<double>{2.0, 4.0, 6.0, 8.0}));

    // The host writes between the two steps, and the second reads what it wrote.
    SaTest::write_tensor(pool.tensor(doubled), std::vector<double>(4, 10.0));
    SA_EXPECT_OK(second->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(tripled)),
              (std::vector<double>{30.0, 30.0, 30.0, 30.0}));
}

/** @brief Tests that a range that ends behind the last node is refused with its bounds. */
TEST_F(Small_CompiledStep, ARangeOutsideTheGraphIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    static_cast<void>(builder.scale(leaf, 2.0));

    TensorPool pool(backend.engine());
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    const Graph::NodeID count = static_cast<Graph::NodeID>(graph.node_count());
    SA_EXPECT_FAILURE(CompiledStep::compile(backend, graph, pool, nullptr, nullptr,
                                            Graph::NodeRange{0, count + 1}),
                      StatusCode::InvalidDescription, "does not lie in a graph");
    SA_EXPECT_FAILURE(
        CompiledStep::compile(backend, graph, pool, nullptr, nullptr, Graph::NodeRange{1, 0}),
        StatusCode::InvalidDescription, "[1, 0)");
}

/** @brief Tests that compile refuses a graph whose refused emitter result was dropped. */
TEST_F(Small_CompiledStep, AGraphWithADroppedRefusalIsNotCompiled)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    SA_ASSERT_OK_AND_ASSIGN(const ValueID doubled, builder.scale(leaf, 2.0));
    static_cast<void>(builder.reshape(doubled, Shape{static_cast<int64_t>(5)}));
    ASSERT_TRUE(graph.has_refusal());

    TensorPool pool(backend.engine());
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_EXPECT_FAILURE(CompiledStep::compile(backend, graph, pool), StatusCode::InvalidDescription,
                      "refused operation");
}

namespace
{
    /** @brief Holds the element count of the chain's input. */
    constexpr int64_t K_CHAIN_ELEMENTS = 8;

    /** @brief Holds the origin the trailing work carries. */
    constexpr std::string_view K_TRAILING_ORIGIN = "step.trailing";

    /** @brief Holds what the observer kept of one run. */
    struct SeenRun
    {
        std::shared_ptr<const SushiRuntime::NodeTable> table; /**< The plan's node table. */
        std::uint64_t graph_id = 0;                            /**< The id of the plan's graph. */
    };

    /** @brief Keeps the node table and graph id of every run it hears. */
    class RunCollector final : public SushiRuntime::RunObserver
    {
        public:
            /** @brief Keeps the table and the graph id of @p observation. */
            void on_run(const SushiRuntime::RunObservation& observation) noexcept override
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                try
                {
                    runs_.push_back(SeenRun{observation.table, observation.graph_id});
                }
                catch (const std::bad_alloc&)
                {
                    lost_ = true;
                }
            }

            /** @brief Returns the runs heard so far. */
            std::vector<SeenRun> runs() const
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                return runs_;
            }

            /** @brief Returns whether a run could not be kept. */
            bool lost() const
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                return lost_;
            }

        private:
            /// @brief Guards the members; the runtime calls from the thread that finished a run.
            mutable std::mutex mutex_;

            /// @brief The runs heard, in order.
            std::vector<SeenRun> runs_;

            /// @brief Whether a run was not kept for want of memory.
            bool lost_ = false;
    };

    /** @brief Copies the first activation back over the input, as an optimizer writes a weight. */
    class CopyBack final : public Graph::StepRecorder
    {
        public:
            /** @brief Binds the recorder to the two tensors, which outlive it. */
            CopyBack(const SushiBLAS::Tensor& from, SushiBLAS::Tensor& to) : from_(&from), to_(&to)
            {

            }

            /** @brief Records one copy and reports one task. */
            [[nodiscard]] Result<std::size_t> record_into_step(
                const Graph::StepRecording& recording) override
            {
                SushiBLAS::BLASOps blas(recording.tasks());
                blas.copy(*from_, *to_);
                return std::size_t{1};
            }

        private:
            /// @brief The tensor copied from.
            const SushiBLAS::Tensor* from_;

            /// @brief The tensor copied over.
            SushiBLAS::Tensor* to_;
    };

    /** @brief Holds relu, reshape and relu over one input, and the step compiled from them. */
    struct Chain
    {
        /** @brief Builds the pool on @p engine; make_chain() fills the rest. */
        explicit Chain(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief Holds the leaf and the three nodes. */
        Graph::Graph graph;

        /** @brief Holds the tensor of each value. */
        TensorPool pool;

        /** @brief Names the input. */
        ValueID input = Graph::K_NO_VALUE;

        /** @brief Names the result of the first relu. */
        ValueID first = Graph::K_NO_VALUE;

        /** @brief Names the reshaped value, which a view produces. */
        ValueID view = Graph::K_NO_VALUE;

        /** @brief Names the result of the second relu. */
        ValueID second = Graph::K_NO_VALUE;

        /** @brief Holds the trailing recorder, or null; the step points at it. */
        std::unique_ptr<CopyBack> trailing;

        /** @brief Holds the compiled step. */
        std::unique_ptr<CompiledStep> step;
    };

    /** @brief Compiles the chain as the step @p label, with trailing work behind it on request. */
    Result<std::unique_ptr<Chain>> make_chain(SushiAI::ExecutionBackend& backend,
                                              std::string_view label,
                                              bool with_trailing)
    {
        std::unique_ptr<Chain> made = std::make_unique<Chain>(backend.engine());
        const GraphBuilder builder(made->graph);
        made->input = builder.parameter(Shape{K_CHAIN_ELEMENTS}, "input");
        SA_TRY_ASSIGN(made->first, builder.relu(made->input));
        SA_TRY_ASSIGN(made->view, builder.reshape(made->first, Shape{2, K_CHAIN_ELEMENTS / 2}));
        SA_TRY_ASSIGN(made->second, builder.relu(made->view));

        // A step runs on a planned pool, where the reshape's result is a view of its operand.
        Graph::PlanOptions plan_options;
        plan_options.reuse = false;
        SA_TRY(made->pool.adopt(Graph::MemoryPlan::build(made->graph, plan_options).value()));
        SA_TRY(made->pool.materialise_leaves(made->graph));
        if (with_trailing)
        {
            // The trailing recorder is bound before lowering, so it materialises what it reads.
            SA_TRY_ASSIGN(SushiBLAS::Tensor& first,
                          made->pool.materialise(made->first, made->graph.value(made->first)));
            made->trailing = std::make_unique<CopyBack>(first, made->pool.tensor(made->input));
        }
        SA_TRY_ASSIGN(made->step,
                      CompiledStep::compile(backend, made->graph, made->pool, made->trailing.get(),
                                            nullptr, {}, label));
        SaTest::write_tensor(made->pool.tensor(made->input),
                             std::vector<double>(static_cast<std::size_t>(K_CHAIN_ELEMENTS), 1.0));
        return std::move(made);
    }

    /** @brief Returns the origin the lowering gives node @p id of @p graph. */
    std::string origin_of(const Graph::Graph& graph, Graph::NodeID id)
    {
        return std::to_string(id) + ":" + std::string(Ops::name_of(graph.node(id).op));
    }

    /** @brief Returns the origin of the node that produces @p value in @p graph. */
    std::string origin_of_producer(const Graph::Graph& graph, ValueID value)
    {
        return origin_of(graph, graph.value(value).producer);
    }

    /** @brief Returns the origin of every node of @p graph. */
    std::set<std::string> node_origins_of(const Graph::Graph& graph)
    {
        std::set<std::string> known;
        for (Graph::NodeID id = 0; id < static_cast<Graph::NodeID>(graph.node_count()); ++id)
            known.insert(origin_of(graph, id));
        return known;
    }

    /** @brief Expects every origin of @p origins to be non-empty and to be in @p known. */
    void expect_origins_known(const std::multiset<std::string>& origins,
                              const std::set<std::string>& known)
    {
        for (const std::string& origin : origins)
        {
            EXPECT_FALSE(origin.empty());
            EXPECT_EQ(known.count(origin), 1u) << "'" << origin << "' names no node of the graph";
        }
    }

    /** @brief Returns the origins of the rows of @p table. */
    std::multiset<std::string> origins_of(const SushiRuntime::NodeTable& table)
    {
        std::multiset<std::string> origins;
        for (const SushiRuntime::NodeTableRow& row : table.rows())
            origins.insert(row.origin);
        return origins;
    }

    /** @brief Provides a host backend, which keeps the origin of each task it records. */
    class Small_CompiledStepAttribution : public ::testing::Test
    {
        protected:
            /** @brief Holds the host adapter the chain is compiled on. */
            HostExecutionBackend backend;
    };

    /** @brief Provides the Engine every case compiles on. */
    class Medium_CompiledStepAttribution : public SaTest::EngineTest
    {
        protected:
            /** @brief Runs the steps "first.step" and "second.step" and returns the runs heard. */
            std::vector<SeenRun> observe_two_steps()
            {
                const auto collector = std::make_shared<RunCollector>();
                const SushiRuntime::RunObserverHandle handle =
                    SaTest::TestEnvironment::runtime().add_run_observer(collector);
                Result<std::unique_ptr<Chain>> first = make_chain(backend(), "first.step", false);
                Result<std::unique_ptr<Chain>> second = make_chain(backend(), "second.step", false);
                if (!first.has_value() || !second.has_value())
                    return {};

                SA_EXPECT_OK(first.value()->step->run());
                flush();
                SA_EXPECT_OK(second.value()->step->run());
                flush();
                if (collector->lost())
                    return {};

                return collector->runs();
            }
    };
} // namespace

/** @brief Tests that no task of a step has an empty origin and each names a node of its graph. */
TEST_F(Medium_CompiledStepAttribution, EveryTaskOfANodeCarriesItsOrigin)
{
    const auto collector = std::make_shared<RunCollector>();
    const SushiRuntime::RunObserverHandle handle =
        SaTest::TestEnvironment::runtime().add_run_observer(collector);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(backend(), "chain.step", false));

    SA_EXPECT_OK(chain->step->run());
    flush();

    ASSERT_FALSE(collector->lost());
    const std::vector<SeenRun> runs = collector->runs();
    ASSERT_EQ(runs.size(), 1u);
    ASSERT_NE(runs[0].table, nullptr);
    const std::multiset<std::string> origins = origins_of(*runs[0].table);
    ASSERT_FALSE(origins.empty());
    expect_origins_known(origins, node_origins_of(chain->graph));
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->first)), 1u);
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->second)), 1u);
    EXPECT_EQ(origins.count(std::string(K_TRAILING_ORIGIN)), 0u);
}

/** @brief Tests that two steps keep their own labels. */
TEST_F(Medium_CompiledStepAttribution, TwoStepsHaveTwoLabels)
{
    const std::vector<SeenRun> runs = observe_two_steps();

    ASSERT_EQ(runs.size(), 2u);
    ASSERT_NE(runs[0].table, nullptr);
    ASSERT_NE(runs[1].table, nullptr);
    EXPECT_EQ(runs[0].table->label(), "first.step");
    EXPECT_EQ(runs[1].table->label(), "second.step");
}

/** @brief Tests that two steps over equal node indices keep apart by graph id. */
TEST_F(Medium_CompiledStepAttribution, TwoStepsHaveTwoGraphIds)
{
    const std::vector<SeenRun> runs = observe_two_steps();

    ASSERT_EQ(runs.size(), 2u);
    EXPECT_NE(runs[0].graph_id, 0u);
    EXPECT_NE(runs[0].graph_id, runs[1].graph_id);
}

/** @brief Tests that two steps of equal graphs carry equal origins. */
TEST_F(Medium_CompiledStepAttribution, TwoStepsOfEqualGraphsCarryEqualOrigins)
{
    const std::vector<SeenRun> runs = observe_two_steps();

    ASSERT_EQ(runs.size(), 2u);
    ASSERT_NE(runs[0].table, nullptr);
    ASSERT_NE(runs[1].table, nullptr);
    EXPECT_EQ(origins_of(*runs[0].table), origins_of(*runs[1].table));
}

/** @brief Tests that the work a trailing recorder adds carries the origin step.trailing. */
TEST_F(Small_CompiledStepAttribution, TheOptimizerCarriesTheTrailingOrigin)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(backend, "chain.step", true));

    const std::vector<std::string> recorded = chain->step->task_origins();

    const std::multiset<std::string> origins(recorded.begin(), recorded.end());
    EXPECT_EQ(origins.count(std::string(K_TRAILING_ORIGIN)), 1u);
    EXPECT_EQ(origins.count(std::string()), 0u);
}

/** @brief Tests that the node behind a view records nothing and leaves no origin to the next. */
TEST_F(Small_CompiledStepAttribution, AViewLeavesNoOriginOpen)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(backend, "chain.step", false));

    const std::vector<std::string> recorded = chain->step->task_origins();

    const std::multiset<std::string> origins(recorded.begin(), recorded.end());
    EXPECT_EQ(origins.count(origin_of_producer(chain->graph, chain->view)), 0u);
    EXPECT_EQ(origins.count(std::string()), 0u);
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->second)), 1u);
}

namespace
{
    /** @brief Holds the element count of the tensor the scaling recorder scales. */
    constexpr int64_t K_SCALED_ELEMENTS = 4;

    /** @brief Holds the factor each replay of the scaling recorder multiplies by. */
    constexpr float K_SCALE = 2.0f;

    /** @brief Records one scal of a tensor, the work of a step that holds no graph node. */
    class ScaleInPlace final : public Graph::StepRecorder
    {
        public:
            /** @brief Binds the recorder to @p target, which outlives it. */
            explicit ScaleInPlace(SushiBLAS::Tensor& target) : target_(&target)
            {

            }

            /** @brief Records one scal and reports one task. */
            [[nodiscard]] Result<std::size_t> record_into_step(
                const Graph::StepRecording& recording) override
            {
                SushiBLAS::BLASOps blas(recording.tasks());
                blas.scal(K_SCALE, *target_);
                return std::size_t{1};
            }

        private:
            /// @brief The tensor scaled.
            SushiBLAS::Tensor* target_;
    };
} // namespace

/** @brief Tests that a step of an empty graph and one recorder runs the recorder every replay. */
TEST_F(Small_CompiledStep, AnEmptyGraphWithATrailingRecorderRunsTheRecorderEachRun)
{
    // The tensor lives in a pool that another graph's leaf materialised; the step's graph is empty.
    Graph::Graph holder;
    const GraphBuilder holder_builder(holder);
    const ValueID leaf = holder_builder.parameter(Shape{K_SCALED_ELEMENTS}, "leaf");
    TensorPool pool(backend.engine());
    SA_EXPECT_OK(pool.materialise_leaves(holder));
    SaTest::write_tensor(pool.tensor(leaf),
                         std::vector<double>(static_cast<std::size_t>(K_SCALED_ELEMENTS), 1.0));

    const Graph::Graph empty;
    ScaleInPlace recorder(pool.tensor(leaf));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<CompiledStep> step,
                            CompiledStep::compile(backend, empty, pool, &recorder));

    EXPECT_EQ(step->operation_count(), 1u);
    EXPECT_FALSE(step->empty());

    const std::size_t elements = static_cast<std::size_t>(K_SCALED_ELEMENTS);
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaf)), std::vector<double>(elements, 2.0));
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaf)), std::vector<double>(elements, 4.0));
    SA_EXPECT_OK(step->run());
    EXPECT_EQ(SaTest::read_tensor(pool.tensor(leaf)), std::vector<double>(elements, 8.0));
    EXPECT_EQ(step->run_count(), 3u);
}
