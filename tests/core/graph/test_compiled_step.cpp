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
 * @file test_compiled_step.cpp
 * @brief Tests a compiled step's run index and the label and node origins it gives its graph.
 * @author Mustafa Garip
 */

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
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/node_range.hpp>
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

    /** @brief Holds the element count of the masked value. */
    constexpr std::size_t K_ELEMENTS = 256;

    /** @brief Holds the drop rate; a kept element is scaled by 4/3. */
    constexpr double K_RATE = 0.25;

    /** @brief Holds the probability that an element is kept. */
    constexpr double K_KEEP = 1.0 - K_RATE;

    /** @brief Holds the site seed of the one dropout of every case. */
    constexpr std::uint64_t K_SEED = 0x5EEDC0DE20261011ull;

    /** @brief Bounds a FLOAT32 scale read back as a double. */
    constexpr double K_TOLERANCE = 1e-6;

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
    Result<std::unique_ptr<MaskedOnes>> make_masked_ones(SushiBLAS::Engine& engine)
    {
        std::unique_ptr<MaskedOnes> made = std::make_unique<MaskedOnes>(engine);
        const GraphBuilder builder(made->graph);

        const ValueID ones =
            builder.parameter(Shape{static_cast<int64_t>(K_ELEMENTS)}, "ones");
        SA_TRY_ASSIGN(const Graph::DropoutResult dropped, builder.dropout(ones, K_RATE, K_SEED));
        made->masked = dropped.output;

        SA_TRY(made->pool.materialise_leaves(made->graph));
        SA_TRY_ASSIGN(made->step, CompiledStep::compile(engine, made->graph, made->pool));
        SaTest::write_tensor(made->pool.tensor(ones), std::vector<double>(K_ELEMENTS, 1.0));
        return std::move(made);
    }

    /** @brief Expects @p mask to hold what the generator draws for K_SEED in run @p run. */
    void expect_mask_of_run(const std::vector<double>& mask, std::uint64_t run)
    {
        ASSERT_EQ(mask.size(), K_ELEMENTS);
        for (std::size_t element = 0; element < K_ELEMENTS; ++element)
        {
            const double expected =
                Graph::dropout_keeps(K_SEED, run, element, K_KEEP) ? 1.0 / K_KEEP : 0.0;
            EXPECT_NEAR(mask[element], expected, K_TOLERANCE) << "at element " << element;
        }
    }
} // namespace

/** @brief Tests that a step restored to a count draws the mask of that run next. */
TEST_F(Medium_CompiledStep, ARestoredRunCountIsTheIndexOfTheNextRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> whole, make_masked_ones(*engine));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> resumed,
                            make_masked_ones(*engine));
    ASSERT_EQ(whole->step->replay_prelude_count(), 1u);

    for (int run = 0; run < 3; ++run)
        SA_EXPECT_OK(whole->step->run());
    const std::vector<double> third = SaTest::read_tensor(whole->pool.tensor(whole->masked));

    EXPECT_EQ(resumed->step->run_count(), 0u);
    SA_EXPECT_OK(resumed->step->restore_run_count(2));
    EXPECT_EQ(resumed->step->run_count(), 2u);
    SA_EXPECT_OK(resumed->step->run());

    EXPECT_EQ(resumed->step->run_count(), 3u);
    EXPECT_EQ(SaTest::read_tensor(resumed->pool.tensor(resumed->masked)), third);
    expect_mask_of_run(third, 2);
}

/** @brief Tests that run() draws the mask of its run count, so single-machine draws stay put. */
TEST_F(Medium_CompiledStep, RunDrawsTheMaskOfItsRunCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    for (std::uint64_t run = 0; run < 3; ++run)
    {
        SA_EXPECT_OK(made->step->run());
        expect_mask_of_run(SaTest::read_tensor(made->pool.tensor(made->masked)), run);
    }
}

/** @brief Tests that run_at() draws the mask of the index it is given and counts the run. */
TEST_F(Medium_CompiledStep, RunAtDrawsTheMaskOfItsIndexAndCountsTheRun)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    SA_EXPECT_OK(made->step->run_at(7));
    expect_mask_of_run(SaTest::read_tensor(made->pool.tensor(made->masked)), 7);
    EXPECT_EQ(made->step->run_count(), 1u);

    SA_EXPECT_OK(made->step->run());
    expect_mask_of_run(SaTest::read_tensor(made->pool.tensor(made->masked)), 1);
}

/** @brief Tests that a step that has run refuses a restore and keeps its count. */
TEST_F(Medium_CompiledStep, ARestoreAfterARunIsRefusedWithTheCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    SA_EXPECT_OK(made->step->run());
    SA_EXPECT_FAILURE(made->step->restore_run_count(4), StatusCode::InvalidDescription,
                      "before its first run", "run 1");
    EXPECT_EQ(made->step->run_count(), 1u);
}

/** @brief Tests that a count above 0 is restored once, and that 0 leaves a new step new. */
TEST_F(Medium_CompiledStep, ASecondRestoreIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<MaskedOnes> made, make_masked_ones(*engine));

    SA_EXPECT_OK(made->step->restore_run_count(0));
    SA_EXPECT_OK(made->step->restore_run_count(5));
    SA_EXPECT_FAILURE(made->step->restore_run_count(7), StatusCode::InvalidDescription,
                      "run 5");
    EXPECT_EQ(made->step->run_count(), 5u);
}

/** @brief Tests that two ranges of a graph run as two steps over the tensors of one pool. */
TEST_F(Medium_CompiledStep, TheSecondRangeReadsWhatThePoolHoldsAfterTheFirst)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    const ValueID doubled = builder.scale(leaf, 2.0);
    const ValueID tripled = builder.scale(doubled, 3.0);
    const Graph::NodeID cut = graph.value(doubled).producer + 1;
    ASSERT_LT(cut, graph.node_count());

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> first,
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                              Graph::NodeRange{0, cut}));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<CompiledStep> second,
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
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
TEST_F(Medium_CompiledStep, ARangeOutsideTheGraphIsRefused)
{
    Graph::Graph graph;
    const GraphBuilder builder(graph);
    const ValueID leaf = builder.parameter(Shape{static_cast<int64_t>(4)}, "leaf");
    static_cast<void>(builder.scale(leaf, 2.0));

    TensorPool pool(*engine);
    SA_EXPECT_OK(pool.materialise_leaves(graph));
    const Graph::NodeID count = static_cast<Graph::NodeID>(graph.node_count());
    SA_EXPECT_FAILURE(CompiledStep::compile(*engine, graph, pool, nullptr, nullptr,
                                            Graph::NodeRange{0, count + 1}),
                      StatusCode::InvalidDescription, "does not lie in a graph");
    SA_EXPECT_FAILURE(
        CompiledStep::compile(*engine, graph, pool, nullptr, nullptr, Graph::NodeRange{1, 0}),
        StatusCode::InvalidDescription, "[1, 0)");
}

namespace
{
    /** @brief Provides the Engine every case compiles on. */
    class Medium_CompiledStepAttribution : public SaTest::EngineTest
    {

    };

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
    Result<std::unique_ptr<Chain>> make_chain(SushiBLAS::Engine& engine, std::string_view label,
                                              bool with_trailing)
    {
        std::unique_ptr<Chain> made = std::make_unique<Chain>(engine);
        const GraphBuilder builder(made->graph);
        made->input = builder.parameter(Shape{K_CHAIN_ELEMENTS}, "input");
        made->first = builder.relu(made->input);
        made->view = builder.reshape(made->first, Shape{2, K_CHAIN_ELEMENTS / 2});
        made->second = builder.relu(made->view);

        SA_TRY(made->pool.materialise_leaves(made->graph));
        if (with_trailing)
        {
            made->trailing = std::make_unique<CopyBack>(made->pool.tensor(made->first),
                                                        made->pool.tensor(made->input));
        }
        SA_TRY_ASSIGN(made->step,
                      CompiledStep::compile(engine, made->graph, made->pool, made->trailing.get(),
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

    /** @brief Returns the origins of the rows of @p table. */
    std::multiset<std::string> origins_of(const SushiRuntime::NodeTable& table)
    {
        std::multiset<std::string> origins;
        for (const SushiRuntime::NodeTableRow& row : table.rows())
            origins.insert(row.origin);
        return origins;
    }
} // namespace

/** @brief Tests that no task of a step has an empty origin and each names a node of its graph. */
TEST_F(Medium_CompiledStepAttribution, EveryTaskOfANodeCarriesItsOrigin)
{
    const auto collector = std::make_shared<RunCollector>();
    const SushiRuntime::RunObserverHandle handle =
        engine->runtime().add_run_observer(collector);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(*engine, "chain.step", false));

    SA_EXPECT_OK(chain->step->run());
    flush();

    ASSERT_FALSE(collector->lost());
    const std::vector<SeenRun> runs = collector->runs();
    ASSERT_EQ(runs.size(), 1u);
    ASSERT_NE(runs[0].table, nullptr);
    std::set<std::string> known;
    for (Graph::NodeID id = 0; id < static_cast<Graph::NodeID>(chain->graph.node_count()); ++id)
        known.insert(origin_of(chain->graph, id));
    const std::multiset<std::string> origins = origins_of(*runs[0].table);
    ASSERT_FALSE(origins.empty());
    for (const std::string& origin : origins)
    {
        EXPECT_FALSE(origin.empty());
        EXPECT_EQ(known.count(origin), 1u) << "'" << origin << "' names no node of the graph";
    }
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->first)), 1u);
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->second)), 1u);
    EXPECT_EQ(origins.count(std::string(K_TRAILING_ORIGIN)), 0u);
}

/** @brief Tests that two steps of equal node indices keep apart by label and graph id. */
TEST_F(Medium_CompiledStepAttribution, TwoStepsHaveTwoLabels)
{
    const auto collector = std::make_shared<RunCollector>();
    const SushiRuntime::RunObserverHandle handle =
        engine->runtime().add_run_observer(collector);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> first,
                            make_chain(*engine, "first.step", false));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> second,
                            make_chain(*engine, "second.step", false));

    SA_EXPECT_OK(first->step->run());
    flush();
    SA_EXPECT_OK(second->step->run());
    flush();

    ASSERT_FALSE(collector->lost());
    const std::vector<SeenRun> runs = collector->runs();
    ASSERT_EQ(runs.size(), 2u);
    ASSERT_NE(runs[0].table, nullptr);
    ASSERT_NE(runs[1].table, nullptr);
    EXPECT_EQ(runs[0].table->label(), "first.step");
    EXPECT_EQ(runs[1].table->label(), "second.step");
    EXPECT_NE(runs[0].graph_id, 0u);
    EXPECT_NE(runs[0].graph_id, runs[1].graph_id);
    EXPECT_EQ(origins_of(*runs[0].table), origins_of(*runs[1].table));
}

/** @brief Tests that the work a trailing recorder adds carries the origin step.trailing. */
TEST_F(Medium_CompiledStepAttribution, TheOptimizerCarriesTheTrailingOrigin)
{
    const auto collector = std::make_shared<RunCollector>();
    const SushiRuntime::RunObserverHandle handle =
        engine->runtime().add_run_observer(collector);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(*engine, "chain.step", true));

    SA_EXPECT_OK(chain->step->run());
    flush();

    ASSERT_FALSE(collector->lost());
    const std::vector<SeenRun> runs = collector->runs();
    ASSERT_EQ(runs.size(), 1u);
    ASSERT_NE(runs[0].table, nullptr);
    const std::multiset<std::string> origins = origins_of(*runs[0].table);
    EXPECT_EQ(origins.count(std::string(K_TRAILING_ORIGIN)), 1u);
    EXPECT_EQ(origins.count(std::string()), 0u);
}

/** @brief Tests that the node behind a view records nothing and leaves no origin to the next. */
TEST_F(Medium_CompiledStepAttribution, AViewLeavesNoOriginOpen)
{
    const auto collector = std::make_shared<RunCollector>();
    const SushiRuntime::RunObserverHandle handle =
        engine->runtime().add_run_observer(collector);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Chain> chain,
                            make_chain(*engine, "chain.step", false));

    SA_EXPECT_OK(chain->step->run());
    flush();

    ASSERT_FALSE(collector->lost());
    const std::vector<SeenRun> runs = collector->runs();
    ASSERT_EQ(runs.size(), 1u);
    ASSERT_NE(runs[0].table, nullptr);
    const std::multiset<std::string> origins = origins_of(*runs[0].table);
    EXPECT_EQ(origins.count(origin_of_producer(chain->graph, chain->view)), 0u);
    EXPECT_EQ(origins.count(std::string()), 0u);
    EXPECT_GE(origins.count(origin_of_producer(chain->graph, chain->second)), 1u);
}
