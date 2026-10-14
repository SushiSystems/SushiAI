/**************************************************************************/
/* test_model_state.cpp                                                   */
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
 * @file test_model_state.cpp
 * @brief Tests ModelState's listing, lookup, refusals and attachment to pools.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_ModelState : public SaTest::EngineTest {};

    using Integration_ModelStateDeathTest = Integration_ModelState;

    /** @brief The absolute tolerance of a value that is stored and read back unchanged. */
    constexpr double K_STORED_TOLERANCE = 0.0;

    /** @brief The architecture every case that needs a parameter model traces. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP make_mlp()
    {
        return MLP(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    /** @brief Traces @p model into @p graph from a [4, 6] input. */
    void trace_mlp(MLP& model, Graph::Graph& graph)
    {
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");
        static_cast<void>(model.forward(builder, x));
    }

    /** @brief Builds an entry of a given name and identity, for the from_entries cases. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = Shape{2};
        return entry;
    }
} // namespace

/** @brief Tests that a traced Sequential lists its parameters in walk order with their shapes. */
TEST_F(Integration_ModelState, CreateListsParametersInWalkOrder)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    const auto entries = state.entries();
    ASSERT_EQ(entries.size(), 4u);
    EXPECT_EQ(entries[0].name, "0.weight");
    EXPECT_EQ(entries[0].shape, (Shape{8, 6}));
    EXPECT_EQ(entries[1].name, "0.bias");
    EXPECT_EQ(entries[1].shape, (Shape{8}));
    EXPECT_EQ(entries[2].name, "2.weight");
    EXPECT_EQ(entries[2].shape, (Shape{4, 8}));
    EXPECT_EQ(entries[3].name, "2.bias");
    EXPECT_EQ(entries[3].shape, (Shape{4}));
    for (const NN::StateEntry& entry : entries)
        EXPECT_EQ(entry.kind, NN::StateKind::Parameter);
}

/**
 * @brief Tests that a training trace of a BatchNorm2d lists its two parameters and its two
 *        buffers under the row's name, with the buffers' constant starting values.
 */
TEST_F(Integration_ModelState, CreateListsBuffersAfterATrainingTrace)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    SA_ASSERT_OK_AND_ASSIGN(const auto norm,
                            model.add("bn", NN::BatchNorm2d(3, NN::NormOptions{}), model.input()));
    SA_EXPECT_OK(model.output("normalised", norm));

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4, 4, 3}, "x");
    NN::BufferUpdateList list;
    SA_EXPECT_OK(model.trace(NN::Trace::training(builder, list), x));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    const auto entries = state.entries();
    ASSERT_EQ(entries.size(), 4u);
    EXPECT_EQ(entries[0].name, "bn.weight");
    EXPECT_EQ(entries[0].kind, NN::StateKind::Parameter);
    EXPECT_EQ(entries[1].name, "bn.bias");
    EXPECT_EQ(entries[1].kind, NN::StateKind::Parameter);
    EXPECT_EQ(entries[2].name, "bn.running_mean");
    EXPECT_EQ(entries[2].kind, NN::StateKind::Buffer);
    EXPECT_EQ(entries[2].init.kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(entries[2].init.constant, 0.0, K_STORED_TOLERANCE);
    EXPECT_EQ(entries[3].name, "bn.running_var");
    EXPECT_EQ(entries[3].kind, NN::StateKind::Buffer);
    EXPECT_EQ(entries[3].init.kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(entries[3].init.constant, 1.0, K_STORED_TOLERANCE);
}

/** @brief Tests that a model never traced gives a valid state with no entries. */
TEST_F(Integration_ModelState, AnUntracedModelGivesAnEmptyState)
{
    MLP model = make_mlp();

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    EXPECT_TRUE(state.entries().empty());
}

/** @brief Tests that a prefix given to create() is joined to every entry name. */
TEST_F(Integration_ModelState, APrefixIsJoinedToEveryName)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::create(*engine, model, "model"));
    const auto entries = state.entries();
    ASSERT_FALSE(entries.empty());
    EXPECT_EQ(entries[0].name, "model.0.weight");
}

/** @brief Tests that two entries sharing a name are refused and the name is reported. */
TEST_F(Integration_ModelState, TwoEntriesWithOneNameAreRefused)
{
    std::vector<NN::StateEntry> entries{make_entry("shared", 1), make_entry("shared", 2)};

    const auto result = NN::ModelState::from_entries(*engine, std::move(entries));
    SA_EXPECT_FAILURE(result, StatusCode::InvalidDescription, "shared");
}

/** @brief Tests that an entry with no identity stops the process and the report names it. */
TEST_F(Integration_ModelStateDeathTest, AnEntryWithNoIdentityIsFatal)
{
    std::vector<NN::StateEntry> entries{make_entry("orphan", 0)};

    SA_EXPECT_FATAL((void)NN::ModelState::from_entries(*engine, std::move(entries)), "'orphan'",
                    "no identity");
}

/** @brief Tests that the name form and the identity form return the same tensor. */
TEST_F(Integration_ModelState, TensorByNameAndByStateAreTheSameStorage)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    const Graph::StateID id = state.entries()[0].state;
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* by_name, state.tensor("0.weight"));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* by_state, state.tensor(id));
    EXPECT_EQ(by_name, by_state);
}

/** @brief Tests that an unknown name is refused, and the refusal names it and the entry count. */
TEST_F(Integration_ModelState, AnUnknownNameIsRefusedAndNamed)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    SA_EXPECT_FAILURE(state.tensor("missing.weight"), StatusCode::InvalidDescription,
                      "missing.weight", "4 entries");
}

/** @brief Tests that attach() binds each state leaf of the graph to the state's own storage. */
TEST_F(Integration_ModelState, AttachBindsEveryStateLeafToTheStatesStorage)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool pool(*engine);
    SA_EXPECT_OK(state.attach(graph, pool));
    ASSERT_EQ(graph.state_leaves().size(), 4u);
    for (const std::pair<Graph::StateID, Graph::ValueID>& leaf : graph.state_leaves())
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* held, state.tensor(leaf.first));
        EXPECT_EQ(pool.tensor(leaf.second).data(), held->data());
    }
}

/** @brief Tests that attach() refuses a graph traced from a different model. */
TEST_F(Integration_ModelState, AttachRefusesAGraphOfAnotherModel)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    MLP other = make_mlp();
    Graph::Graph other_graph;
    trace_mlp(other, other_graph);

    Graph::TensorPool pool(*engine);
    SA_EXPECT_FAILURE(state.attach(other_graph, pool), StatusCode::Incompatible);
}

/** @brief Tests that two pools attached to one state read and write the same numbers. */
TEST_F(Integration_ModelState, TwoPoolsAttachedToOneStateShareStorage)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    Graph::TensorPool first(*engine);
    Graph::TensorPool second(*engine);
    SA_EXPECT_OK(state.attach(graph, first));
    SA_EXPECT_OK(state.attach(graph, second));

    const Graph::ValueID leaf = graph.state_leaves().front().second;
    first.tensor(leaf).data_as<float>()[0] = 42.5f;
    EXPECT_NEAR(second.tensor(leaf).data_as<float>()[0], 42.5, K_STORED_TOLERANCE);
}
