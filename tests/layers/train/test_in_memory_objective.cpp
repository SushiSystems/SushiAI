/**************************************************************************/
/* test_in_memory_objective.cpp                                           */
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
 * @file test_in_memory_objective.cpp
 * @brief Tests that InMemoryObjective keeps the promise of the Objective interface.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/in_memory_objective.hpp>
#include <SushiAI/train/objective.hpp>

using namespace SushiAI;
using Train::InMemoryObjective;

/** @brief Tests that declare adds the scripted leaves and roles and counts the call. */
TEST(Small_InMemoryObjective, DeclareAddsTheScriptedLeavesAndRoles)
{
    InMemoryObjective objective;
    objective.script_leaf("targets", 4);
    objective.script_required(Data::TargetDescription{"labels", Shape{1}});
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Train::ObjectiveLeaves leaves;
    const Train::Objective& port = objective;

    port.declare(builder, leaves);

    EXPECT_TRUE(leaves.has("targets"));
    EXPECT_EQ(leaves.at("targets"), 4u);
    ASSERT_EQ(leaves.required().size(), 1u);
    EXPECT_EQ(leaves.required()[0].role, "labels");
    EXPECT_EQ(objective.declare_count(), 1u);
}

/** @brief Tests that loss returns the scripted value and counts the call. */
TEST(Small_InMemoryObjective, LossReturnsTheScriptedValue)
{
    InMemoryObjective objective;
    objective.script_loss(9);
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID loss,
                            objective.loss(builder, graph, outputs, leaves));

    EXPECT_EQ(loss, 9u);
    EXPECT_EQ(objective.loss_count(), 1u);
}

/** @brief Tests that a loss nobody scripted is refused as InvalidDescription. */
TEST(Small_InMemoryObjective, AnUnscriptedLossIsRefused)
{
    const InMemoryObjective objective;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "no loss value");
}

/** @brief Tests that a scripted failure is returned in place of the loss value. */
TEST(Small_InMemoryObjective, AScriptedFailureReplacesTheLoss)
{
    InMemoryObjective objective;
    objective.script_loss(9);
    objective.script_loss_failure(Status::from_message(StatusCode::InvalidDescription, []()
    {
        return std::string("the output 'logits' is unread");
    }));
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const NN::ModelOutputs outputs;
    const Train::ObjectiveLeaves leaves;

    SA_EXPECT_FAILURE(objective.loss(builder, graph, outputs, leaves),
                      StatusCode::InvalidDescription, "logits");
}

/** @brief Tests that the objective names itself and takes no batch to feed it. */
TEST(Small_InMemoryObjective, NamesItselfAndIsFedByNoBatch)
{
    const InMemoryObjective objective;

    EXPECT_EQ(objective.name(), "in-memory");
    EXPECT_FALSE(objective.is_fed_by_batch());
}
