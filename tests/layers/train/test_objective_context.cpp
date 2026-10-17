/**************************************************************************/
/* test_objective_context.cpp                                             */
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
 * @file test_objective_context.cpp
 * @brief Tests the listing of the extents a traced model published.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/steps.hpp>

using namespace SushiAI;
using Train::OutputShape;

/** @brief Tests that each output is listed under its name with its extents, in order. */
TEST(Medium_ObjectiveContext, ListsEveryOutputWithItsExtentsInPublishedOrder)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    NN::ModelOutputs outputs;
    SA_EXPECT_OK(outputs.add("coarse", builder.input(Shape{2, 4, 4, 14}, "coarse")));
    SA_EXPECT_OK(outputs.add("fine", builder.input(Shape{2, 8, 8, 14}, "fine")));

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<OutputShape> listed,
                            Train::list_output_shapes(graph, outputs));

    ASSERT_EQ(listed.size(), 2u);
    EXPECT_EQ(listed[0].name, "coarse");
    EXPECT_EQ(listed[0].shape, (Shape{2, 4, 4, 14}));
    EXPECT_EQ(listed[1].name, "fine");
    EXPECT_EQ(listed[1].shape, (Shape{2, 8, 8, 14}));
}

/** @brief Tests that a model with no output lists nothing. */
TEST(Medium_ObjectiveContext, AnEmptyOutputSetListsNothing)
{
    const Graph::Graph graph;
    const NN::ModelOutputs outputs;

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<OutputShape> listed,
                            Train::list_output_shapes(graph, outputs));

    EXPECT_TRUE(listed.empty());
}

/** @brief Tests that the probe trace of a model gives its head shapes with no engine. */
TEST(Medium_ObjectiveContext, AProbeTraceGivesTheShapesWithoutAnEngine)
{
    NN::ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef hidden,
                            model.add("hidden", NN::Linear{6, 5}, model.input()));
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef logits,
                            model.add("logits", NN::Linear{5, 3}, hidden));
    SA_EXPECT_OK(model.output("hidden", hidden));
    SA_EXPECT_OK(model.output("logits", logits));

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, Shape{4, 6}));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<OutputShape> listed,
                            Train::list_output_shapes(probe.graph, probe.outputs));

    ASSERT_EQ(listed.size(), 2u);
    EXPECT_EQ(listed[0].name, "hidden");
    EXPECT_EQ(listed[0].shape, (Shape{4, 5}));
    EXPECT_EQ(listed[1].name, "logits");
    EXPECT_EQ(listed[1].shape, (Shape{4, 3}));
}

/** @brief Tests that an output is found by its name and that an absent name gives null. */
TEST(Medium_ObjectiveContext, FindsAnOutputByItsName)
{
    const std::vector<OutputShape> outputs = {OutputShape{"coarse", Shape{2, 4, 4, 14}},
                                              OutputShape{"fine", Shape{2, 8, 8, 14}}};

    const OutputShape* const fine = Train::find_output_shape(outputs, "fine");
    ASSERT_NE(fine, nullptr);
    EXPECT_EQ(fine->shape, (Shape{2, 8, 8, 14}));
    EXPECT_EQ(Train::find_output_shape(outputs, "medium"), nullptr);
}
