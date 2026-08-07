/**************************************************************************/
/* test_objective.cpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: declaring leaves and emitting a loss are IR-to-IR, so the whole
// objective seam is testable without an Engine. Only score() needs a tensor,
// and that is covered by the integration tier through the trainer.
//
// The load-bearing test here is BuildGraphAndBuildClassifierProduceTheSameIr.
// The objective seam's whole promise is that it changed no numbers, and the
// way to check that is not "the loss still falls" - it is that the two entry
// points emit the same graph, node for node, in the same order. A graph
// comparison fails on a reordering that a loss comparison would absorb.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using Train::CrossEntropyObjective;
using Train::ObjectiveLeaves;

namespace
{
    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return Mlp(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    /** @brief Renders a graph as one line per node: op, inputs, outputs. */
    std::vector<std::string> describe(const Graph::Graph& graph)
    {
        std::vector<std::string> rows;
        rows.reserve(graph.node_count());

        for (std::size_t i = 0; i < graph.node_count(); ++i)
        {
            const Graph::Node& node = graph.node(static_cast<Graph::NodeId>(i));

            std::string row = std::to_string(node.op.value);
            row += " <-";
            for (Graph::ValueId in : node.inputs) row += " " + std::to_string(in);
            row += " ->";
            for (Graph::ValueId out : node.outputs) row += " " + std::to_string(out);
            rows.push_back(row);
        }
        return rows;
    }
}

TEST(Unit_Objective, BuildGraphAndBuildClassifierProduceTheSameIr)
{
    const int64_t batch = 4;
    const int64_t features = 6;
    const int64_t classes = 3;

    Mlp through_wrapper = make_model(features, 8, classes);
    Train::TrainingGraph a = Train::build_classifier(through_wrapper, batch, features, classes);

    Mlp through_seam = make_model(features, 8, classes);
    const CrossEntropyObjective objective(batch, classes);
    Train::TrainingGraph b = Train::build_graph(through_seam, objective, batch, features);

    EXPECT_EQ(a.graph.value_count(), b.graph.value_count());
    EXPECT_EQ(a.graph.node_count(), b.graph.node_count());
    EXPECT_EQ(describe(a.graph), describe(b.graph));

    // The value ids the loop reads are positions, so they must agree too.
    EXPECT_EQ(a.spec.inputs, b.spec.inputs);
    EXPECT_EQ(a.spec.targets, b.spec.targets);
    EXPECT_EQ(a.spec.logits, b.spec.logits);
    EXPECT_EQ(a.spec.loss, b.spec.loss);
}

TEST(Unit_Objective, TheTargetLeafIsDeclaredWhereItAlwaysWas)
{
    // Between the input leaf and the model's own trace. Value ids are
    // positions, so declaring it anywhere else renumbers the whole graph -
    // and every regression test pinning a node count or a byte-for-byte
    // result would then compare two different graphs while appearing to
    // compare one.
    Mlp model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 3);
    Train::TrainingGraph built = Train::build_graph(model, objective, 4, 6);

    EXPECT_EQ(built.spec.inputs, 0);
    EXPECT_EQ(built.spec.targets, 1);
    EXPECT_TRUE(built.leaves.has("targets"));
    EXPECT_EQ(built.leaves.at("targets"), built.spec.targets);
}

TEST(Unit_Objective, TheObjectiveOwnsTheOutputShapeCheck)
{
    // What shape an output must be is the objective's question - a detector's
    // is not a classifier's - so the message comes from the objective and not
    // from the graph builder.
    Mlp model = make_model(6, 8, 3);
    const CrossEntropyObjective objective(4, 5);   // asks for 5 classes, model gives 3

    SA_EXPECT_ERROR((void)Train::build_graph(model, objective, 4, 6),
                    "[4, 3]", "[4, 5]");
}

TEST(Unit_Objective, LeavesRefuseADuplicateRoleAndNameAnAbsentOne)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    ObjectiveLeaves leaves;
    leaves.add("targets", builder.input(Shape{2, 3}, "targets"));

    SA_EXPECT_ERROR(leaves.add("targets", builder.input(Shape{2, 3}, "again")),
                    "'targets'", "twice");
    SA_EXPECT_ERROR((void)leaves.at("collocation"), "'collocation'", "'targets'");

    EXPECT_TRUE(leaves.has("targets"));
    EXPECT_FALSE(leaves.has("collocation"));
    EXPECT_EQ(leaves.all().size(), 1u);
}

TEST(Unit_Objective, AnEmptyLeafSetSaysSoRatherThanListingNothing)
{
    const ObjectiveLeaves leaves;
    SA_EXPECT_ERROR((void)leaves.at("targets"), "'targets'", "(none)");
}

TEST(Unit_Objective, TheRegistryBuildsCrossEntropyAndListsWhatItHas)
{
    const Config::Value empty = Config::parse_json("{}", "<test>");

    const std::unique_ptr<Train::IObjective> objective =
        Train::make_objective("cross_entropy", 4, 3, empty, "objective");
    EXPECT_EQ(objective->name(), "cross_entropy");

    SA_EXPECT_ERROR((void)Train::make_objective("yolo", 4, 3, empty, "objective"),
                    "yolo", "objective type", "'cross_entropy'");
}

TEST(Unit_Objective, CrossEntropyTakesNoKeysAtAll)
{
    // It has no hyperparameters, so a key written onto it is an error rather
    // than a line whose author believed it did something.
    const Config::Value stray = Config::parse_json(R"({"label_smoothing": 0.1})", "<test>");
    SA_EXPECT_ERROR((void)Train::make_objective("cross_entropy", 4, 3, stray, "objective"),
                    "objective", "'label_smoothing'");
}
