/**************************************************************************/
/* test_module_list.cpp                                                   */
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
 * @file test_module_list.cpp
 * @brief Tests ModuleList: its storage, its index names and its place in the state walk.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>

using namespace SushiAI;
using NN::Linear;
using NN::ModuleList;

static_assert(NN::ModuleContainer<ModuleList<Linear>>,
              "ModuleList must be walkable as a container");
static_assert(!NN::Detail::HasStateWalk<ModuleList<Linear>>::value,
              "ModuleList must be enumerated by visit_children, not by a walk of its own");
static_assert(!NN::Detail::HasParameterWalk<ModuleList<Linear>>::value,
              "ModuleList must be enumerated by visit_children, not by walk_parameters");

namespace
{
    /** @brief Holds a list of layers under the name "blocks", then one layer after it. */
    struct Stage
    {
        ModuleList<Linear> blocks{};
        Linear head{4, 2};

        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("blocks", &Stage::blocks),
                NN::child("head", &Stage::head),
            };
        }
    };

    /** @brief Holds one buffer, so a list of it carries state that is not a parameter. */
    struct Counter
    {
        NN::Buffer count{};

        static constexpr auto fields()
        {
            return std::tuple{ NN::buffer("count", &Counter::count) };
        }
    };

    /** @brief Collects the walk of @p model as "name:extents" strings. */
    template<typename M>
    std::vector<std::string> parameter_names(M& model)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }

    /** @brief Reads the element one past the end of a two-element list, for a death test. */
    void read_past_the_end()
    {
        ModuleList<Linear> list;
        list.push_back(Linear{4, 4});
        list.push_back(Linear{4, 4});
        (void)list[2];
    }
}

/** @brief Tests that push_back() appends, so an index returns the child pushed at it. */
TEST(Small_ModuleList, PushBackAppendsInOrder)
{
    ModuleList<Linear> list;
    list.push_back(Linear{4, 8});
    list.push_back(Linear{8, 3});

    ASSERT_EQ(list.size(), 2u);
    EXPECT_EQ(list[0].out_features, 8);
    EXPECT_EQ(list[1].out_features, 3);

    const ModuleList<Linear>& read_only = list;
    EXPECT_EQ(read_only[1].in_features, 8);
}

namespace
{
    /** @brief Lists each child with its index, as the enumeration of a row hands them out. */
    using IndexedChildren = std::vector<std::pair<std::size_t, const Linear*>>;

    /** @brief Enumerates @p list through begin() and end(), counting the index. */
    IndexedChildren through_begin_and_end(ModuleList<Linear>& list)
    {
        IndexedChildren children;
        std::size_t index = 0;
        for (Linear& layer : list)
            children.emplace_back(index++, &layer);

        return children;
    }

    /** @brief Enumerates @p list through visit_children(). */
    IndexedChildren through_visit_children(ModuleList<Linear>& list)
    {
        IndexedChildren children;
        list.visit_children([&children](std::size_t index, Linear& child)
        {
            children.emplace_back(index, &child);
        });

        return children;
    }

    /** @brief Names a way to enumerate the children of a list. */
    struct EnumerationRow
    {
        std::string name;                                  /**< The row's name. */
        IndexedChildren (*enumerate)(ModuleList<Linear>&); /**< Hands out the children. */
    };

    /** @brief Appends @p count layers of 4 to 4 to @p list. */
    void append_linears(ModuleList<Linear>& list, std::size_t count)
    {
        for (std::size_t appended = 0; appended < count; ++appended)
            list.push_back(Linear{4, 4});
    }
}

/** @brief Runs the enumeration case over begin() and end() and over visit_children(). */
class Small_ModuleListEnumeration : public ::testing::TestWithParam<EnumerationRow>
{
};

/** @brief Tests that the enumeration of the row hands out each child with its index, in order. */
TEST_P(Small_ModuleListEnumeration, ChildrenComeInIndexOrder)
{
    ModuleList<Linear> list;
    list.push_back(Linear{4, 8});
    list.push_back(Linear{8, 3});
    list.push_back(Linear{3, 1});

    const IndexedChildren children = GetParam().enumerate(list);

    EXPECT_EQ(children, (IndexedChildren{{0, &list[0]}, {1, &list[1]}, {2, &list[2]}}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ModuleListEnumeration,
    ::testing::Values(EnumerationRow{"ThroughBeginAndEnd", through_begin_and_end},
                      EnumerationRow{"ThroughVisitChildren", through_visit_children}),
    [](const ::testing::TestParamInfo<EnumerationRow>& info) { return info.param.name; });

/** @brief Tests that the walk names the state of a list's children by index. */
TEST(Small_ModuleList, TheWalkNamesChildrenByIndex)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    ModuleList<Linear> list;
    list.push_back(Linear{4, 8});
    list.push_back(Linear{8, 3, false});

    Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    for (Linear& layer : list)
    {
        SA_ASSERT_OK_AND_ASSIGN(x, layer.forward(builder, x));
    }

    EXPECT_EQ(parameter_names(list), (std::vector<std::string>{
        "0.weight:[8, 4]", "0.bias:[8]",
        "1.weight:[3, 8]",
    }));
    EXPECT_EQ(NN::parameter_count(list), 3u);
}

/** @brief Tests that a list listed as a child field is named by the field, then by index. */
TEST(Small_ModuleList, AListUnderAChildFieldIsNamedByFieldThenIndex)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Stage stage;
    stage.blocks.push_back(Linear{4, 4});
    stage.blocks.push_back(Linear{4, 4});

    Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    for (Linear& layer : stage.blocks)
    {
        SA_ASSERT_OK_AND_ASSIGN(x, layer.forward(builder, x));
    }
    SA_EXPECT_OK(stage.head.forward(builder, x));

    EXPECT_EQ(parameter_names(stage), (std::vector<std::string>{
        "blocks.0.weight:[4, 4]", "blocks.0.bias:[4]",
        "blocks.1.weight:[4, 4]", "blocks.1.bias:[4]",
        "head.weight:[2, 4]", "head.bias:[2]",
    }));
}

/** @brief Tests that the walk names the buffers of a list's children by index. */
TEST(Small_ModuleList, TheWalkNamesBuffersByIndex)
{
    ModuleList<Counter> list;
    list.push_back(Counter{});
    list.push_back(Counter{});
    list.push_back(Counter{});
    SA_EXPECT_OK(list[0].count.define(Shape{1}, 0.0));
    SA_EXPECT_OK(list[2].count.define(Shape{1}, 0.0));

    std::vector<std::string> names;
    NN::for_each_buffer(list, [&names](const std::string& name, NN::Buffer&)
    {
        names.push_back(name);
    });

    // Child 1 keeps its index although its buffer is undefined and so is not yielded.
    EXPECT_EQ(names, (std::vector<std::string>{"0.count", "2.count"}));
    EXPECT_EQ(NN::parameter_count(list), 0u);
}

/** @brief Tests that growing the list past its storage keeps a traced child's identity. */
TEST(Small_ModuleList, GrowingTheListKeepsATracedChildsStateIdentity)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    ModuleList<Linear> list;
    list.push_back(Linear{4, 4});
    SA_EXPECT_OK(list[0].forward(builder, builder.input(Shape{2, 4}, "x")));

    const Graph::StateID weight_state = list[0].weight.state();
    const Graph::ValueID weight_leaf = list[0].weight.id_in(graph);
    const Graph::ValueID bias_leaf = list[0].bias.id_in(graph);
    ASSERT_NE(weight_state, Graph::K_NO_STATE);

    // 64 appends to a one-element vector move its storage at least once.
    append_linears(list, 64);

    EXPECT_EQ(list[0].weight.state(), weight_state);
    EXPECT_EQ(list[0].weight.id_in(graph), weight_leaf);
    EXPECT_EQ(list[0].bias.id_in(graph), bias_leaf);
}

/** @brief Tests that an index past the end stops the process and names the index and the size. */
TEST(Medium_ModuleListDeathTest, AnIndexPastTheEndIsFatal)
{
    SA_EXPECT_FATAL(read_past_the_end(), "index 2", "holds 2 children");
}
