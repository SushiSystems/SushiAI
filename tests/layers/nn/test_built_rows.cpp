/**************************************************************************/
/* test_built_rows.cpp                                                    */
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
 * @file test_built_rows.cpp
 * @brief Tests BuiltRows, read_row_above, find_embedding and the linked rows of a registry.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/layer_rows/layer_registry.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_attributes.hpp>
#include <SushiAI/nn/node_registry.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST body, where Test names the fixture base.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Names the key the test row states its row under. */
    constexpr std::string_view K_NAMES_KEY = "names";

    /** @brief Parses the attributes of one row. */
    Config::Value attributes_of(std::string_view text)
    {
        return SaTest::require_value(Config::parse_json(text, "<test>"));
    }

    /** @brief Parses an embedding, a layer norm and a follower row whose "names" is @p names. */
    Config::ModelSpec follower_model(int names)
    {
        return SaTest::model_from(
            R"({
        "format_version": 2,
        "input": {"shape": [4], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 10, "embedding_dim": 3},
            {"type": "layernorm"},
            {"type": "follower", "names": )" +
            std::to_string(names) + R"(}
        ]
    })");
    }

    /** @brief Builds an Identity node, and refuses unless "names" is an embedding row above. */
    Result<std::unique_ptr<NN::Node>> make_follower_node(NN::ShapeSpan inputs,
                                                         const Config::Value& attributes,
                                                         std::string_view context,
                                                         const NN::BuiltRows& above)
    {
        SA_TRY_ASSIGN(const std::size_t row,
                      NN::Detail::read_row_above(attributes, K_NAMES_KEY, context,
                                                 above.size()));
        SA_FAIL_IF(NN::find_embedding(above.node(row)) == nullptr,
                   StatusCode::InvalidDescription, "{} names row {}, which is no embedding",
                   context, row);

        return NN::Detail::wrap_block(NN::Identity{}, inputs[0]);
    }

    /** @brief Returns the layer rows of nn and the linked row "follower". */
    NN::NodeRegistry make_registry()
    {
        NN::NodeRegistry registry = SaTest::make_layer_registry();
        SA_EXPECT_OK(registry.add(NN::LinkedNodeEntry{"follower", &make_follower_node}));
        return registry;
    }

    /** @brief Asks an empty view for its first row. */
    void node_of_no_row()
    {
        const NN::BuiltRows none;
        static_cast<void>(none.node(0));
    }
} // namespace

/** @brief Tests that a reference is a row number, -1 for the row before, or an object. */
TEST(Small_LinkedRows, ARowReferenceTakesThreeForms)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t absolute,
                            NN::Detail::read_row_above(attributes_of(R"({"names": 0})"),
                                                       K_NAMES_KEY, "layer 3", 3));
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t before,
                            NN::Detail::read_row_above(attributes_of(R"({"names": -1})"),
                                                       K_NAMES_KEY, "layer 3", 3));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::size_t object,
        NN::Detail::read_row_above(attributes_of(R"({"names": {"node": 1}})"), K_NAMES_KEY,
                                   "layer 3", 3));

    EXPECT_EQ(absolute, 0u);
    EXPECT_EQ(before, 2u);
    EXPECT_EQ(object, 1u);
}

/** @brief Holds an attribute row, the rows above it and the substrings of the refusal. */
struct RowReferenceRow
{
    std::string name;     /**< The row's name. */
    std::string json;     /**< The attribute row, as JSON. */
    std::string context;  /**< The context the refusal names. */
    std::size_t above;    /**< The number of rows above. */
    std::string first;    /**< A substring the message holds. */
    std::string second;   /**< A second substring the message holds; empty for none. */
    std::string third;    /**< A third substring the message holds; empty for none. */
    std::string fourth;   /**< A fourth substring the message holds; empty for none. */
};

/** @brief Runs the row-reference refusal over each reference that does not fit. */
class Small_LinkedRowsRefusal : public ::testing::TestWithParam<RowReferenceRow>
{
};

/** @brief Tests that a row that is not above, and every other form, is refused by name. */
TEST_P(Small_LinkedRowsRefusal, ARowReferenceThatDoesNotFitIsRefused)
{
    const RowReferenceRow& row = GetParam();

    SA_EXPECT_FAILURE(
        NN::Detail::read_row_above(attributes_of(row.json), K_NAMES_KEY, row.context, row.above),
        StatusCode::InvalidDescription, row.first, row.second, row.third, row.fourth);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LinkedRowsRefusal,
    ::testing::Values(
        RowReferenceRow{"ARowNotAbove", R"({"names": 3})", "layer 3", 3, "layer 3", "'names'",
                        "row 3", "not above"},
        RowReferenceRow{"ANegativeRowBeyondTheTop", R"({"names": -2})", "layer 3", 3, "row -2",
                        "", "", ""},
        RowReferenceRow{"TheRowBeforeWithNoRowAbove", R"({"names": -1})", "layer 0", 0,
                        "layer 0", "0 rows are above", "", ""},
        RowReferenceRow{"AString", R"({"names": "input"})", "layer 3", 3, "row number", "string",
                        "", ""},
        RowReferenceRow{"AnObjectWithAPort", R"({"names": {"node": 0, "port": "x"}})", "layer 3",
                        3, "'port'", "", "", ""},
        RowReferenceRow{"AnObjectWithNoNode", R"({"names": {}})", "layer 3", 3, "'node'", "", "",
                        ""},
        RowReferenceRow{"AnAbsentKey", "{}", "layer 3", 3, "'names'", "absent", "", ""}),
    [](const ::testing::TestParamInfo<RowReferenceRow>& info) { return info.param.name; });

/** @brief Tests that a linked type is found and listed, and is no node type. */
TEST(Small_LinkedRows, ALinkedTypeIsFoundAndListed)
{
    NN::NodeRegistry registry = make_registry();

    ASSERT_NE(registry.find_linked("follower"), nullptr);
    EXPECT_EQ(registry.find_node("follower"), nullptr);
    ASSERT_EQ(registry.linked().size(), 1u);
    EXPECT_EQ(registry.linked()[0].type, "follower");
}

/** @brief Holds a type name in use, the kind of entry that claims it again, and the name. */
struct TypeInUseRow
{
    std::string name;   /**< The row's name. */
    std::string type;   /**< The type name that is already in use. */
    bool linked;        /**< Whether the second entry is a linked one; a node entry otherwise. */
    std::string quoted; /**< The quoted name the message holds. */
};

/** @brief Runs the type-in-use refusal over each kind of entry. */
class Small_LinkedRowsTypeInUse : public ::testing::TestWithParam<TypeInUseRow>
{
};

/** @brief Tests that a type name already in use is refused whatever the kind of the entry. */
TEST_P(Small_LinkedRowsTypeInUse, ATypeNameInUseIsRefused)
{
    const TypeInUseRow& row = GetParam();
    NN::NodeRegistry registry = make_registry();

    const Status refused =
        row.linked ? registry.add(NN::LinkedNodeEntry{row.type, &make_follower_node})
                   : registry.add(NN::NodeEntry{row.type, &NN::make_embedding_node});

    SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription, row.quoted, "already");
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_LinkedRowsTypeInUse,
    ::testing::Values(TypeInUseRow{"ANodeTypeAsALinkedType", "concat", true, "'concat'"},
                      TypeInUseRow{"AnotherNodeTypeAsALinkedType", "relu", true, "'relu'"},
                      TypeInUseRow{"ALinkedTypeAsANodeType", "follower", false, "'follower'"}),
    [](const ::testing::TestParamInfo<TypeInUseRow>& info) { return info.param.name; });

/** @brief Tests that build_model hands a linked row the nodes of the rows above it. */
TEST(Small_LinkedRows, BuildModelHandsALinkedRowTheRowsAbove)
{
    const NN::NodeRegistry registry = make_registry();

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph model,
                            NN::build_model(registry, follower_model(0)));

    EXPECT_EQ(model.size(), 3u);
    EXPECT_NE(NN::find_embedding(model.node(0)), nullptr);
    EXPECT_EQ(NN::find_embedding(model.node(1)), nullptr);
}

/** @brief Tests that build_model refuses a linked row that names a row with no embedding. */
TEST(Small_LinkedRows, BuildModelRefusesALinkedRowThatNamesANonEmbedding)
{
    const NN::NodeRegistry registry = make_registry();

    SA_EXPECT_FAILURE(NN::build_model(registry, follower_model(-1)),
                      StatusCode::InvalidDescription, "layer 2 'follower'", "row 1",
                      "no embedding");
}

/** @brief Tests that a linked row built alone sees no row above it. */
TEST(Small_LinkedRows, ALinkedRowBuiltAloneSeesNoRow)
{
    const NN::NodeRegistry registry = make_registry();
    const Shape sample{4, 3};

    SA_EXPECT_FAILURE(NN::make_node(registry, "follower", NN::ShapeSpan(&sample, 1),
                                    attributes_of(R"({"names": 0})"), "layer 0 'follower'"),
                      StatusCode::InvalidDescription, "layer 0 'follower'", "not above");
}

/** @brief Tests that an unknown type is refused and the message lists the linked type. */
TEST(Small_LinkedRows, AnUnknownTypeListsTheLinkedType)
{
    const NN::NodeRegistry registry = make_registry();
    const Shape sample{4, 3};

    SA_EXPECT_FAILURE(NN::make_node(registry, "missing", NN::ShapeSpan(&sample, 1),
                                    attributes_of("{}"), "layer 0 'missing'"),
                      StatusCode::InvalidDescription, "'missing'", "'follower'");
}

/** @brief Tests that the view counts the nodes of its graph and that no view counts none. */
TEST(Small_LinkedRows, BuiltRowsCountsTheNodesOfItsGraph)
{
    NN::ModelGraph model(Shape{4});
    NN::Embedding table{};
    table.num_embeddings = 10;
    table.embedding_dim = 3;
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef first,
                            model.add("0", std::move(table), model.input()));
    static_cast<void>(first);

    const NN::BuiltRows above(model);
    const NN::BuiltRows none;

    EXPECT_EQ(above.size(), 1u);
    EXPECT_EQ(&above.node(0), &model.node(0));
    EXPECT_EQ(none.size(), 0u);
}

/** @brief Tests that asking a view for a row it does not hold stops the process. */
TEST(Medium_LinkedRowsDeathTest, ARowOutsideTheViewIsFatal)
{
    SA_EXPECT_FATAL(node_of_no_row(), "BuiltRows::node", "row 0");
}
