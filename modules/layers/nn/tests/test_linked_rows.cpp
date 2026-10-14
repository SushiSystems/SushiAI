/**************************************************************************/
/* test_linked_rows.cpp                                                   */
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
 * @file test_linked_rows.cpp
 * @brief Tests BuiltRows, read_row_above, find_embedding and the linked rows of a registry.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_attributes.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/layer_rows/layer_registry.hpp>

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
        return Config::parse_json(text, "<test>");
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
TEST(Unit_LinkedRows, ARowReferenceTakesThreeForms)
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

/** @brief Tests that a row that is not above, and every other form, is refused by name. */
TEST(Unit_LinkedRows, ARowReferenceThatDoesNotFitIsRefused)
{
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of(R"({"names": 3})"), K_NAMES_KEY,
                                                 "layer 3", 3),
                      StatusCode::InvalidDescription, "layer 3", "'names'", "row 3",
                      "not above");
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of(R"({"names": -2})"), K_NAMES_KEY,
                                                 "layer 3", 3),
                      StatusCode::InvalidDescription, "row -2");
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of(R"({"names": -1})"), K_NAMES_KEY,
                                                 "layer 0", 0),
                      StatusCode::InvalidDescription, "layer 0", "0 rows are above");
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of(R"({"names": "input"})"),
                                                 K_NAMES_KEY, "layer 3", 3),
                      StatusCode::InvalidDescription, "row number", "string");
    SA_EXPECT_FAILURE(
        NN::Detail::read_row_above(attributes_of(R"({"names": {"node": 0, "port": "x"}})"),
                                   K_NAMES_KEY, "layer 3", 3),
        StatusCode::InvalidDescription, "'port'");
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of(R"({"names": {}})"), K_NAMES_KEY,
                                                 "layer 3", 3),
                      StatusCode::InvalidDescription, "'node'");
    SA_EXPECT_FAILURE(NN::Detail::read_row_above(attributes_of("{}"), K_NAMES_KEY, "layer 3", 3),
                      StatusCode::InvalidDescription, "'names'", "absent");
}

/** @brief Tests that a linked type is found, listed, and shares one namespace of types. */
TEST(Unit_LinkedRows, ALinkedTypeHasOneRowAmongAllKinds)
{
    NN::NodeRegistry registry = make_registry();

    ASSERT_NE(registry.find_linked("follower"), nullptr);
    EXPECT_EQ(registry.find_node("follower"), nullptr);
    ASSERT_EQ(registry.linked().size(), 1u);
    EXPECT_EQ(registry.linked()[0].type, "follower");

    SA_EXPECT_FAILURE(registry.add(NN::LinkedNodeEntry{"concat", &make_follower_node}),
                      StatusCode::InvalidDescription, "'concat'", "already");
    SA_EXPECT_FAILURE(registry.add(NN::LinkedNodeEntry{"relu", &make_follower_node}),
                      StatusCode::InvalidDescription, "'relu'", "already");
    SA_EXPECT_FAILURE(registry.add(NN::NodeEntry{"follower", &NN::make_embedding_node}),
                      StatusCode::InvalidDescription, "'follower'", "already");
}

/** @brief Tests that build_model hands a linked row the nodes of the rows above it. */
TEST(Unit_LinkedRows, BuildModelHandsALinkedRowTheRowsAbove)
{
    const NN::NodeRegistry registry = make_registry();
    const Config::ModelSpec tied = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [4], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 10, "embedding_dim": 3},
            {"type": "layernorm"},
            {"type": "follower", "names": 0}
        ]
    })");
    const Config::ModelSpec misnamed = SaTest::model_from(R"({
        "format_version": 2,
        "input": {"shape": [4], "dtype": "int32"},
        "layers": [
            {"type": "embedding", "num_embeddings": 10, "embedding_dim": 3},
            {"type": "layernorm"},
            {"type": "follower", "names": -1}
        ]
    })");

    SA_ASSERT_OK_AND_ASSIGN(const NN::ModelGraph model, NN::build_model(registry, tied));
    EXPECT_EQ(model.size(), 3u);
    EXPECT_NE(NN::find_embedding(model.node(0)), nullptr);
    EXPECT_EQ(NN::find_embedding(model.node(1)), nullptr);

    SA_EXPECT_FAILURE(NN::build_model(registry, misnamed), StatusCode::InvalidDescription,
                      "layer 2 'follower'", "row 1", "no embedding");
}

/** @brief Tests that a linked row built alone sees no row, and that an unknown type lists it. */
TEST(Unit_LinkedRows, ALinkedRowBuiltAloneSeesNoRow)
{
    const NN::NodeRegistry registry = make_registry();
    const Shape sample{4, 3};

    SA_EXPECT_FAILURE(NN::make_node(registry, "follower", NN::ShapeSpan(&sample, 1),
                                    attributes_of(R"({"names": 0})"), "layer 0 'follower'"),
                      StatusCode::InvalidDescription, "layer 0 'follower'", "not above");
    SA_EXPECT_FAILURE(NN::make_node(registry, "missing", NN::ShapeSpan(&sample, 1),
                                    attributes_of("{}"), "layer 0 'missing'"),
                      StatusCode::InvalidDescription, "'missing'", "'follower'");
}

/** @brief Tests that the view counts the nodes of its graph and that no view counts none. */
TEST(Unit_LinkedRows, BuiltRowsCountsTheNodesOfItsGraph)
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
TEST(Unit_LinkedRowsDeathTest, ARowOutsideTheViewIsFatal)
{
    SA_EXPECT_FATAL(node_of_no_row(), "BuiltRows::node", "row 0");
}
