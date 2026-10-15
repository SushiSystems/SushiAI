/**************************************************************************/
/* test_xml_element.cpp                                                   */
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
 * @file test_xml_element.cpp
 * @brief Tests the parser of the XML subset: what it reads and the line of what it refuses.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/xml_element.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

/** @brief Tests that elements, their text, their order and their lines are read. */
TEST(Small_DatasetsXmlElement, ElementsTextAndLinesAreRead)
{
    SA_ASSERT_OK_AND_ASSIGN(const XMLElement root,
                            parse_xml("<a>\n  <b> one </b>\n  <c><d>two</d></c>\n  <b/>\n</a>\n"));

    EXPECT_EQ(root.name, "a");
    EXPECT_EQ(root.line, 1u);
    EXPECT_TRUE(root.text.empty());
    ASSERT_EQ(root.children.size(), 3u);
    EXPECT_EQ(root.children[0].name, "b");
    EXPECT_EQ(root.children[0].text, "one");
    EXPECT_EQ(root.children[0].line, 2u);
    EXPECT_EQ(root.children[1].children[0].text, "two");
    EXPECT_EQ(root.children[2].line, 4u);
    EXPECT_TRUE(root.children[2].text.empty());

    EXPECT_EQ(root.find_child("b"), &root.children[0]);
    EXPECT_EQ(root.find_child("c"), &root.children[1]);
    EXPECT_EQ(root.find_child("e"), nullptr);
}

/** @brief Tests that each of the five entities becomes its character. */
TEST(Small_DatasetsXmlElement, TheFiveEntitiesAreReplaced)
{
    SA_ASSERT_OK_AND_ASSIGN(const XMLElement root,
                            parse_xml("<n>a &amp; b &lt;c&gt; &quot;d&quot; &apos;e&apos;</n>"));

    EXPECT_EQ(root.text, "a & b <c> \"d\" 'e'");
}

/** @brief Tests that a declaration, a byte order mark, comments and attributes are passed. */
TEST(Small_DatasetsXmlElement, TheDeclarationCommentsAndAttributesAreSkipped)
{
    const std::string text = "\xEF\xBB\xBF<?xml version=\"1.0\"?>\n<!-- head -->\n"
                             "<a verified=\"yes\" note='x > y'><!-- in --><b k=\"v\"/>t</a>\n"
                             "<!-- tail -->\n";

    SA_ASSERT_OK_AND_ASSIGN(const XMLElement root, parse_xml(text));

    EXPECT_EQ(root.name, "a");
    EXPECT_EQ(root.line, 3u);
    EXPECT_EQ(root.text, "t");
    ASSERT_EQ(root.children.size(), 1u);
    EXPECT_EQ(root.children[0].name, "b");
}

/** @brief Tests that a closing tag of another name is refused with both lines. */
TEST(Small_DatasetsXmlElement, AClosingTagOfAnotherNameIsRefused)
{
    SA_EXPECT_FAILURE(parse_xml("<a>\n<b>\n</a>\n"), StatusCode::InvalidFile, "line 3",
                      "parse_xml", "'</a>' closes '<b>'", "line 2");
}

/** @brief Tests that an element left open is refused with the line it opened on. */
TEST(Small_DatasetsXmlElement, AnElementLeftOpenIsRefused)
{
    SA_EXPECT_FAILURE(parse_xml("<a>\n<b>text\n"), StatusCode::InvalidFile, "line 3",
                      "the element 'b', opened on line 2, is not closed");
    SA_EXPECT_FAILURE(parse_xml("<a>\n<!-- open\n"), StatusCode::InvalidFile, "line 2",
                      "a comment is not closed");
    SA_EXPECT_FAILURE(parse_xml("<a\n"), StatusCode::InvalidFile, "line 2", "has no '>'");
}

/** @brief Tests that an entity outside the five is refused with its line. */
TEST(Small_DatasetsXmlElement, AnUnknownEntityIsRefused)
{
    SA_EXPECT_FAILURE(parse_xml("<a>\n\n&nbsp;</a>"), StatusCode::InvalidFile, "line 3",
                      "'&' opens none of");
    SA_EXPECT_FAILURE(parse_xml("<a>&#65;</a>"), StatusCode::InvalidFile, "line 1");
}

/** @brief Tests that markup outside the subset is refused with its line. */
TEST(Small_DatasetsXmlElement, MarkupOutsideTheSubsetIsRefused)
{
    SA_EXPECT_FAILURE(parse_xml("<a>\n<![CDATA[x]]></a>"), StatusCode::InvalidFile, "line 2",
                      "CDATA");
    SA_EXPECT_FAILURE(parse_xml("<?xml version=\"1.0\"?>\n<!DOCTYPE a>\n<a/>"),
                      StatusCode::InvalidFile, "line 2", "declaration");
    SA_EXPECT_FAILURE(parse_xml("<a>\n\n<?pi x?></a>"), StatusCode::InvalidFile, "line 3",
                      "processing instruction");
}

/** @brief Tests that an attribute without a quoted value is refused with its name. */
TEST(Small_DatasetsXmlElement, AMalformedAttributeIsRefused)
{
    SA_EXPECT_FAILURE(parse_xml("<a>\n<b k=v/></a>"), StatusCode::InvalidFile, "line 2",
                      "attribute 'k'", "not quoted");
    SA_EXPECT_FAILURE(parse_xml("<a k></a>"), StatusCode::InvalidFile, "line 1",
                      "attribute 'k' has no '='");
}

/** @brief Tests that a document with no element, or with two roots, is refused. */
TEST(Small_DatasetsXmlElement, ADocumentNeedsExactlyOneRoot)
{
    SA_EXPECT_FAILURE(parse_xml("\n\n"), StatusCode::InvalidFile, "line 3", "holds no element");
    SA_EXPECT_FAILURE(parse_xml("text"), StatusCode::InvalidFile, "line 1",
                      "text stands where an element must open");
    SA_EXPECT_FAILURE(parse_xml("<a/>\n<b/>"), StatusCode::InvalidFile, "line 2",
                      "content follows the root element 'a'");
    SA_EXPECT_FAILURE(parse_xml("<1a/>"), StatusCode::InvalidFile, "line 1",
                      "a name must open with");
}

/** @brief Tests that elements open at once beyond K_XML_MAX_DEPTH are refused. */
TEST(Small_DatasetsXmlElement, NestingBeyondTheDepthIsRefused)
{
    std::string deep;
    for (std::int32_t level = 0; level <= K_XML_MAX_DEPTH; ++level)
        deep += "<e>\n";

    SA_EXPECT_FAILURE(parse_xml(deep), StatusCode::InvalidFile, "line 65",
                      "more than 64 elements are open at once");
}
