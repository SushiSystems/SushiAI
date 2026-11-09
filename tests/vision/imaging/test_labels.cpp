/**************************************************************************/
/* test_labels.cpp                                                        */
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
 * @file test_labels.cpp
 * @brief Tests the label parser on valid text and on every row the reference rejects.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/imaging/labels.hpp>
#include <support/tolerance.hpp>
#include <support/test_names.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Test::require_value;

    /** @brief Holds the class count the cases parse with. */
    constexpr std::int32_t K_CLASS_COUNT = 80;

    /**
     * @brief Holds the absolute tolerance of a parsed value, a fraction of the image: 2 times the
     *        float unit roundoff, one float ulp at 1, since a parsed decimal and a float literal
     *        round the same value once each.
     */
    constexpr float K_FRACTION_TOLERANCE =
        static_cast<float>(2.0 * TestSupport::unit_roundoff<float>());

    /** @brief Checks the four values of a normalized box against expected values. */
    void expect_box(const NormalizedBox& actual, float x_center, float y_center, float width,
                    float height)
    {
        EXPECT_NEAR(actual.x_center, x_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.y_center, y_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.width, width, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.height, height, K_FRACTION_TOLERANCE);
    }

    /** @brief Checks that a polygon holds exactly the listed points, in order. */
    void expect_polygon_points(const Polygon& polygon,
                               std::initializer_list<std::array<float, 2>> expected)
    {
        ASSERT_EQ(polygon.size(), expected.size());
        std::size_t index = 0;
        for (const std::array<float, 2>& point : expected)
        {
            EXPECT_NEAR(polygon[index].x, point[0], K_FRACTION_TOLERANCE) << "point " << index;
            EXPECT_NEAR(polygon[index].y, point[1], K_FRACTION_TOLERANCE) << "point " << index;
            ++index;
        }
    }

    /** @brief Describes a text, the class count it parses with and the notes a refusal names. */
    struct RefusalRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the label text the parser refuses. */
        const char* text;

        /** @brief Holds the class count the text parses with. */
        std::int32_t class_count;

        /** @brief Holds the first substring the message names; empty when unused. */
        const char* first;

        /** @brief Holds the second substring the message names; empty when unused. */
        const char* second;

        /** @brief Holds the third substring the message names; empty when unused. */
        const char* third;
    };

    /** @brief Checks that a row's text is refused as an invalid file naming its substrings. */
    void expect_refused(const RefusalRow& row)
    {
        SA_EXPECT_FAILURE(parse_label_text(row.text, row.class_count), StatusCode::InvalidFile,
                          row.first, row.second, row.third);
    }

    /** @brief Returns the name of a refusal row. */
    std::string refusal_name(const ::testing::TestParamInfo<RefusalRow>& info)
    {
        return info.param.name;
    }

    /** @brief Runs each box-row column-count refusal. */
    class Small_ImagingLabelsColumnCount : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Runs each field-is-not-a-number refusal. */
    class Small_ImagingLabelsNotANumber : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Runs each coordinate-outside-the-tolerance refusal. */
    class Small_ImagingLabelsCoordinateBand : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Runs each class-out-of-range refusal. */
    class Small_ImagingLabelsClassRange : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Runs each polygon-row point-count refusal. */
    class Small_ImagingLabelsPolygonPointCount : public ::testing::TestWithParam<RefusalRow>
    {
    };

    /** @brief Describes a text that holds no object, whatever its blank bytes. */
    struct BlankRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the text that holds no row. */
        const char* text;
    };

    /** @brief Runs each text that is a background image. */
    class Small_ImagingLabelsBackground : public ::testing::TestWithParam<BlankRow>
    {
    };

    /** @brief Describes a text whose rows are split by one kind of line ending. */
    struct LineEndingRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the text, its rows split by the ending. */
        const char* text;

        /** @brief Holds the classes the rows give, in file order. */
        std::vector<std::int32_t> classes;
    };

    /** @brief Runs each line-ending case. */
    class Small_ImagingLabelsLineEndings : public ::testing::TestWithParam<LineEndingRow>
    {
    };

    /** @brief Describes a polygon row and the box around its points. */
    struct PolygonBoundsRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the polygon row text. */
        const char* text;

        /** @brief Holds the class of the row. */
        std::int32_t class_id;

        /** @brief Holds the x centre, y centre, width and height of the box. */
        std::array<float, 4> box;
    };

    /** @brief Runs each polygon-bounds case. */
    class Small_ImagingLabelsPolygonBounds : public ::testing::TestWithParam<PolygonBoundsRow>
    {
    };

    /** @brief Describes a text with repeated rows and the objects that remain. */
    struct DuplicateRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the text with repeated rows. */
        const char* text;

        /** @brief Holds the classes that remain, in file order. */
        std::vector<std::int32_t> classes;

        /** @brief Holds the x centre, y centre, width and height of each remaining object. */
        std::vector<std::array<float, 4>> boxes;
    };

    /** @brief Runs each duplicate-removal case. */
    class Small_ImagingLabelsDuplicates : public ::testing::TestWithParam<DuplicateRow>
    {
    };
} // namespace

TEST(Small_ImagingLabels, ParsesBoxRowsInFileOrder)
{
    const FileLabels labels = require_value(parse_label_text(
        "0 0.5 0.25 0.5 0.125\n"
        "3 0.25 0.75 0.125 0.0625\n",
        K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{0, 3}));
    expect_box(labels.boxes()[0], 0.5f, 0.25f, 0.5f, 0.125f);
    expect_box(labels.boxes()[1], 0.25f, 0.75f, 0.125f, 0.0625f);
}

/** @brief Tests that a text with no row gives an empty set, a background image. */
TEST_P(Small_ImagingLabelsBackground, TextWithNoRowIsABackgroundImage)
{
    const FileLabels labels = require_value(parse_label_text(GetParam().text, K_CLASS_COUNT));

    EXPECT_TRUE(labels.is_empty());
    EXPECT_TRUE(labels.classes().empty());
}

INSTANTIATE_TEST_SUITE_P(Rows, Small_ImagingLabelsBackground,
                         ::testing::Values(BlankRow{"EmptyText", ""},
                                           BlankRow{"BlankLines", "\n  \n\t\r\n \r\n"}),
                         [](const ::testing::TestParamInfo<BlankRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });

TEST(Small_ImagingLabels, AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows)
{
    const FileLabels labels = require_value(parse_label_text(
        "\n"
        "  1\t0.5   0.5 0.25\t\t0.25  \n"
        "\n"
        "2 0.25 0.25 0.5 0.5",
        K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2}));
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.25f, 0.25f);
    expect_box(labels.boxes()[1], 0.25f, 0.25f, 0.5f, 0.5f);
}

/** @brief Tests that rows are split by every kind of line ending the parser knows. */
TEST_P(Small_ImagingLabelsLineEndings, RowsAreSplitByEveryLineEnding)
{
    const LineEndingRow& row = GetParam();

    const FileLabels labels = require_value(parse_label_text(row.text, K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), row.classes.size());
    EXPECT_EQ(labels.classes(), row.classes);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsLineEndings,
    ::testing::Values(
        LineEndingRow{"CarriageReturn",
                      "1 0.5 0.5 0.25 0.25\r\n"
                      "2 0.25 0.25 0.5 0.5\r"
                      "4 0.75 0.75 0.5 0.5\r\n",
                      {1, 2, 4}},
        LineEndingRow{"VerticalTabAndFormFeed",
                      "1 0.5 0.5 0.25 0.25\v2 0.25 0.25 0.5 0.5\f4 0.75 0.75 0.5 0.5",
                      {1, 2, 4}},
        LineEndingRow{"SeparatorBytes",
                      "1 0.5 0.5 0.25 0.25\x1c"
                      "2 0.25 0.25 0.5 0.5\x1d"
                      "4 0.75 0.75 0.5 0.5\x1e"
                      "6 0.5 0.5 0.5 0.5",
                      {1, 2, 4, 6}}),
    [](const ::testing::TestParamInfo<LineEndingRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

TEST(Small_ImagingLabels, AcceptsTheUnitSeparatorByteBetweenFields)
{
    const FileLabels labels = require_value(parse_label_text("3\x1f"
                                               "0.5\x1f"
                                               "0.25 0.5\x1f\x1f"
                                               "0.125", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 3);
    expect_box(labels.boxes()[0], 0.5f, 0.25f, 0.5f, 0.125f);
}

TEST(Small_ImagingLabels, AcceptsExponentsBareDecimalPointsAndAPlusSign)
{
    const FileLabels labels =
        require_value(parse_label_text("+7.0 5e-1 .5 2.5E-1 +0.125", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 7);
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.25f, 0.125f);
}

/** @brief Tests that a box row of any column count but five is refused, naming the count. */
TEST_P(Small_ImagingLabelsColumnCount, RejectsABoxRowOfTheWrongColumnCount)
{
    expect_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsColumnCount,
    ::testing::Values(
        RefusalRow{"FourColumns", "0 0.5 0.5 0.5", K_CLASS_COUNT, "parse_label_text: line 1",
                   "5 columns", "this row has 4"},
        RefusalRow{"SixColumns", "0 0.5 0.5 0.5 0.5 0.9", K_CLASS_COUNT, "line 1", "5 columns",
                   "this row has 6"},
        RefusalRow{"OneColumn", "0", K_CLASS_COUNT, "5 columns", "this row has 1", ""}),
    refusal_name);

TEST(Small_ImagingLabels, NamesTheLineOfTheRejectedRow)
{
    SA_EXPECT_FAILURE(parse_label_text("0 0.5 0.5 0.5 0.5\n\n1 0.5 0.5 0.5\n", K_CLASS_COUNT),
                      StatusCode::InvalidFile, "line 3");
}

/** @brief Tests that a field that is not a finite number is refused, quoting the field. */
TEST_P(Small_ImagingLabelsNotANumber, RejectsAFieldThatIsNotANumber)
{
    expect_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsNotANumber,
    ::testing::Values(
        RefusalRow{"Letters", "0 0.5 abc 0.5 0.5", K_CLASS_COUNT, "line 1", "'abc'",
                   "not a number"},
        RefusalRow{"CharactersAfterTheNumber", "0 0.5 0.5x 0.5 0.5", K_CLASS_COUNT, "'0.5x'",
                   "not a number", ""},
        RefusalRow{"DecimalComma", "0 0,5 0.5 0.5 0.5", K_CLASS_COUNT, "'0,5'", "not a number",
                   ""},
        RefusalRow{"DoubledSign", "0 +-0.5 0.5 0.5 0.5", K_CLASS_COUNT, "'+-0.5'",
                   "not a number", ""},
        RefusalRow{"NotANumberText", "0 nan 0.5 0.5 0.5", K_CLASS_COUNT, "'nan'", "", ""},
        RefusalRow{"InfinityText", "0 0.5 inf 0.5 0.5", K_CLASS_COUNT, "'inf'", "", ""},
        RefusalRow{"OverflowText", "0 0.5 0.5 1e999 0.5", K_CLASS_COUNT, "'1e999'", "", ""}),
    refusal_name);

TEST(Small_ImagingLabels, RejectsAFiniteValueAboveTheFloatRange)
{
    SA_EXPECT_FAILURE(parse_label_text("0 1e39 0.5 0.5 0.5", K_CLASS_COUNT),
                      StatusCode::InvalidFile, "'1e39'", "not a finite number");
}

TEST(Small_ImagingLabels, RejectsANonzeroValueBelowTheSmallestNormalDouble)
{
    SA_EXPECT_FAILURE(parse_label_text("0 1e-310 0.5 0.5 0.5", K_CLASS_COUNT),
                      StatusCode::InvalidFile, "line 1", "'1e-310'");
    SA_EXPECT_FAILURE(parse_label_text("0 1e-999 0.5 0.5 0.5", K_CLASS_COUNT),
                      StatusCode::InvalidFile, "line 1", "'1e-999'");
}

TEST(Small_ImagingLabels, AcceptsZeroAndTheSmallestNormalDouble)
{
    const FileLabels labels =
        require_value(parse_label_text("0 0.0 2.2250738585072014e-308 0.5 0.5", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 0.0f, 0.0f, 0.5f, 0.5f);
}

/** @brief Tests that a coordinate outside the tolerance band around the image is refused. */
TEST_P(Small_ImagingLabelsCoordinateBand, RejectsACoordinateOutsideTheToleranceBand)
{
    expect_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsCoordinateBand,
    ::testing::Values(
        RefusalRow{"CenterBelow", "0 -0.02 0.5 0.5 0.5", K_CLASS_COUNT, "line 1", "negative", ""},
        RefusalRow{"HeightBelow", "0 0.5 0.5 0.5 -0.5", K_CLASS_COUNT, "negative", "", ""},
        RefusalRow{"CenterAbove", "0 1.02 0.5 0.5 0.5", K_CLASS_COUNT, "line 1", "above", ""},
        RefusalRow{"HeightAbove", "0 0.5 0.5 0.5 640", K_CLASS_COUNT, "above", "", ""}),
    refusal_name);

TEST(Small_ImagingLabels, StoresACoordinateInsideTheToleranceUnclipped)
{
    const FileLabels labels =
        require_value(parse_label_text("0 1.005 -0.005 0.5 0.5", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 1.005f, -0.005f, 0.5f, 0.5f);
}

TEST(Small_ImagingLabels, AcceptsACoordinateOnEitherToleranceBound)
{
    const FileLabels labels =
        require_value(parse_label_text("0 1.01 -0.01 0.5 0.5", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 1.01f, -0.01f, 0.5f, 0.5f);
}

/** @brief Tests that a class field that is negative, fractional or past the count is refused. */
TEST_P(Small_ImagingLabelsClassRange, RejectsAClassOutsideTheClassCount)
{
    expect_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsClassRange,
    ::testing::Values(
        RefusalRow{"Negative", "-1 0.5 0.5 0.5 0.5", K_CLASS_COUNT, "line 1", "class", "negative"},
        RefusalRow{"Fraction", "1.5 0.5 0.5 0.5 0.5", K_CLASS_COUNT, "class", "whole number", ""},
        RefusalRow{"AtTheClassCount", "80 0.5 0.5 0.5 0.5", K_CLASS_COUNT, "class",
                   "out of range", "80 classes"},
        RefusalRow{"AtASingleClassCount", "1 0.5 0.5 0.5 0.5", 1, "out of range", "1 classes",
                   ""}),
    refusal_name);

TEST(Small_ImagingLabels, AcceptsTheLastClass)
{
    const FileLabels labels = require_value(parse_label_text("79 0.5 0.5 0.5 0.5", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 79);
}

/** @brief Tests that a repeated row is removed, keeping the first and the file order. */
TEST_P(Small_ImagingLabelsDuplicates, RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder)
{
    const DuplicateRow& row = GetParam();

    const FileLabels labels = require_value(parse_label_text(row.text, K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), row.classes.size());
    EXPECT_EQ(labels.classes(), row.classes);
    for (std::size_t i = 0; i < row.boxes.size(); ++i)
        expect_box(labels.boxes()[i], row.boxes[i][0], row.boxes[i][1], row.boxes[i][2],
                   row.boxes[i][3]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsDuplicates,
    ::testing::Values(
        DuplicateRow{"IdenticalRows",
                     "5 0.5 0.5 0.25 0.25\n"
                     "2 0.25 0.25 0.5 0.5\n"
                     "5 0.5 0.5 0.25 0.25\n"
                     "9 0.75 0.75 0.125 0.125\n"
                     "2 0.25 0.25 0.5 0.5\n",
                     {5, 2, 9},
                     {{0.5f, 0.5f, 0.25f, 0.25f},
                      {0.25f, 0.25f, 0.5f, 0.5f},
                      {0.75f, 0.75f, 0.125f, 0.125f}}},
        DuplicateRow{"ADifferentSpelling",
                     "5 0.5 0.5 0.25 0.25\n"
                     "5.0 5e-1 0.50 .25 0.250\n",
                     {5},
                     {{0.5f, 0.5f, 0.25f, 0.25f}}}),
    [](const ::testing::TestParamInfo<DuplicateRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

TEST(Small_ImagingLabels, KeepsRowsThatDifferInOneValue)
{
    const FileLabels labels = require_value(parse_label_text(
        "5 0.5 0.5 0.25 0.25\n"
        "6 0.5 0.5 0.25 0.25\n"
        "5 0.5 0.5 0.25 0.5\n",
        K_CLASS_COUNT));

    EXPECT_EQ(labels.size(), 3u);
}

/** @brief Tests that a polygon row reduces to the box around its points. */
TEST_P(Small_ImagingLabelsPolygonBounds, ReducesAPolygonRowToTheBoxAroundItsPoints)
{
    const PolygonBoundsRow& row = GetParam();

    const FileLabels labels = require_value(parse_label_text(row.text, K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], row.class_id);
    expect_box(labels.boxes()[0], row.box[0], row.box[1], row.box[2], row.box[3]);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsPolygonBounds,
    ::testing::Values(
        PolygonBoundsRow{"ThreePoints", "1 0.25 0.5 0.75 0.25 0.5 0.75", 1,
                         {0.5f, 0.5f, 0.5f, 0.5f}},
        PolygonBoundsRow{"FivePoints", "2 0.5 0.125 0.875 0.5 0.5 0.625 0.125 0.5 0.25 0.25", 2,
                         {0.5f, 0.375f, 0.75f, 0.5f}}),
    [](const ::testing::TestParamInfo<PolygonBoundsRow>& info)
    {
        return ::SushiAI::Test::parameter_name(info.param.name);
    });

/** @brief Tests that a polygon row keeps its points, in file order, beside its box. */
TEST(Small_ImagingLabels, KeepsThePointsOfAPolygonRowInFileOrder)
{
    const FileLabels labels = require_value(parse_label_text(
        "2 0.5 0.125 0.875 0.5 0.5 0.625 0.125 0.5 0.25 0.25", K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 1u);
    ASSERT_EQ(labels.polygons().size(), 1u);
    expect_polygon_points(labels.polygons()[0],
                          {{0.5f, 0.125f},
                           {0.875f, 0.5f},
                           {0.5f, 0.625f},
                           {0.125f, 0.5f},
                           {0.25f, 0.25f}});
}

/** @brief Tests that a box row gives its object a polygon of no point. */
TEST(Small_ImagingLabels, GivesABoxRowNoPolygon)
{
    const FileLabels labels =
        require_value(parse_label_text("0 0.5 0.25 0.5 0.125\n3 0.25 0.75 0.125 0.0625\n",
                                       K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 2u);
    ASSERT_EQ(labels.polygons().size(), 2u);
    EXPECT_TRUE(labels.polygons()[0].empty());
    EXPECT_TRUE(labels.polygons()[1].empty());
}

TEST(Small_ImagingLabels, RejectsABoxRowInAPolygonFile)
{
    SA_EXPECT_FAILURE(parse_label_text("1 0.25 0.5 0.75 0.25 0.5 0.75\n" "0 0.5 0.5 0.5 0.5\n",
                                       K_CLASS_COUNT),
                      StatusCode::InvalidFile, "line 2", "polygon");
}

/** @brief Tests that a polygon row with a half point or under three points is refused. */
TEST_P(Small_ImagingLabelsPolygonPointCount, RejectsAPolygonRowOfTheWrongPointCount)
{
    expect_refused(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImagingLabelsPolygonPointCount,
    ::testing::Values(
        RefusalRow{"AHalfPoint", "1 0.25 0.5 0.75 0.25 0.5 0.75 0.5", K_CLASS_COUNT, "line 1",
                   "polygon", "this row has 8"},
        RefusalRow{"FewerThanThreePoints",
                   "1 0.25 0.5 0.75 0.25 0.5 0.75\n"
                   "1 0.25 0.5\n",
                   K_CLASS_COUNT, "line 2", "three points", "this row has 3"}),
    refusal_name);

TEST(Small_ImagingLabels, RejectsAPolygonWhoseBoxLeavesTheImage)
{
    SA_EXPECT_FAILURE(parse_label_text("1 0.0 0.5 1.5 0.25 0.5 0.75", K_CLASS_COUNT),
                      StatusCode::InvalidFile, "above");
}

TEST(Small_ImagingLabels, RemovesARepeatedPolygonAndKeepsOneWithTheSameBox)
{
    const FileLabels labels = require_value(parse_label_text(
        "1 0.25 0.25 0.75 0.25 0.75 0.75\n"
        "1 0.25 0.25 0.75 0.25 0.75 0.75\n"
        "1 0.25 0.25 0.25 0.75 0.75 0.75\n",
        K_CLASS_COUNT));

    ASSERT_EQ(labels.size(), 2u);
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.5f, 0.5f);
    expect_box(labels.boxes()[1], 0.5f, 0.5f, 0.5f, 0.5f);
    ASSERT_EQ(labels.polygons()[0].size(), 3u);
    ASSERT_EQ(labels.polygons()[1].size(), 3u);
    EXPECT_NEAR(labels.polygons()[0][1].x, 0.75f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(labels.polygons()[1][1].x, 0.25f, K_FRACTION_TOLERANCE);
}
