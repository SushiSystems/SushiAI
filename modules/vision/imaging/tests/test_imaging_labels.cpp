/**************************************************************************/
/* test_imaging_labels.cpp                                                */
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
 * @file test_imaging_labels.cpp
 * @brief Tests the label parser on valid text and on every row the reference rejects.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/imaging/labels.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI::Vision::Imaging;

namespace
{
    /** @brief Holds the class count the cases parse with. */
    constexpr std::int32_t K_CLASS_COUNT = 80;

    /** @brief Holds the absolute tolerance of a parsed value, a fraction of the image. */
    constexpr float K_FRACTION_TOLERANCE = 1e-7f;

    /** @brief Checks the four values of a normalized box against expected values. */
    void expect_box(const NormalizedBox& actual, float x_center, float y_center, float width,
                    float height)
    {
        EXPECT_NEAR(actual.x_center, x_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.y_center, y_center, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.width, width, K_FRACTION_TOLERANCE);
        EXPECT_NEAR(actual.height, height, K_FRACTION_TOLERANCE);
    }
} // namespace

TEST(Unit_ImagingLabels, ParsesBoxRowsInFileOrder)
{
    const FileLabels labels = parse_label_text(
        "0 0.5 0.25 0.5 0.125\n"
        "3 0.25 0.75 0.125 0.0625\n",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{0, 3}));
    expect_box(labels.boxes()[0], 0.5f, 0.25f, 0.5f, 0.125f);
    expect_box(labels.boxes()[1], 0.25f, 0.75f, 0.125f, 0.0625f);
}

TEST(Unit_ImagingLabels, EmptyTextIsABackgroundImage)
{
    const FileLabels labels = parse_label_text("", K_CLASS_COUNT);

    EXPECT_TRUE(labels.is_empty());
    EXPECT_TRUE(labels.classes().empty());
}

TEST(Unit_ImagingLabels, TextOfBlankLinesIsABackgroundImage)
{
    EXPECT_TRUE(parse_label_text("\n  \n\t\r\n \r\n", K_CLASS_COUNT).is_empty());
}

TEST(Unit_ImagingLabels, AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows)
{
    const FileLabels labels = parse_label_text(
        "\n"
        "  1\t0.5   0.5 0.25\t\t0.25  \n"
        "\n"
        "2 0.25 0.25 0.5 0.5",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2}));
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.25f, 0.25f);
    expect_box(labels.boxes()[1], 0.25f, 0.25f, 0.5f, 0.5f);
}

TEST(Unit_ImagingLabels, AcceptsCarriageReturnLineEndings)
{
    const FileLabels labels = parse_label_text(
        "1 0.5 0.5 0.25 0.25\r\n"
        "2 0.25 0.25 0.5 0.5\r"
        "4 0.75 0.75 0.5 0.5\r\n",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2, 4}));
}

TEST(Unit_ImagingLabels, AcceptsVerticalTabAndFormFeedLineEndings)
{
    const FileLabels labels = parse_label_text(
        "1 0.5 0.5 0.25 0.25\v2 0.25 0.25 0.5 0.5\f4 0.75 0.75 0.5 0.5", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2, 4}));
}

TEST(Unit_ImagingLabels, AcceptsTheSeparatorBytesAsLineEndings)
{
    const FileLabels labels = parse_label_text(
        "1 0.5 0.5 0.25 0.25\x1c"
        "2 0.25 0.25 0.5 0.5\x1d"
        "4 0.75 0.75 0.5 0.5\x1e"
        "6 0.5 0.5 0.5 0.5",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{1, 2, 4, 6}));
}

TEST(Unit_ImagingLabels, AcceptsTheUnitSeparatorByteBetweenFields)
{
    const FileLabels labels = parse_label_text("3\x1f"
                                               "0.5\x1f"
                                               "0.25 0.5\x1f\x1f"
                                               "0.125",
                                               K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 3);
    expect_box(labels.boxes()[0], 0.5f, 0.25f, 0.5f, 0.125f);
}

TEST(Unit_ImagingLabels, AcceptsExponentsBareDecimalPointsAndAPlusSign)
{
    const FileLabels labels = parse_label_text("+7.0 5e-1 .5 2.5E-1 +0.125", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 7);
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.25f, 0.125f);
}

TEST(Unit_ImagingLabels, RejectsARowOfFourColumns)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 0.5", K_CLASS_COUNT),
                    "parse_label_text: line 1",
                    "5 columns", "this row has 4");
}

TEST(Unit_ImagingLabels, RejectsARowOfSixColumns)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 0.5 0.5 0.9", K_CLASS_COUNT), "line 1",
                    "5 columns", "this row has 6");
}

TEST(Unit_ImagingLabels, RejectsARowOfOneColumn)
{
    SA_EXPECT_ERROR((void)parse_label_text("0", K_CLASS_COUNT), "5 columns", "this row has 1");
}

TEST(Unit_ImagingLabels, NamesTheLineOfTheRejectedRow)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 0.5 0.5\n\n1 0.5 0.5 0.5\n", K_CLASS_COUNT),
                    "line 3");
}

TEST(Unit_ImagingLabels, RejectsAFieldThatIsNotANumber)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 abc 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "'abc'", "not a number");
}

TEST(Unit_ImagingLabels, RejectsAFieldWithCharactersAfterTheNumber)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5x 0.5 0.5", K_CLASS_COUNT), "'0.5x'",
                    "not a number");
}

TEST(Unit_ImagingLabels, RejectsADecimalComma)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 0,5 0.5 0.5 0.5", K_CLASS_COUNT), "'0,5'",
                    "not a number");
}

TEST(Unit_ImagingLabels, RejectsADoubledSign)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 +-0.5 0.5 0.5 0.5", K_CLASS_COUNT), "'+-0.5'",
                    "not a number");
}

TEST(Unit_ImagingLabels, RejectsNotANumberAndInfinity)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 nan 0.5 0.5 0.5", K_CLASS_COUNT), "'nan'");
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 inf 0.5 0.5", K_CLASS_COUNT), "'inf'");
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 1e999 0.5", K_CLASS_COUNT), "'1e999'");
}

TEST(Unit_ImagingLabels, RejectsAFiniteValueAboveTheFloatRange)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 1e39 0.5 0.5 0.5", K_CLASS_COUNT), "'1e39'",
                    "not a finite number");
}

TEST(Unit_ImagingLabels, RejectsANonzeroValueBelowTheSmallestNormalDouble)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 1e-310 0.5 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "'1e-310'");
    SA_EXPECT_ERROR((void)parse_label_text("0 1e-999 0.5 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "'1e-999'");
}

TEST(Unit_ImagingLabels, AcceptsZeroAndTheSmallestNormalDouble)
{
    const FileLabels labels =
        parse_label_text("0 0.0 2.2250738585072014e-308 0.5 0.5", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 0.0f, 0.0f, 0.5f, 0.5f);
}

TEST(Unit_ImagingLabels, RejectsACoordinateBelowTheTolerance)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 -0.02 0.5 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "negative");
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 0.5 -0.5", K_CLASS_COUNT), "negative");
}

TEST(Unit_ImagingLabels, RejectsACoordinateAboveTheTolerance)
{
    SA_EXPECT_ERROR((void)parse_label_text("0 1.02 0.5 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "above");
    SA_EXPECT_ERROR((void)parse_label_text("0 0.5 0.5 0.5 640", K_CLASS_COUNT), "above");
}

TEST(Unit_ImagingLabels, StoresACoordinateInsideTheToleranceUnclipped)
{
    const FileLabels labels = parse_label_text("0 1.005 -0.005 0.5 0.5", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 1.005f, -0.005f, 0.5f, 0.5f);
}

TEST(Unit_ImagingLabels, AcceptsACoordinateOnEitherToleranceBound)
{
    const FileLabels labels = parse_label_text("0 1.01 -0.01 0.5 0.5", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 1.01f, -0.01f, 0.5f, 0.5f);
}

TEST(Unit_ImagingLabels, RejectsANegativeClass)
{
    SA_EXPECT_ERROR((void)parse_label_text("-1 0.5 0.5 0.5 0.5", K_CLASS_COUNT), "line 1",
                    "class", "negative");
}

TEST(Unit_ImagingLabels, RejectsAClassWithAFraction)
{
    SA_EXPECT_ERROR((void)parse_label_text("1.5 0.5 0.5 0.5 0.5", K_CLASS_COUNT), "class",
                    "whole number");
}

TEST(Unit_ImagingLabels, RejectsAClassAtOrAboveTheClassCount)
{
    SA_EXPECT_ERROR((void)parse_label_text("80 0.5 0.5 0.5 0.5", K_CLASS_COUNT), "class",
                    "out of range", "80 classes");
    SA_EXPECT_ERROR((void)parse_label_text("1 0.5 0.5 0.5 0.5", 1), "out of range", "1 classes");
}

TEST(Unit_ImagingLabels, AcceptsTheLastClass)
{
    const FileLabels labels = parse_label_text("79 0.5 0.5 0.5 0.5", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 79);
}

TEST(Unit_ImagingLabels, RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder)
{
    const FileLabels labels = parse_label_text(
        "5 0.5 0.5 0.25 0.25\n"
        "2 0.25 0.25 0.5 0.5\n"
        "5 0.5 0.5 0.25 0.25\n"
        "9 0.75 0.75 0.125 0.125\n"
        "2 0.25 0.25 0.5 0.5\n",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels.classes(), (std::vector<std::int32_t>{5, 2, 9}));
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.25f, 0.25f);
    expect_box(labels.boxes()[1], 0.25f, 0.25f, 0.5f, 0.5f);
    expect_box(labels.boxes()[2], 0.75f, 0.75f, 0.125f, 0.125f);
}

TEST(Unit_ImagingLabels, RemovesARowThatRepeatsAnotherInADifferentSpelling)
{
    const FileLabels labels = parse_label_text(
        "5 0.5 0.5 0.25 0.25\n"
        "5.0 5e-1 0.50 .25 0.250\n",
        K_CLASS_COUNT);

    EXPECT_EQ(labels.size(), 1u);
}

TEST(Unit_ImagingLabels, KeepsRowsThatDifferInOneValue)
{
    const FileLabels labels = parse_label_text(
        "5 0.5 0.5 0.25 0.25\n"
        "6 0.5 0.5 0.25 0.25\n"
        "5 0.5 0.5 0.25 0.5\n",
        K_CLASS_COUNT);

    EXPECT_EQ(labels.size(), 3u);
}

TEST(Unit_ImagingLabels, ReducesAPolygonRowToTheBoxAroundItsPoints)
{
    const FileLabels labels = parse_label_text("1 0.25 0.5 0.75 0.25 0.5 0.75", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels.classes()[0], 1);
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.5f, 0.5f);
}

/** @brief Tests that a polygon row keeps its points, in file order, beside its box. */
TEST(Unit_ImagingLabels, KeepsThePointsOfAPolygonRowInFileOrder)
{
    const FileLabels labels = parse_label_text(
        "2 0.5 0.125 0.875 0.5 0.5 0.625 0.125 0.5 0.25 0.25", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    ASSERT_EQ(labels.polygons().size(), 1u);
    const Polygon& polygon = labels.polygons()[0];
    ASSERT_EQ(polygon.size(), 5u);
    const float expected[5][2] = {
        {0.5f, 0.125f}, {0.875f, 0.5f}, {0.5f, 0.625f}, {0.125f, 0.5f}, {0.25f, 0.25f}};
    for (std::size_t i = 0; i < polygon.size(); ++i)
    {
        EXPECT_NEAR(polygon[i].x, expected[i][0], K_FRACTION_TOLERANCE) << "point " << i;
        EXPECT_NEAR(polygon[i].y, expected[i][1], K_FRACTION_TOLERANCE) << "point " << i;
    }
}

/** @brief Tests that a box row gives its object a polygon of no point. */
TEST(Unit_ImagingLabels, GivesABoxRowNoPolygon)
{
    const FileLabels labels =
        parse_label_text("0 0.5 0.25 0.5 0.125\n3 0.25 0.75 0.125 0.0625\n", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 2u);
    ASSERT_EQ(labels.polygons().size(), 2u);
    EXPECT_TRUE(labels.polygons()[0].empty());
    EXPECT_TRUE(labels.polygons()[1].empty());
}

TEST(Unit_ImagingLabels, ReducesAPolygonOfManyPointsToItsBounds)
{
    const FileLabels labels = parse_label_text(
        "2 0.5 0.125 0.875 0.5 0.5 0.625 0.125 0.5 0.25 0.25", K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 1u);
    expect_box(labels.boxes()[0], 0.5f, 0.375f, 0.75f, 0.5f);
}

TEST(Unit_ImagingLabels, RejectsABoxRowInAPolygonFile)
{
    SA_EXPECT_ERROR((void)parse_label_text("1 0.25 0.5 0.75 0.25 0.5 0.75\n"
                                           "0 0.5 0.5 0.5 0.5\n",
                                           K_CLASS_COUNT),
                    "line 2", "polygon");
}

TEST(Unit_ImagingLabels, RejectsAPolygonRowWithAHalfPoint)
{
    SA_EXPECT_ERROR(
        (void)parse_label_text("1 0.25 0.5 0.75 0.25 0.5 0.75 0.5", K_CLASS_COUNT), "line 1",
        "polygon", "this row has 8");
}

TEST(Unit_ImagingLabels, RejectsAPolygonRowOfFewerThanThreePoints)
{
    SA_EXPECT_ERROR((void)parse_label_text("1 0.25 0.5 0.75 0.25 0.5 0.75\n"
                                           "1 0.25 0.5\n",
                                           K_CLASS_COUNT),
                    "line 2", "three points", "this row has 3");
}

TEST(Unit_ImagingLabels, RejectsAPolygonWhoseBoxLeavesTheImage)
{
    SA_EXPECT_ERROR((void)parse_label_text("1 0.0 0.5 1.5 0.25 0.5 0.75", K_CLASS_COUNT), "above");
}

TEST(Unit_ImagingLabels, RemovesARepeatedPolygonAndKeepsOneWithTheSameBox)
{
    const FileLabels labels = parse_label_text(
        "1 0.25 0.25 0.75 0.25 0.75 0.75\n"
        "1 0.25 0.25 0.75 0.25 0.75 0.75\n"
        "1 0.25 0.25 0.25 0.75 0.75 0.75\n",
        K_CLASS_COUNT);

    ASSERT_EQ(labels.size(), 2u);
    expect_box(labels.boxes()[0], 0.5f, 0.5f, 0.5f, 0.5f);
    expect_box(labels.boxes()[1], 0.5f, 0.5f, 0.5f, 0.5f);
    ASSERT_EQ(labels.polygons()[0].size(), 3u);
    ASSERT_EQ(labels.polygons()[1].size(), 3u);
    EXPECT_NEAR(labels.polygons()[0][1].x, 0.75f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(labels.polygons()[1][1].x, 0.25f, K_FRACTION_TOLERANCE);
}
