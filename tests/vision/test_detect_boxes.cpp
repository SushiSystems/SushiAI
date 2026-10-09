/**************************************************************************/
/* test_detect_boxes.cpp                                                  */
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
 * @file test_detect_boxes.cpp
 * @brief Tests box coding, box formats and IoU against hand cases and the reference decode.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/detect/anchors.hpp>
#include <SushiAI/vision/detect/box_coding.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/iou.hpp>
#include <SushiAI/vision/detect/level_rows.hpp>
#include "../common/reference_fixture.hpp"
#include "../common/test_throws.hpp"

namespace
{
    using SushiAI::Vision::Detect::FeatureLevel;
    using SushiAI::Test::near_reference;
    using SushiAI::Test::ReferenceFile;
    using SushiAI::Test::ReferenceTensor;

    /** @brief Holds the tolerance of hand cases whose values float32 represents exactly. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Holds the tolerance of a decode in pixels: two float32 steps at 640. */
    constexpr float K_DECODE_TOLERANCE = 1.3e-4f;

    /** @brief Holds the tolerance of decode then encode in grid units: two steps at 80. */
    constexpr float K_ROUND_TRIP_TOLERANCE = 1.6e-5f;

    /** @brief Holds the tolerance of an IoU value, eight float32 steps at 1. */
    constexpr float K_IOU_TOLERANCE = 1e-6f;

    /** @brief Holds the anchors of a 640 px input, from "levels" in MANIFEST.json. */
    constexpr std::size_t K_REFERENCE_ANCHORS = 8400;

    /** @brief Holds the levels MANIFEST.json records under "levels" for a 640 px input. */
    const std::vector<FeatureLevel> K_REFERENCE_LEVELS{{80, 80, 8.0f}, {40, 40, 16.0f},
                                                       {20, 20, 32.0f}};

    /** @brief Holds the reference anchor grid in grid units. */
    struct ReferenceGrid
    {
        std::vector<float> points;  /**< `[8400, 2]`. */
        std::vector<float> strides; /**< `[8400]`. */
    };

    /** @brief Returns the anchor grid of the reference levels. */
    ReferenceGrid make_reference_grid()
    {
        ReferenceGrid grid;
        grid.points.resize(K_REFERENCE_ANCHORS * 2);
        grid.strides.resize(K_REFERENCE_ANCHORS);
        SushiAI::Vision::Detect::make_anchors(K_REFERENCE_LEVELS, grid.points, grid.strides);
        return grid;
    }

    /**
     * @brief Joins a branch's three raw box outputs into `[8400, 4]`.
     * @return An empty string, or the reason a level could not be read.
     */
    std::string join_raw_boxes(ReferenceFile& file, const std::string& branch,
                               std::vector<float>& joined)
    {
        joined.assign(K_REFERENCE_ANCHORS * 4, 0.0f);
        std::int64_t offset = 0;
        for (std::size_t index = 0; index < K_REFERENCE_LEVELS.size(); ++index)
        {
            const FeatureLevel& level = K_REFERENCE_LEVELS[index];
            const ReferenceTensor raw =
                file.read(branch + ".level" + std::to_string(index) + ".boxes");
            if (!raw.found)
                return raw.reason;

            const std::int64_t cells = level.height * level.width;
            SushiAI::Vision::Detect::join_level(raw.floats(), {1, 8400, 4, offset, cells}, joined);
            offset += cells;
        }
        return {};
    }
}

/** @brief Tests that decoding subtracts left and top from the point and adds right and bottom. */
TEST(Unit_DetectBoxes, DecodesDistancesToCorners)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{1.5f - 0.5f, 2.5f - 1.0f, 1.5f + 1.5f, 2.5f + 2.0f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that the centre-size decode is the corner decode's midpoint and extent. */
TEST(Unit_DetectBoxes, DecodesDistancesToCentreAndSize)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::decode_xywh(distances, points, boxes);

    const std::vector<float> expected{(1.0f + 3.0f) / 2.0f, (1.5f + 4.5f) / 2.0f, 3.0f - 1.0f,
                                      4.5f - 1.5f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that the anchor points repeat for every image of a batch. */
TEST(Unit_DetectBoxes, RepeatsTheAnchorPointsForEachImage)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> distances{0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f,
                                       0.0f, 0.0f, 2.0f, 2.0f, 0.5f, 0.5f, 1.5f, 1.5f};
    std::vector<float> boxes(16);
    SushiAI::Vision::Detect::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0.5f, 2.5f, 0.5f,
                                      0.5f, 0.5f, 2.5f, 2.5f, 1.0f, 0.0f, 3.0f, 2.0f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that encoding a decoded box returns the distances it was decoded from. */
TEST(Unit_DetectBoxes, EncodeInvertsDecode)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f, 0.5f, 1.5f};
    const std::vector<float> distances{0.25f, 0.5f, 0.75f, 1.0f, 2.0f, 0.0f,
                                       0.5f,  3.0f, 1.5f,  1.5f, 0.0f, 4.0f};
    std::vector<float> boxes(12);
    std::vector<float> recovered(12);
    SushiAI::Vision::Detect::decode_xyxy(distances, points, boxes);
    SushiAI::Vision::Detect::encode_distances(points, boxes, recovered);

    EXPECT_TRUE(near_reference(recovered, distances, K_EXACT));
}

/** @brief Tests that scaling multiplies all four values of a box by its anchor's stride. */
TEST(Unit_DetectBoxes, ScalesEachBoxByItsAnchorStride)
{
    const std::vector<float> boxes{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> strides{8.0f, 16.0f};
    std::vector<float> scaled(8);
    SushiAI::Vision::Detect::scale_boxes(boxes, strides, scaled);

    const std::vector<float> expected{8.0f, 16.0f, 24.0f, 32.0f, 16.0f, 32.0f, 48.0f, 64.0f};
    EXPECT_TRUE(near_reference(scaled, expected, K_EXACT));
}

/** @brief Tests that corners convert to centre and size and back to the same corners. */
TEST(Unit_DetectBoxes, ConvertsBetweenCornersAndCentreSize)
{
    const std::vector<float> corners{1.0f, 2.0f, 5.0f, 8.0f};
    std::vector<float> centred(4);
    std::vector<float> back(4);
    SushiAI::Vision::Detect::xyxy_to_xywh(corners, centred);
    SushiAI::Vision::Detect::xywh_to_xyxy(centred, back);

    const std::vector<float> expected{(1.0f + 5.0f) / 2.0f, (2.0f + 8.0f) / 2.0f, 5.0f - 1.0f,
                                      8.0f - 2.0f};
    EXPECT_TRUE(near_reference(centred, expected, K_EXACT));
    EXPECT_TRUE(near_reference(back, corners, K_EXACT));
}

/** @brief Tests that rows which do not cover the anchors a whole number of times are refused. */
TEST(Unit_DetectBoxes, RejectsRowsThatDoNotCoverTheAnchors)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> distances(12, 1.0f);
    std::vector<float> boxes(12);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::decode_xyxy(distances, points, boxes), "2 anchors",
                    "12");
}

/** @brief Tests that decoding the reference's raw one-to-many boxes gives its xywh decode. */
TEST(Unit_DetectBoxes, ReproducesTheReferenceOneToManyDecode)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2many.decoded.boxes");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2many", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    SushiAI::Vision::Detect::decode_xywh(raw, grid.points, boxes);
    SushiAI::Vision::Detect::scale_boxes(boxes, grid.strides, boxes);

    EXPECT_TRUE(near_reference(boxes, reference.floats(), K_DECODE_TOLERANCE));
}

/** @brief Tests that decoding the reference's raw one-to-one boxes gives its xyxy decode. */
TEST(Unit_DetectBoxes, ReproducesTheReferenceOneToOneDecode)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2one.decoded.boxes");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2one", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    SushiAI::Vision::Detect::decode_xyxy(raw, grid.points, boxes);
    SushiAI::Vision::Detect::scale_boxes(boxes, grid.strides, boxes);

    EXPECT_TRUE(near_reference(boxes, reference.floats(), K_DECODE_TOLERANCE));
}

/** @brief Tests that encoding the decode of the reference's raw boxes returns the raw boxes. */
TEST(Unit_DetectBoxes, EncodeInvertsTheDecodeOfTheReferenceBoxes)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "forward.sacp");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2one", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    std::vector<float> recovered(raw.size());
    SushiAI::Vision::Detect::decode_xyxy(raw, grid.points, boxes);
    SushiAI::Vision::Detect::encode_distances(grid.points, boxes, recovered);

    EXPECT_TRUE(near_reference(recovered, raw, K_ROUND_TRIP_TOLERANCE));
}

/** @brief Tests that a box has IoU 1 and complete IoU 1 with itself. */
TEST(Unit_DetectBoxes, IdenticalBoxesScoreOne)
{
    const std::vector<float> box{0.0f, 0.0f, 2.0f, 2.0f};

    // intersection 2 * 2 = 4 and union 4 + 4 - 4 = 4; no centre shift and no aspect gap.
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(box, box), 4.0f / 4.0f,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(box, box), 1.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that disjoint boxes have IoU 0 and a complete IoU lowered by their distance. */
TEST(Unit_DetectBoxes, DisjointBoxesScoreZeroAndPayForDistance)
{
    const std::vector<float> left{0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> right{2.0f, 0.0f, 3.0f, 1.0f};

    const float hull_diagonal = 3.0f * 3.0f + 1.0f * 1.0f;
    const float centre_shift = 2.0f + 3.0f - 0.0f - 1.0f;
    const float centre_distance = (centre_shift * centre_shift) / 4.0f;
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(left, right), 0.0f,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(left, right),
                0.0f - centre_distance / hull_diagonal, K_IOU_TOLERANCE);
}

/** @brief Tests that partly overlapping squares give 1/7 and lose the centre term only. */
TEST(Unit_DetectBoxes, OverlappingSquaresLoseTheCentreTerm)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f};
    const std::vector<float> second{1.0f, 1.0f, 3.0f, 3.0f};

    const float iou = (1.0f * 1.0f) / (4.0f + 4.0f - 1.0f);
    const float hull_diagonal = 3.0f * 3.0f + 3.0f * 3.0f;
    const float centre_distance = (2.0f * 2.0f + 2.0f * 2.0f) / 4.0f;
    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(first, second), iou,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(first, second),
                iou - centre_distance / hull_diagonal, K_IOU_TOLERANCE);
}

/** @brief Tests that boxes of different aspect ratio also lose the aspect term. */
TEST(Unit_DetectBoxes, DifferentAspectRatiosLoseTheAspectTerm)
{
    const std::vector<float> wide{0.0f, 0.0f, 4.0f, 2.0f};
    const std::vector<float> square{0.0f, 0.0f, 2.0f, 2.0f};

    const double pi = std::acos(-1.0);
    const double iou = (2.0 * 2.0) / (8.0 + 4.0 - 4.0);
    const double hull_diagonal = 4.0 * 4.0 + 2.0 * 2.0;
    const double centre_distance = ((0.0 + 2.0 - 0.0 - 4.0) * (0.0 + 2.0 - 0.0 - 4.0)) / 4.0;
    const double angle_gap = std::atan(2.0 / 2.0) - std::atan(4.0 / 2.0);
    const double aspect = 4.0 / (pi * pi) * angle_gap * angle_gap;
    const double trade_off = aspect / (1.0 - iou + aspect);
    const double expected = iou - (centre_distance / hull_diagonal + aspect * trade_off);

    EXPECT_NEAR(SushiAI::Vision::Detect::intersection_over_union(wide, square), iou,
                K_IOU_TOLERANCE);
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(wide, square), expected, K_IOU_TOLERANCE);
}

/** @brief Tests that a zero-size box gives finite values: IoU 0 and complete IoU -0.05. */
TEST(Unit_DetectBoxes, ADegenerateBoxStaysFinite)
{
    const std::vector<float> point{1.0f, 1.0f, 1.0f, 1.0f};
    const std::vector<float> box{0.0f, 0.0f, 2.0f, 2.0f};

    // The point's width over its height is 0 / eps = 0, so the angle gap is atan(1) = pi / 4.
    const double pi = std::acos(-1.0);
    const double aspect = 4.0 / (pi * pi) * (pi / 4.0) * (pi / 4.0);
    const double trade_off = aspect / (1.0 - 0.0 + aspect);
    const float iou = SushiAI::Vision::Detect::intersection_over_union(point, box);
    const float complete = SushiAI::Vision::Detect::complete_iou(point, box);

    EXPECT_NEAR(iou, 0.0f, K_IOU_TOLERANCE);
    EXPECT_NEAR(complete, 0.0 - aspect * trade_off, K_IOU_TOLERANCE);
    EXPECT_TRUE(std::isfinite(SushiAI::Vision::Detect::complete_iou(point, point)));
    EXPECT_NEAR(SushiAI::Vision::Detect::complete_iou(point, point), 0.0f, K_IOU_TOLERANCE);
}

/** @brief Tests that the elementwise forms pair row i with row i. */
TEST(Unit_DetectBoxes, ElementwiseFormsPairRows)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> second{1.0f, 1.0f, 3.0f, 3.0f, 2.0f, 0.0f, 3.0f, 1.0f};
    std::vector<float> ious(2);
    std::vector<float> completes(2);
    SushiAI::Vision::Detect::elementwise_iou(first, second, ious);
    SushiAI::Vision::Detect::elementwise_complete_iou(first, second, completes);

    const std::vector<float> expected_ious{1.0f / 7.0f, 0.0f};
    const std::vector<float> expected_completes{1.0f / 7.0f - 2.0f / 18.0f, -4.0f / 10.0f};
    EXPECT_TRUE(near_reference(ious, expected_ious, K_IOU_TOLERANCE));
    EXPECT_TRUE(near_reference(completes, expected_completes, K_IOU_TOLERANCE));
}

/** @brief Tests that the pairwise form fills row n, column m with IoU of box n and box m. */
TEST(Unit_DetectBoxes, PairwiseFormFillsEveryPair)
{
    const std::vector<float> first{0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    const std::vector<float> second{0.0f, 0.0f, 2.0f, 2.0f, 1.0f, 1.0f, 3.0f, 3.0f,
                                    5.0f, 5.0f, 6.0f, 6.0f};
    std::vector<float> ious(6);
    SushiAI::Vision::Detect::pairwise_iou(first, second, ious);

    const std::vector<float> expected{4.0f / (4.0f + 4.0f - 4.0f), 1.0f / (4.0f + 4.0f - 1.0f),
                                      0.0f, 1.0f / (1.0f + 4.0f - 1.0f), 0.0f, 0.0f};
    EXPECT_TRUE(near_reference(ious, expected, K_IOU_TOLERANCE));
}

/** @brief Tests that an elementwise IoU over arrays of different length is refused. */
TEST(Unit_DetectBoxes, RejectsElementwiseInputsOfDifferentLength)
{
    const std::vector<float> first(8, 0.0f);
    const std::vector<float> second(4, 0.0f);
    std::vector<float> ious(2);
    SA_EXPECT_ERROR(SushiAI::Vision::Detect::elementwise_iou(first, second, ious),
                    "got 8, 4 and 2");
}
