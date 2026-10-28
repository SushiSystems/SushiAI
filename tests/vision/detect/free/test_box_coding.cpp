/**************************************************************************/
/* test_box_coding.cpp                                                    */
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
 * @file test_box_coding.cpp
 * @brief Tests box coding against hand cases and the reference decode.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/reference_files/reference_file.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/iou.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::FeatureLevel;

    using SushiAI::Test::near_reference;

    using SushiAI::Test::ReferenceFile;

    using SushiAI::Test::ReferenceTensor;

    /** @brief Holds the tolerance of hand cases whose values float32 represents exactly. */
    constexpr float K_EXACT = 0.0f;

    /** @brief Holds the tolerance of a decode in pixels: two float32 steps at 640. */
    constexpr float K_DECODE_TOLERANCE = 1.3e-4f;

    /** @brief Holds the tolerance of decode then encode in grid units: two steps at 80. */
    constexpr float K_ROUND_TRIP_TOLERANCE = 1.6e-5f;

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
        SA_EXPECT_OK(SushiAI::Vision::Detect::Free::make_anchors(K_REFERENCE_LEVELS, grid.points,
                                                                 grid.strides));
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
            SA_EXPECT_OK(SushiAI::Vision::Detect::Free::join_level(raw.floats(),
                                                                   {1, 8400, 4, offset, cells},
                                                                   joined));
            offset += cells;
        }
        return {};
    }

} // namespace

/** @brief Tests that decoding subtracts left and top from the point and adds right and bottom. */
TEST(Medium_DetectBoxes, DecodesDistancesToCorners)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{1.5f - 0.5f, 2.5f - 1.0f, 1.5f + 1.5f, 2.5f + 2.0f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that the centre-size decode is the corner decode's midpoint and extent. */
TEST(Medium_DetectBoxes, DecodesDistancesToCentreAndSize)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::Free::decode_xywh(distances, points, boxes);

    const std::vector<float> expected{(1.0f + 3.0f) / 2.0f, (1.5f + 4.5f) / 2.0f, 3.0f - 1.0f,
                                      4.5f - 1.5f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that the anchor points repeat for every image of a batch. */
TEST(Medium_DetectBoxes, RepeatsTheAnchorPointsForEachImage)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> distances{0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f,
                                       0.0f, 0.0f, 2.0f, 2.0f, 0.5f, 0.5f, 1.5f, 1.5f};
    std::vector<float> boxes(16);
    SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0.5f, 2.5f, 0.5f,
                                      0.5f, 0.5f, 2.5f, 2.5f, 1.0f, 0.0f, 3.0f, 2.0f};
    EXPECT_TRUE(near_reference(boxes, expected, K_EXACT));
}

/** @brief Tests that encoding a decoded box returns the distances it was decoded from. */
TEST(Medium_DetectBoxes, EncodeInvertsDecode)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f, 0.5f, 1.5f};
    const std::vector<float> distances{0.25f, 0.5f, 0.75f, 1.0f, 2.0f, 0.0f,
                                       0.5f,  3.0f, 1.5f,  1.5f, 0.0f, 4.0f};
    std::vector<float> boxes(12);
    std::vector<float> recovered(12);
    SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes);
    SushiAI::Vision::Detect::Free::encode_distances(points, boxes, recovered);

    EXPECT_TRUE(near_reference(recovered, distances, K_EXACT));
}

/** @brief Tests that scaling multiplies all four values of a box by its anchor's stride. */
TEST(Medium_DetectBoxes, ScalesEachBoxByItsAnchorStride)
{
    const std::vector<float> boxes{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> strides{8.0f, 16.0f};
    std::vector<float> scaled(8);
    SushiAI::Vision::Detect::Free::scale_boxes(boxes, strides, scaled);

    const std::vector<float> expected{8.0f, 16.0f, 24.0f, 32.0f, 16.0f, 32.0f, 48.0f, 64.0f};
    EXPECT_TRUE(near_reference(scaled, expected, K_EXACT));
}

/** @brief Tests that rows which do not cover the anchors a whole number of times are refused. */
TEST(Medium_DetectBoxes, RejectsRowsThatDoNotCoverTheAnchors)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> distances(12, 1.0f);
    std::vector<float> boxes(12);
    SA_EXPECT_FATAL(SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes),
                    "2 anchors", "12");
}

/** @brief Tests that decoding the reference's raw one-to-many boxes gives its xywh decode. */
TEST(Medium_DetectBoxes, ReproducesTheReferenceOneToManyDecode)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2many.decoded.boxes");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2many", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    SushiAI::Vision::Detect::Free::decode_xywh(raw, grid.points, boxes);
    SushiAI::Vision::Detect::Free::scale_boxes(boxes, grid.strides, boxes);

    EXPECT_TRUE(near_reference(boxes, reference.floats(), K_DECODE_TOLERANCE));
}

/** @brief Tests that decoding the reference's raw one-to-one boxes gives its xyxy decode. */
TEST(Medium_DetectBoxes, ReproducesTheReferenceOneToOneDecode)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    SA_READ_REFERENCE_OR_FAIL(reference, forward, "one2one.decoded.boxes");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2one", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    SushiAI::Vision::Detect::Free::decode_xyxy(raw, grid.points, boxes);
    SushiAI::Vision::Detect::Free::scale_boxes(boxes, grid.strides, boxes);

    EXPECT_TRUE(near_reference(boxes, reference.floats(), K_DECODE_TOLERANCE));
}

/** @brief Tests that encoding the decode of the reference's raw boxes returns the raw boxes. */
TEST(Medium_DetectBoxes, EncodeInvertsTheDecodeOfTheReferenceBoxes)
{
    SA_OPEN_REFERENCE_OR_SKIP(forward, "yolo26/forward.sacp");
    std::vector<float> raw;
    const std::string missing = join_raw_boxes(forward, "one2one", raw);
    ASSERT_TRUE(missing.empty()) << missing;

    const ReferenceGrid grid = make_reference_grid();
    std::vector<float> boxes(raw.size());
    std::vector<float> recovered(raw.size());
    SushiAI::Vision::Detect::Free::decode_xyxy(raw, grid.points, boxes);
    SushiAI::Vision::Detect::Free::encode_distances(grid.points, boxes, recovered);

    EXPECT_TRUE(near_reference(recovered, raw, K_ROUND_TRIP_TOLERANCE));
}
