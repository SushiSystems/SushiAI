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
#include <support/tolerance.hpp>

namespace
{
    using SushiAI::Vision::Detect::Free::FeatureLevel;

    using SushiAI::Test::near_reference;

    using SushiAI::Test::ReferenceFile;

    using SushiAI::Test::ReferenceTensor;

    /**
     * @brief Holds the absolute the tolerance of a decode in pixels: two float32 steps at 640: 4096
     *        times the float unit roundoff, the smallest power of two not tighter than the earlier
     *        1.3e-4f.
     */
    constexpr float K_DECODE_TOLERANCE =
        static_cast<float>(4096.0 * TestSupport::unit_roundoff<float>());

    /**
     * @brief Holds the absolute the tolerance of decode then encode in grid units: two steps at 80:
     *        512 times the float unit roundoff, the smallest power of two not tighter than the
     *        earlier 1.6e-5f.
     */
    constexpr float K_ROUND_TRIP_TOLERANCE =
        static_cast<float>(512.0 * TestSupport::unit_roundoff<float>());

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

    /** @brief Holds the anchor points and the box distances one round-trip row encodes back. */
    struct RoundTripInput
    {
        std::string missing;            /**< Why the input is absent, or empty. */
        std::vector<float> points;      /**< Anchor points, `[N, 2]`. */
        std::vector<float> distances;   /**< Distances, `[N, 4]`. */
    };

    /** @brief Names a round-trip row: its inputs and the largest error it allows. */
    struct RoundTripRow
    {
        const char* name;                 /**< Names the row. */
        RoundTripInput (*load)();         /**< Reads the inputs of the row. */
        float largest_error;              /**< The largest absolute error allowed. */
    };

    /** @brief Holds six hand anchors, as `[N, 2]`. */
    const std::vector<float> K_HAND_POINTS{0.5f, 0.5f, 1.5f, 0.5f, 0.5f, 1.5f};

    /** @brief Holds the distances of the hand anchors, as `[N, 4]`; float32 holds each exactly. */
    const std::vector<float> K_HAND_DISTANCES{0.25f, 0.5f,  0.75f, 1.0f, 2.0f, 0.0f,
                                              0.5f,  3.0f,  1.5f,  1.5f, 0.0f, 4.0f};

    /** @brief Returns the 8400 anchors of the reference grid and its raw one-to-one boxes. */
    RoundTripInput load_reference_input()
    {
        RoundTripInput input;
        ReferenceFile forward(SA_TEST_FIXTURE_DIR, "yolo26/forward.sacp");
        if (!forward.is_present())
        {
            input.missing = forward.reason();
            return input;
        }

        input.missing = join_raw_boxes(forward, "one2one", input.distances);
        input.points = make_reference_grid().points;
        return input;
    }

    /** @brief Holds the rows of the round trip: the reference's raw boxes. */
    const RoundTripRow K_ROUND_TRIP_ROWS[] = {
        {"ReferenceBoxes", &load_reference_input, K_ROUND_TRIP_TOLERANCE}};

    /** @brief Returns the largest absolute difference between @p actual and @p expected. */
    float largest_difference(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        float largest = 0.0f;
        for (std::size_t i = 0; i < expected.size(); ++i)
            largest = std::fmax(largest, std::fabs(actual[i] - expected[i]));

        return largest;
    }

    /** @brief Runs a round-trip row; skipped when the row's inputs are absent. */
    class Medium_DetectBoxesRoundTrip : public ::testing::TestWithParam<RoundTripRow>
    {
        protected:
            /** @brief Reads the inputs of the row, and skips the case naming what is missing. */
            void SetUp() override
            {
                input_ = GetParam().load();
                if (!input_.missing.empty())
                    GTEST_SKIP() << input_.missing;
            }

            /** @brief Holds the inputs of the row. */
            RoundTripInput input_;
    };

} // namespace

/** @brief Tests that decoding subtracts left and top from the point and adds right and bottom. */
TEST(Small_DetectBoxes, DecodesDistancesToCorners)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{1.5f - 0.5f, 2.5f - 1.0f, 1.5f + 1.5f, 2.5f + 2.0f};
    EXPECT_EQ(boxes, expected);
}

/** @brief Tests that the centre-size decode is the corner decode's midpoint and extent. */
TEST(Small_DetectBoxes, DecodesDistancesToCentreAndSize)
{
    const std::vector<float> points{1.5f, 2.5f};
    const std::vector<float> distances{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> boxes(4);
    SushiAI::Vision::Detect::Free::decode_xywh(distances, points, boxes);

    const std::vector<float> expected{(1.0f + 3.0f) / 2.0f, (1.5f + 4.5f) / 2.0f, 3.0f - 1.0f,
                                      4.5f - 1.5f};
    EXPECT_EQ(boxes, expected);
}

/** @brief Tests that the anchor points repeat for every image of a batch. */
TEST(Small_DetectBoxes, RepeatsTheAnchorPointsForEachImage)
{
    const std::vector<float> points{0.5f, 0.5f, 1.5f, 0.5f};
    const std::vector<float> distances{0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f,
                                       0.0f, 0.0f, 2.0f, 2.0f, 0.5f, 0.5f, 1.5f, 1.5f};
    std::vector<float> boxes(16);
    SushiAI::Vision::Detect::Free::decode_xyxy(distances, points, boxes);

    const std::vector<float> expected{0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0.5f, 2.5f, 0.5f,
                                      0.5f, 0.5f, 2.5f, 2.5f, 1.0f, 0.0f, 3.0f, 2.0f};
    EXPECT_EQ(boxes, expected);
}

/** @brief Tests that scaling multiplies all four values of a box by its anchor's stride. */
TEST(Small_DetectBoxes, ScalesEachBoxByItsAnchorStride)
{
    const std::vector<float> boxes{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> strides{8.0f, 16.0f};
    std::vector<float> scaled(8);
    SushiAI::Vision::Detect::Free::scale_boxes(boxes, strides, scaled);

    const std::vector<float> expected{8.0f, 16.0f, 24.0f, 32.0f, 16.0f, 32.0f, 48.0f, 64.0f};
    EXPECT_EQ(scaled, expected);
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

/** @brief Tests that encoding the boxes decoded from hand distances returns them exactly. */
TEST(Small_DetectBoxes, EncodeReturnsHandDistancesExactly)
{
    std::vector<float> boxes(K_HAND_DISTANCES.size());
    std::vector<float> recovered(K_HAND_DISTANCES.size());
    SushiAI::Vision::Detect::Free::decode_xyxy(K_HAND_DISTANCES, K_HAND_POINTS, boxes);
    SushiAI::Vision::Detect::Free::encode_distances(K_HAND_POINTS, boxes, recovered);

    EXPECT_EQ(recovered, K_HAND_DISTANCES);
}

/** @brief Tests that encoding a decoded reference box returns the distances it came from. */
TEST_P(Medium_DetectBoxesRoundTrip, EncodeInvertsDecode)
{
    std::vector<float> boxes(input_.distances.size());
    std::vector<float> recovered(input_.distances.size());
    SushiAI::Vision::Detect::Free::decode_xyxy(input_.distances, input_.points, boxes);
    SushiAI::Vision::Detect::Free::encode_distances(input_.points, boxes, recovered);

    EXPECT_LE(largest_difference(recovered, input_.distances), GetParam().largest_error);
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_DetectBoxesRoundTrip,
                         ::testing::ValuesIn(K_ROUND_TRIP_ROWS),
                         [](const ::testing::TestParamInfo<RoundTripRow>& info)
                         {
                             return std::string(info.param.name);
                         });
