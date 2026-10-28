/**************************************************************************/
/* average_precision.hpp                                                  */
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
 * @file average_precision.hpp
 * @brief Declares the detection metric: precision, recall and average precision per class.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /** @brief Holds the number of IoU thresholds a detection is matched at. */
        inline constexpr std::size_t K_IOU_THRESHOLD_COUNT = 10;

        /**
         * @brief Holds the float32 values of `torch.linspace(0.5, 0.95, 10)`.
         * @see U/models/yolo/detect/val.py:59
         */
        inline constexpr std::array<float, K_IOU_THRESHOLD_COUNT> K_IOU_THRESHOLDS{
            0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f};

        /** @brief Holds the recall points the precision envelope is integrated over. */
        inline constexpr std::size_t K_RECALL_POINTS = 101;

        /** @brief Holds the confidence points precision and recall are sampled at. */
        inline constexpr std::size_t K_CONFIDENCE_POINTS = 1000;

        /** @brief Holds the term the reference adds to a label count and to an F1 denominator. */
        inline constexpr double K_METRIC_EPS = 1e-16;

        /** @brief Holds one class's row of the validation table. */
        struct ClassMetrics
        {
            std::int64_t label = 0;     /**< Class index. */
            std::int64_t images = 0;    /**< Images that hold a box of the class. */
            std::int64_t instances = 0; /**< Ground-truth boxes of the class. */
            double precision = 0.0;     /**< Precision at the confidence of the best mean F1. */
            double recall = 0.0;        /**< Recall at the same confidence. */
            double ap50 = 0.0;          /**< Average precision at IoU 0.5. */
            double ap50_95 = 0.0;       /**< Mean of the average precision at the ten IoUs. */
        };

        /**
         * @brief Holds the validation table: one row per class with ground truth, and the means.
         * @see U/utils/metrics.py:963, 972, 981, 999
         */
        struct MetricsSummary
        {
            std::vector<ClassMetrics> classes; /**< Rows in ascending class index. */
            std::int64_t images = 0;           /**< Images accumulated. */
            std::int64_t instances = 0;        /**< Ground-truth boxes accumulated. */
            double precision = 0.0;            /**< Mean of the rows' precision. */
            double recall = 0.0;               /**< Mean of the rows' recall. */
            double map50 = 0.0;                /**< Mean of the rows' AP at IoU 0.5. */
            double map50_95 = 0.0;             /**< Mean of the AP over rows and IoUs. */
        };

        /**
         * @brief Marks, for each IoU threshold, the detections that match a ground-truth box.
         * @param detections `[N, 6]` rows; @p truth_boxes `[M, 4]`; @p truth_classes `[M]`.
         * @param correct Receives `[N, K_IOU_THRESHOLD_COUNT]`; 1 where the detection matched.
         * @return InvalidDescription for an array of the wrong extent, a negative class or a score
         *         that is not finite.
         * @see U/engine/validator.py:332-353, U/models/yolo/detect/val.py:391-394
         */
        [[nodiscard]] Status match_detections(SushiRuntime::span<const float> detections,
                                              SushiRuntime::span<const float> truth_boxes,
                                              SushiRuntime::span<const std::int64_t> truth_classes,
                                              SushiRuntime::span<std::uint8_t> correct);

        /**
         * @brief Returns the area under a precision envelope sampled at 101 recall points.
         * @param recall Recall after each detection in descending score; never decreases.
         * @param precision Precision after each detection.
         * @return The area, or InvalidDescription when the two curves differ in length.
         * @see U/utils/metrics.py:782-793
         */
        [[nodiscard]] Result<double> integrate_precision(
            SushiRuntime::span<const double> recall, SushiRuntime::span<const double> precision);

        /**
         * @brief Accumulates matched detections image by image and reports the validation table.
         * @see U/utils/metrics.py:843-893, U/models/yolo/detect/val.py:228-263
         */
        class AveragePrecision final
        {
            public:
                /**
                 * @brief Matches one image's detections to its ground truth and stores the result.
                 * @param detections `[N, 6]` rows; @p truth_boxes `[M, 4]`; @p truth_classes `[M]`.
                 * @return InvalidDescription for an array of the wrong extent, a negative class or
                 *         a bad score; nothing is stored then.
                 * @pre The detections passed the validation confidence and the per-image limit.
                 * @see U/models/yolo/detect/val.py:198, U/engine/validator.py:144-145
                 */
                [[nodiscard]] Status add_image(
                    SushiRuntime::span<const float> detections,
                    SushiRuntime::span<const float> truth_boxes,
                    SushiRuntime::span<const std::int64_t> truth_classes);

                /**
                 * @brief Computes the table over every image added since the last clear().
                 * @return No rows and zero means when no ground-truth box was added.
                 * @see U/utils/metrics.py:843-893
                 */
                [[nodiscard]] MetricsSummary summarize() const;

                /** @brief Drops every accumulated image. */
                void clear() noexcept;

                /** @brief Returns the number of images added since the last clear(). */
                [[nodiscard]] std::int64_t image_count() const noexcept;

            private:
                /// @brief Holds the number of images added.
                std::int64_t images_ = 0;

                /// @brief Holds the score of each accumulated detection, in the order added.
                std::vector<float> scores_;

                /// @brief Holds the class of each accumulated detection.
                std::vector<std::int64_t> classes_;

                /// @brief Holds the match flags, `[detections, K_IOU_THRESHOLD_COUNT]`.
                std::vector<std::uint8_t> correct_;

                /// @brief Holds the class of every accumulated ground-truth box.
                std::vector<std::int64_t> truth_classes_;

                /// @brief Holds each image's distinct ground-truth classes, image after image.
                std::vector<std::int64_t> image_classes_;
        };

    } // namespace Vision::Detect
} // namespace SushiAI
