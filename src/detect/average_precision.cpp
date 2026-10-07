/**************************************************************************/
/* average_precision.cpp                                                  */
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
 * @file average_precision.cpp
 * @brief Implements detection matching, the 101-point average precision and the metric table.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#include <SushiAI/detect/average_precision.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <SushiAI/core/common.hpp>
#include <SushiAI/detect/box_format.hpp>
#include <SushiAI/detect/detection_row.hpp>
#include <SushiAI/detect/iou.hpp>

namespace SushiAI
{
    namespace Detect
    {
        namespace
        {
            /** @brief Holds the width of the box filter over the mean F1 curve. */
            constexpr std::size_t K_SMOOTH_WIDTH = 101;

            /** @brief Holds the lanes of numpy's unrolled summation. */
            constexpr std::size_t K_SUM_LANES = 8;

            /** @brief Holds the longest run numpy sums without splitting it in two. */
            constexpr std::size_t K_SUM_BLOCK = 128;

            /** @brief Holds the first float32 that no longer separates neighbouring indices. */
            constexpr float K_CLASS_LIMIT = 16777216.0f;

            /** @brief Holds a class index and how many times it occurs. */
            struct LabelCount
            {
                std::int64_t label = 0; /**< Class index. */
                std::int64_t count = 0; /**< Occurrences. */
            };

            /** @brief Holds what the curves of one class are computed from. */
            struct ClassRun
            {
                const std::size_t* ranked = nullptr;    /**< Its detections, best score first. */
                std::size_t count = 0;                  /**< Detections of the class; not zero. */
                std::int64_t instances = 0;             /**< Its ground-truth boxes; not zero. */
                const float* scores = nullptr;          /**< Score of every detection. */
                const std::uint8_t* correct = nullptr;  /**< Match flags of every detection. */
            };

            /** @brief Returns point @p index of `numpy.linspace(0, 1, points)`. */
            double grid_point(std::size_t index, std::size_t points) noexcept
            {
                if (index + 1 == points)
                    return 1.0;
                return static_cast<double>(index) * (1.0 / static_cast<double>(points - 1));
            }

            /** @brief Returns the sum of @p count values in the order `numpy.add.reduce` adds. */
            double sum_pairwise(const double* values, std::size_t count) noexcept
            {
                if (count < K_SUM_LANES)
                    return std::accumulate(values, values + count, 0.0);

                if (count > K_SUM_BLOCK)
                {
                    std::size_t half = count / 2;
                    half -= half % K_SUM_LANES;
                    return sum_pairwise(values, half) + sum_pairwise(values + half, count - half);
                }

                double lanes[K_SUM_LANES];
                std::copy(values, values + K_SUM_LANES, lanes);
                std::size_t at = K_SUM_LANES;
                for (; at < count - count % K_SUM_LANES; at += K_SUM_LANES)
                {
                    for (std::size_t lane = 0; lane < K_SUM_LANES; ++lane)
                        lanes[lane] += values[at + lane];
                }

                double sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                             ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
                for (; at < count; ++at)
                    sum += values[at];
                return sum;
            }

            /**
             * @brief Returns `numpy.interp(x, xs, ys, left, right)` for one point.
             * @pre @p xs is not empty, never decreases, and is as long as @p ys.
             */
            double interpolate(double x,
                               const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               double left,
                               double right) noexcept
            {
                if (x < xs.front())
                    return left;
                if (x > xs.back())
                    return right;

                // upper_bound gives the last index at or below x, which numpy.interp reads.
                const std::size_t at =
                    static_cast<std::size_t>(std::upper_bound(xs.begin(), xs.end(), x) -
                                             xs.begin()) - 1;
                if (at + 1 == xs.size() || xs[at] == x)
                    return ys[at];

                const double slope = (ys[at + 1] - ys[at]) / (xs[at + 1] - xs[at]);
                return slope * (x - xs[at]) + ys[at];
            }

            /** @brief Returns the class index a detection row stores as a float. */
            std::int64_t read_class(const float* row) noexcept
            {
                return static_cast<std::int64_t>(row[K_DETECTION_CLASS]);
            }

            /** @brief Throws unless one image's detections and ground truth are well formed. */
            void check_image(SushiRuntime::span<const float> detections,
                             SushiRuntime::span<const float> truth_boxes,
                             SushiRuntime::span<const std::int64_t> truth_classes)
            {
                SA_THROW_IF(detections.size() % K_DETECTION_VALUES != 0,
                            "detections need [N, {}] values; got {}", K_DETECTION_VALUES,
                            detections.size());
                SA_THROW_IF(truth_boxes.size() != truth_classes.size() * K_BOX_VALUES,
                            "ground truth needs [M, 4] boxes and [M] classes; got {} values "
                            "and {} classes",
                            truth_boxes.size(), truth_classes.size());

                for (std::size_t box = 0; box < truth_classes.size(); ++box)
                {
                    SA_THROW_IF(truth_classes[box] < 0,
                                "ground-truth box {} has the negative class {}", box,
                                truth_classes[box]);
                }

                for (std::size_t at = 0; at < detections.size(); at += K_DETECTION_VALUES)
                {
                    const float score = detections[at + K_DETECTION_SCORE];
                    const float label = detections[at + K_DETECTION_CLASS];
                    SA_THROW_IF(!std::isfinite(score), "detection {} has the score {}",
                                at / K_DETECTION_VALUES, score);
                    SA_THROW_IF(!(label >= 0.0f) || label >= K_CLASS_LIMIT ||
                                    label != std::floor(label),
                                "detection {} has the class {}, which is not a class index",
                                at / K_DETECTION_VALUES, label);
                }
            }

            /**
             * @brief Writes the `[N, K_IOU_THRESHOLD_COUNT]` match flags of one checked image.
             * @pre check_image() accepted the inputs and @p correct has room for the flags.
             */
            void mark_matches(SushiRuntime::span<const float> detections,
                              SushiRuntime::span<const float> truth_boxes,
                              SushiRuntime::span<const std::int64_t> truth_classes,
                              std::uint8_t* correct)
            {
                const std::size_t count = detections.size() / K_DETECTION_VALUES;
                const std::size_t truths = truth_classes.size();
                std::fill(correct, correct + count * K_IOU_THRESHOLD_COUNT, std::uint8_t{0});
                if (count == 0 || truths == 0)
                    return;

                std::vector<float> boxes(count * K_BOX_VALUES);
                for (std::size_t row = 0; row < count; ++row)
                {
                    const float* from = detections.data() + row * K_DETECTION_VALUES;
                    std::copy(from, from + K_BOX_VALUES, boxes.begin() + row * K_BOX_VALUES);
                }
                std::vector<float> ious(truths * count);
                pairwise_iou(truth_boxes, boxes, ious);

                std::vector<std::size_t> best_truth(count, truths);
                std::vector<float> best_iou(count, 0.0f);
                for (std::size_t row = 0; row < count; ++row)
                {
                    const std::int64_t label = read_class(detections.data() +
                                                          row * K_DETECTION_VALUES);
                    for (std::size_t truth = 0; truth < truths; ++truth)
                    {
                        // A NaN IoU fails >= and so matches nothing, as in the reference.
                        const float iou = ious[truth * count + row];
                        if (truth_classes[truth] != label || !(iou >= K_IOU_THRESHOLDS[0]) ||
                            iou < best_iou[row])
                            continue;
                        best_truth[row] = truth;
                        best_iou[row] = iou;
                    }
                }

                std::vector<std::uint8_t> taken(truths);
                for (std::size_t level = 0; level < K_IOU_THRESHOLD_COUNT; ++level)
                {
                    std::fill(taken.begin(), taken.end(), std::uint8_t{0});
                    for (std::size_t row = 0; row < count; ++row)
                    {
                        const std::size_t truth = best_truth[row];
                        if (truth == truths || !(best_iou[row] >= K_IOU_THRESHOLDS[level]) ||
                            taken[truth] != 0)
                            continue;
                        taken[truth] = 1;
                        correct[row * K_IOU_THRESHOLD_COUNT + level] = 1;
                    }
                }
            }

            /** @brief Returns each distinct value of @p labels with its count, ascending. */
            std::vector<LabelCount> count_labels(std::vector<std::int64_t> labels)
            {
                std::sort(labels.begin(), labels.end());
                std::vector<LabelCount> counts;
                for (const std::int64_t label : labels)
                {
                    if (counts.empty() || counts.back().label != label)
                        counts.push_back(LabelCount{label, 0});
                    ++counts.back().count;
                }
                return counts;
            }

            /** @brief Returns the count @p counts holds for @p label, or zero. */
            std::int64_t find_count(const std::vector<LabelCount>& counts,
                                    std::int64_t label) noexcept
            {
                const auto at = std::lower_bound(counts.begin(), counts.end(), label,
                                                 [](const LabelCount& entry, std::int64_t wanted)
                                                 {
                                                     return entry.label < wanted;
                                                 });
                if (at == counts.end() || at->label != label)
                    return 0;
                return at->count;
            }

            /**
             * @brief Returns the detection indices by ascending class, then descending score.
             * @return Detections of equal class and score keep the order they were added in.
             */
            std::vector<std::size_t> rank_detections(const std::vector<float>& scores,
                                                     const std::vector<std::int64_t>& classes)
            {
                std::vector<std::size_t> ranked(scores.size());
                std::iota(ranked.begin(), ranked.end(), std::size_t{0});
                std::stable_sort(ranked.begin(), ranked.end(),
                                 [&scores, &classes](std::size_t left, std::size_t right) noexcept
                                 {
                                     if (classes[left] != classes[right])
                                         return classes[left] < classes[right];
                                     return scores[left] > scores[right];
                                 });
                return ranked;
            }

            /**
             * @brief Writes one class's AP at each IoU and its curves at IoU 0.5.
             * @param ap Receives `[K_IOU_THRESHOLD_COUNT]`.
             * @param precision_curve Receives `[K_CONFIDENCE_POINTS]`.
             * @param recall_curve Receives `[K_CONFIDENCE_POINTS]`.
             */
            void evaluate_class(const ClassRun& run,
                                double* ap,
                                double* precision_curve,
                                double* recall_curve)
            {
                std::vector<double> recall(run.count);
                std::vector<double> precision(run.count);
                const double labels = static_cast<double>(run.instances) + K_METRIC_EPS;

                // Level 0 runs last, so the curves below read the IoU 0.5 recall and precision.
                for (std::size_t level = K_IOU_THRESHOLD_COUNT; level-- > 0;)
                {
                    std::int64_t hits = 0;
                    for (std::size_t rank = 0; rank < run.count; ++rank)
                    {
                        hits += run.correct[run.ranked[rank] * K_IOU_THRESHOLD_COUNT + level];
                        recall[rank] = static_cast<double>(hits) / labels;
                        precision[rank] =
                            static_cast<double>(hits) / static_cast<double>(rank + 1);
                    }
                    ap[level] = integrate_precision(recall, precision);
                }

                std::vector<double> negated_scores(run.count);
                for (std::size_t rank = 0; rank < run.count; ++rank)
                    negated_scores[rank] = -static_cast<double>(run.scores[run.ranked[rank]]);

                for (std::size_t point = 0; point < K_CONFIDENCE_POINTS; ++point)
                {
                    const double x = -grid_point(point, K_CONFIDENCE_POINTS);
                    recall_curve[point] =
                        interpolate(x, negated_scores, recall, 0.0, recall.back());
                    precision_curve[point] =
                        interpolate(x, negated_scores, precision, 1.0, precision.back());
                }
            }

            /**
             * @brief Returns the confidence point where the smoothed mean F1 is largest.
             * @param precision_curves `[rows, K_CONFIDENCE_POINTS]`.
             * @param recall_curves `[rows, K_CONFIDENCE_POINTS]`.
             * @pre @p rows is not zero.
             */
            std::size_t find_best_confidence(const std::vector<double>& precision_curves,
                                             const std::vector<double>& recall_curves,
                                             std::size_t rows)
            {
                std::vector<double> mean_f1(K_CONFIDENCE_POINTS);
                for (std::size_t point = 0; point < K_CONFIDENCE_POINTS; ++point)
                {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < rows; ++row)
                    {
                        const double p = precision_curves[row * K_CONFIDENCE_POINTS + point];
                        const double r = recall_curves[row * K_CONFIDENCE_POINTS + point];
                        sum += 2.0 * p * r / (p + r + K_METRIC_EPS);
                    }
                    mean_f1[point] = sum / static_cast<double>(rows);
                }

                const std::size_t pad = K_SMOOTH_WIDTH / 2;
                const double weight = 1.0 / static_cast<double>(K_SMOOTH_WIDTH);
                std::size_t best = 0;
                double best_value = 0.0;
                for (std::size_t point = 0; point < K_CONFIDENCE_POINTS; ++point)
                {
                    double smoothed = 0.0;
                    for (std::size_t tap = 0; tap < K_SMOOTH_WIDTH; ++tap)
                    {
                        // The curve is padded with its first and last value, pad points each.
                        const std::size_t shifted = std::max(point + tap, pad) - pad;
                        smoothed += mean_f1[std::min(shifted, K_CONFIDENCE_POINTS - 1)] * weight;
                    }
                    if (point == 0 || smoothed > best_value)
                    {
                        best = point;
                        best_value = smoothed;
                    }
                }
                return best;
            }

            /** @brief Returns the mean of @p values as `numpy.mean` computes it. */
            double mean_of(const std::vector<double>& values) noexcept
            {
                return sum_pairwise(values.data(), values.size()) /
                       static_cast<double>(values.size());
            }
        } // namespace

        void match_detections(SushiRuntime::span<const float> detections,
                              SushiRuntime::span<const float> truth_boxes,
                              SushiRuntime::span<const std::int64_t> truth_classes,
                              SushiRuntime::span<std::uint8_t> correct)
        {
            check_image(detections, truth_boxes, truth_classes);
            const std::size_t count = detections.size() / K_DETECTION_VALUES;
            SA_THROW_IF(correct.size() != count * K_IOU_THRESHOLD_COUNT,
                        "the match flags of {} detections need {} values; got {}", count,
                        count * K_IOU_THRESHOLD_COUNT, correct.size());
            mark_matches(detections, truth_boxes, truth_classes, correct.data());
        }

        double integrate_precision(SushiRuntime::span<const double> recall,
                                   SushiRuntime::span<const double> precision)
        {
            SA_THROW_IF(recall.size() != precision.size(),
                        "a precision curve needs one value per recall value; got {} and {}",
                        precision.size(), recall.size());

            const std::size_t count = recall.size();
            std::vector<double> envelope_recall(count + 3);
            std::vector<double> envelope(count + 3);
            envelope_recall[0] = 0.0;
            envelope[0] = 1.0;
            std::copy(recall.begin(), recall.end(), envelope_recall.begin() + 1);
            std::copy(precision.begin(), precision.end(), envelope.begin() + 1);
            envelope_recall[count + 1] = count == 0 ? 1.0 : recall[count - 1];
            envelope_recall[count + 2] = 1.0;
            envelope[count + 1] = 0.0;
            envelope[count + 2] = 0.0;

            for (std::size_t at = count + 2; at-- > 0;)
                envelope[at] = std::max(envelope[at], envelope[at + 1]);

            double strips[K_RECALL_POINTS - 1];
            double previous = interpolate(0.0, envelope_recall, envelope, 1.0, 0.0);
            for (std::size_t point = 1; point < K_RECALL_POINTS; ++point)
            {
                const double width =
                    grid_point(point, K_RECALL_POINTS) - grid_point(point - 1, K_RECALL_POINTS);
                const double next = interpolate(grid_point(point, K_RECALL_POINTS),
                                                envelope_recall, envelope, 1.0, 0.0);
                strips[point - 1] = width * (next + previous) / 2.0;
                previous = next;
            }
            return sum_pairwise(strips, K_RECALL_POINTS - 1);
        }

        void AveragePrecision::add_image(SushiRuntime::span<const float> detections,
                                         SushiRuntime::span<const float> truth_boxes,
                                         SushiRuntime::span<const std::int64_t> truth_classes)
        {
            check_image(detections, truth_boxes, truth_classes);

            const std::size_t count = detections.size() / K_DETECTION_VALUES;
            const std::size_t held = scores_.size();
            scores_.reserve(held + count);
            classes_.reserve(held + count);
            correct_.resize((held + count) * K_IOU_THRESHOLD_COUNT);
            mark_matches(detections, truth_boxes, truth_classes,
                         correct_.data() + held * K_IOU_THRESHOLD_COUNT);

            for (std::size_t row = 0; row < count; ++row)
            {
                const float* values = detections.data() + row * K_DETECTION_VALUES;
                scores_.push_back(values[K_DETECTION_SCORE]);
                classes_.push_back(read_class(values));
            }

            truth_classes_.insert(truth_classes_.end(), truth_classes.begin(),
                                  truth_classes.end());
            for (const LabelCount& entry :
                 count_labels(std::vector<std::int64_t>(truth_classes.begin(),
                                                        truth_classes.end())))
                image_classes_.push_back(entry.label);
            ++images_;
        }

        MetricsSummary AveragePrecision::summarize() const
        {
            MetricsSummary summary;
            summary.images = images_;
            summary.instances = static_cast<std::int64_t>(truth_classes_.size());

            const std::vector<LabelCount> instances = count_labels(truth_classes_);
            const std::size_t rows = instances.size();
            if (rows == 0)
                return summary;

            const std::vector<LabelCount> images = count_labels(image_classes_);
            const std::vector<std::size_t> ranked = rank_detections(scores_, classes_);
            std::vector<double> ap(rows * K_IOU_THRESHOLD_COUNT, 0.0);
            std::vector<double> precision_curves(rows * K_CONFIDENCE_POINTS, 0.0);
            std::vector<double> recall_curves(rows * K_CONFIDENCE_POINTS, 0.0);

            std::size_t cursor = 0;
            for (std::size_t row = 0; row < rows; ++row)
            {
                const std::int64_t label = instances[row].label;
                while (cursor < ranked.size() && classes_[ranked[cursor]] < label)
                    ++cursor;
                const std::size_t begin = cursor;
                while (cursor < ranked.size() && classes_[ranked[cursor]] == label)
                    ++cursor;
                if (cursor == begin)
                    continue;

                const ClassRun run{ranked.data() + begin, cursor - begin, instances[row].count,
                                   scores_.data(), correct_.data()};
                evaluate_class(run, ap.data() + row * K_IOU_THRESHOLD_COUNT,
                               precision_curves.data() + row * K_CONFIDENCE_POINTS,
                               recall_curves.data() + row * K_CONFIDENCE_POINTS);
            }

            const std::size_t best = find_best_confidence(precision_curves, recall_curves, rows);
            std::vector<double> precisions(rows);
            std::vector<double> recalls(rows);
            std::vector<double> ap50s(rows);
            summary.classes.resize(rows);
            for (std::size_t row = 0; row < rows; ++row)
            {
                const double* row_ap = ap.data() + row * K_IOU_THRESHOLD_COUNT;
                precisions[row] = precision_curves[row * K_CONFIDENCE_POINTS + best];
                recalls[row] = recall_curves[row * K_CONFIDENCE_POINTS + best];
                ap50s[row] = row_ap[0];

                ClassMetrics& metrics = summary.classes[row];
                metrics.label = instances[row].label;
                metrics.images = find_count(images, metrics.label);
                metrics.instances = instances[row].count;
                metrics.precision = precisions[row];
                metrics.recall = recalls[row];
                metrics.ap50 = ap50s[row];
                metrics.ap50_95 = sum_pairwise(row_ap, K_IOU_THRESHOLD_COUNT) /
                                  static_cast<double>(K_IOU_THRESHOLD_COUNT);
            }

            summary.precision = mean_of(precisions);
            summary.recall = mean_of(recalls);
            summary.map50 = mean_of(ap50s);
            summary.map50_95 = mean_of(ap);
            return summary;
        }

        void AveragePrecision::clear() noexcept
        {
            images_ = 0;
            scores_.clear();
            classes_.clear();
            correct_.clear();
            truth_classes_.clear();
            image_classes_.clear();
        }

        std::int64_t AveragePrecision::image_count() const noexcept
        {
            return images_;
        }

    } // namespace Detect
} // namespace SushiAI
