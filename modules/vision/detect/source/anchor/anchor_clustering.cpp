/**************************************************************************/
/* anchor_clustering.cpp                                                  */
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
 * @file anchor_clustering.cpp
 * @brief Implements the derivation of anchors from the sizes of label boxes.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>

#include <SushiAI/core/check.hpp>

#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Holds the cluster index of a box no pass has assigned yet. */
            constexpr std::size_t K_UNASSIGNED = std::numeric_limits<std::size_t>::max();

            /** @brief Holds the share of a side the first refinement sweep moves it by. */
            constexpr double K_FIRST_REFINEMENT_STEP = 0.1;

            /** @brief Holds the step below which the refinement stops. */
            constexpr double K_SMALLEST_REFINEMENT_STEP = 0.001;

            /** @brief Names a side of an anchor. */
            enum class Side : std::uint8_t
            {
                Width,  /**< The width. */
                Height, /**< The height. */
            };

            /** @brief Holds the boxes a clustering uses and the number it left out. */
            struct UsableBoxes
            {
                std::vector<AnchorSize> boxes; /**< The usable boxes, in the caller's order. */
                std::int64_t rejected = 0;     /**< Boxes left out. */
            };

            /** @brief Holds the cluster of every box and its distance to that centre. */
            struct Partition
            {
                std::vector<std::size_t> cluster; /**< Per box, the index of its centre. */
                std::vector<double> distance;     /**< Per box, the distance to that centre. */
            };

            /** @brief Holds a fit measure and the sweeps the refinement made to reach it. */
            struct Refinement
            {
                double fit = 0.0;        /**< Mean best shape_iou after the last sweep. */
                std::int64_t sweeps = 0; /**< Sweeps made. */
            };

            /** @brief Returns the boxes whose sides are finite and above @p min_side. */
            UsableBoxes keep_usable_boxes(SushiRuntime::span<const AnchorSize> boxes,
                                          float min_side)
            {
                UsableBoxes usable;
                usable.boxes.reserve(boxes.size());
                for (std::size_t index = 0; index < boxes.size(); ++index)
                {
                    const AnchorSize box = boxes[index];
                    if (std::isfinite(box.width) && std::isfinite(box.height) &&
                        box.width > min_side && box.height > min_side)
                        usable.boxes.push_back(box);
                    else
                        ++usable.rejected;
                }
                return usable;
            }

            /**
             * @brief Returns the number of different sizes among @p boxes.
             * @param boxes Finite sizes; taken by value because the count sorts them.
             */
            std::size_t count_distinct_boxes(std::vector<AnchorSize> boxes)
            {
                std::sort(boxes.begin(), boxes.end(), [](AnchorSize a, AnchorSize b)
                {
                    return a.width < b.width || (a.width == b.width && a.height < b.height);
                });
                const auto last =
                    std::unique(boxes.begin(), boxes.end(), [](AnchorSize a, AnchorSize b)
                    {
                        return a.width == b.width && a.height == b.height;
                    });
                return static_cast<std::size_t>(last - boxes.begin());
            }

            /** @brief Returns the distance 1 - shape_iou of two sizes. */
            double measure_distance(AnchorSize a, AnchorSize b) noexcept
            {
                return 1.0 - static_cast<double>(shape_iou(a, b));
            }

            /**
             * @brief Returns the index of a box drawn in proportion to its weight.
             * @param weights One weight per box, none negative; not empty.
             * @param total The sum of @p weights in index order; 0 makes the draw even.
             */
            std::size_t draw_weighted_index(const std::vector<double>& weights, double total,
                                            Detail::SeededSequence& sequence) noexcept
            {
                if (!(total > 0.0))
                    return static_cast<std::size_t>(sequence.next() % weights.size());

                const double target = sequence.next_unit() * total;
                double cumulative = 0.0;
                std::size_t last_weighted = 0;
                for (std::size_t index = 0; index < weights.size(); ++index)
                {
                    if (!(weights[index] > 0.0))
                        continue;

                    cumulative += weights[index];
                    last_weighted = index;
                    if (cumulative > target)
                        return index;
                }
                // Rounding can leave the last cumulative sum at the target; that box takes it.
                return last_weighted;
            }

            /**
             * @brief Returns @p count first centres: one box drawn evenly, then boxes drawn far.
             * @pre @p boxes is not empty and @p count is at least 1.
             */
            std::vector<AnchorSize> seed_centres(const std::vector<AnchorSize>& boxes,
                                                 std::size_t count,
                                                 Detail::SeededSequence& sequence)
            {
                std::vector<AnchorSize> centres;
                centres.reserve(count);
                centres.push_back(
                    boxes[static_cast<std::size_t>(sequence.next() % boxes.size())]);

                // weights[i] is the squared distance of box i to the nearest centre so far.
                std::vector<double> weights(boxes.size(),
                                            std::numeric_limits<double>::infinity());
                while (centres.size() < count)
                {
                    const AnchorSize newest = centres.back();
                    double total = 0.0;
                    for (std::size_t index = 0; index < boxes.size(); ++index)
                    {
                        const double distance = measure_distance(boxes[index], newest);
                        weights[index] = std::min(weights[index], distance * distance);
                        total += weights[index];
                    }
                    centres.push_back(boxes[draw_weighted_index(weights, total, sequence)]);
                }
                return centres;
            }

            /**
             * @brief Gives every box to its nearest centre, the lower index on a tie.
             * @return Whether a box changed cluster.
             * @pre @p partition holds one entry per box and @p centres is not empty.
             */
            bool assign_boxes(const std::vector<AnchorSize>& boxes,
                              const std::vector<AnchorSize>& centres,
                              Partition& partition) noexcept
            {
                bool changed = false;
                for (std::size_t index = 0; index < boxes.size(); ++index)
                {
                    std::size_t nearest = 0;
                    double smallest = measure_distance(boxes[index], centres[0]);
                    for (std::size_t centre = 1; centre < centres.size(); ++centre)
                    {
                        const double distance = measure_distance(boxes[index], centres[centre]);
                        if (distance < smallest)
                        {
                            smallest = distance;
                            nearest = centre;
                        }
                    }
                    changed = changed || partition.cluster[index] != nearest;
                    partition.cluster[index] = nearest;
                    partition.distance[index] = smallest;
                }
                return changed;
            }

            /**
             * @brief Gives each cluster without a box the box farthest from its own centre.
             * @pre Every box is assigned and there are at least @p clusters boxes.
             */
            void fill_empty_clusters(Partition& partition, std::size_t clusters)
            {
                std::vector<std::size_t> sizes(clusters, 0);
                for (const std::size_t cluster : partition.cluster)
                    ++sizes[cluster];

                for (std::size_t empty = 0; empty < clusters; ++empty)
                {
                    if (sizes[empty] != 0)
                        continue;

                    // A box that is alone in its cluster stays, or that cluster would empty.
                    std::size_t farthest = K_UNASSIGNED;
                    for (std::size_t index = 0; index < partition.cluster.size(); ++index)
                    {
                        if (sizes[partition.cluster[index]] < 2)
                            continue;
                        if (farthest == K_UNASSIGNED ||
                            partition.distance[index] > partition.distance[farthest])
                            farthest = index;
                    }
                    SA_FATAL_IF(farthest == K_UNASSIGNED,
                                "an empty cluster found no box to take among {} boxes",
                                partition.cluster.size());

                    --sizes[partition.cluster[farthest]];
                    partition.cluster[farthest] = empty;
                    partition.distance[farthest] = 0.0;
                    sizes[empty] = 1;
                }
            }

            /**
             * @brief Returns the median of @p values, the mean of the middle two when even.
             * @pre @p values is not empty; its order is not kept.
             */
            double take_median(std::vector<double>& values) noexcept
            {
                const std::size_t middle = values.size() / 2;
                const auto position = values.begin() + static_cast<std::ptrdiff_t>(middle);

                std::nth_element(values.begin(), position, values.end());
                if (values.size() % 2 != 0)
                    return *position;

                return 0.5 * (*std::max_element(values.begin(), position) + *position);
            }

            /**
             * @brief Sets each centre to the median width and median height of its boxes.
             * @pre Every cluster of @p partition holds at least one box.
             */
            void move_centres_to_medians(const std::vector<AnchorSize>& boxes,
                                         const Partition& partition,
                                         std::vector<AnchorSize>& centres)
            {
                std::vector<std::vector<double>> widths(centres.size());
                std::vector<std::vector<double>> heights(centres.size());
                for (std::size_t index = 0; index < boxes.size(); ++index)
                {
                    const std::size_t cluster = partition.cluster[index];
                    widths[cluster].push_back(static_cast<double>(boxes[index].width));
                    heights[cluster].push_back(static_cast<double>(boxes[index].height));
                }
                for (std::size_t centre = 0; centre < centres.size(); ++centre)
                {
                    centres[centre].width = static_cast<float>(take_median(widths[centre]));
                    centres[centre].height = static_cast<float>(take_median(heights[centre]));
                }
            }

            /**
             * @brief Moves @p centres by assignment and median until no box changes cluster.
             * @return The number of assignment passes made, at most @p max_iterations.
             */
            std::int64_t settle_centres(const std::vector<AnchorSize>& boxes,
                                        std::vector<AnchorSize>& centres,
                                        std::int64_t max_iterations)
            {
                Partition partition;
                partition.cluster.assign(boxes.size(), K_UNASSIGNED);
                partition.distance.assign(boxes.size(), 0.0);

                std::int64_t iterations = 0;
                while (iterations < max_iterations)
                {
                    ++iterations;
                    if (!assign_boxes(boxes, centres, partition))
                        break;

                    fill_empty_clusters(partition, centres.size());
                    move_centres_to_medians(boxes, partition, centres);
                }
                return iterations;
            }

            /** @brief Returns the mean over @p boxes of the best shape_iou with any anchor. */
            double measure_mean_best_iou(const std::vector<AnchorSize>& boxes,
                                         const std::vector<AnchorSize>& anchors) noexcept
            {
                double sum = 0.0;
                for (const AnchorSize& box : boxes)
                {
                    float best = 0.0f;
                    for (const AnchorSize& anchor : anchors)
                        best = std::max(best, shape_iou(box, anchor));
                    sum += static_cast<double>(best);
                }
                return sum / static_cast<double>(boxes.size());
            }

            /** @brief Returns @p anchor with @p side multiplied by @p factor. */
            AnchorSize scale_side(AnchorSize anchor, Side side, double factor) noexcept
            {
                float& length = side == Side::Width ? anchor.width : anchor.height;
                length = static_cast<float>(static_cast<double>(length) * factor);
                return anchor;
            }

            /** @brief Writes, per box, the best shape_iou with every anchor but @p excluded. */
            void find_best_other(const std::vector<AnchorSize>& boxes,
                                 const std::vector<AnchorSize>& anchors, std::size_t excluded,
                                 std::vector<float>& best_other) noexcept
            {
                for (std::size_t index = 0; index < boxes.size(); ++index)
                {
                    float best = 0.0f;
                    for (std::size_t anchor = 0; anchor < anchors.size(); ++anchor)
                    {
                        if (anchor != excluded)
                            best = std::max(best, shape_iou(boxes[index], anchors[anchor]));
                    }
                    best_other[index] = best;
                }
            }

            /**
             * @brief Returns the mean best shape_iou when @p candidate joins the other anchors.
             * @param best_other Per box, the best shape_iou with every other anchor.
             */
            double measure_fit_with(const std::vector<AnchorSize>& boxes,
                                    const std::vector<float>& best_other,
                                    AnchorSize candidate) noexcept
            {
                double sum = 0.0;
                for (std::size_t index = 0; index < boxes.size(); ++index)
                {
                    sum += static_cast<double>(
                        std::max(best_other[index], shape_iou(boxes[index], candidate)));
                }
                return sum / static_cast<double>(boxes.size());
            }

            /**
             * @brief Tries one side of one anchor a step larger and smaller and keeps a gain.
             * @param fit The fit with @p anchor as it stands; replaced when a change is kept.
             * @return Whether a change was kept.
             */
            bool refine_side(const std::vector<AnchorSize>& boxes,
                             const std::vector<float>& best_other, Side side, double step,
                             AnchorSize& anchor, double& fit) noexcept
            {
                const AnchorSize larger = scale_side(anchor, side, 1.0 + step);
                const AnchorSize smaller = scale_side(anchor, side, 1.0 - step);
                const double fit_larger = measure_fit_with(boxes, best_other, larger);
                const double fit_smaller = measure_fit_with(boxes, best_other, smaller);

                const bool take_larger = fit_larger >= fit_smaller;
                const double best_fit = take_larger ? fit_larger : fit_smaller;
                if (!(best_fit > fit))
                    return false;

                anchor = take_larger ? larger : smaller;
                fit = best_fit;
                return true;
            }

            /**
             * @brief Searches each side of each anchor for a larger mean best shape_iou.
             * @param fit The mean best shape_iou of @p anchors on entry.
             * @return The fit reached, never below @p fit, and the sweeps made.
             */
            Refinement refine_anchors(const std::vector<AnchorSize>& boxes,
                                      std::vector<AnchorSize>& anchors, double fit,
                                      std::int64_t max_sweeps)
            {
                Refinement refinement;
                refinement.fit = fit;

                std::vector<float> best_other(boxes.size(), 0.0f);
                double step = K_FIRST_REFINEMENT_STEP;
                while (refinement.sweeps < max_sweeps && step >= K_SMALLEST_REFINEMENT_STEP)
                {
                    bool gained = false;
                    for (std::size_t index = 0; index < anchors.size(); ++index)
                    {
                        find_best_other(boxes, anchors, index, best_other);
                        for (const Side side : {Side::Width, Side::Height})
                        {
                            if (refine_side(boxes, best_other, side, step, anchors[index],
                                            refinement.fit))
                                gained = true;
                        }
                    }
                    ++refinement.sweeps;
                    if (!gained)
                        step *= 0.5;
                }
                return refinement;
            }

            /** @brief Orders @p anchors by ascending area, keeping the order of equal areas. */
            void sort_by_area(std::vector<AnchorSize>& anchors)
            {
                std::stable_sort(anchors.begin(), anchors.end(), [](AnchorSize a, AnchorSize b)
                {
                    return static_cast<double>(a.width) * static_cast<double>(a.height) <
                           static_cast<double>(b.width) * static_cast<double>(b.height);
                });
            }
        } // namespace

        Result<AnchorClustering> cluster_anchors(SushiRuntime::span<const AnchorSize> boxes,
                                                 const ClusteringOptions& options)
        {
            SA_FAIL_IF(options.count < 1, StatusCode::InvalidDescription,
                       "count must be at least 1; got {}", options.count);
            SA_FAIL_IF(options.max_iterations < 1, StatusCode::InvalidDescription,
                       "max_iterations must be at least 1; got {}", options.max_iterations);
            SA_FAIL_IF(options.max_refinement_sweeps < 0, StatusCode::InvalidDescription,
                       "max_refinement_sweeps must not be negative; got {}",
                       options.max_refinement_sweeps);
            SA_FAIL_IF(!(std::isfinite(options.min_side) && options.min_side >= 0.0f),
                       StatusCode::InvalidDescription,
                       "min_side must be finite and not negative; got {}", options.min_side);

            const UsableBoxes usable = keep_usable_boxes(boxes, options.min_side);
            const std::size_t count = static_cast<std::size_t>(options.count);
            const std::size_t distinct = count_distinct_boxes(usable.boxes);
            SA_FAIL_IF(distinct < count, StatusCode::Incompatible,
                       "{} anchors need as many distinct boxes; of {} boxes {} are usable and "
                       "{} of those are distinct",
                       options.count, boxes.size(), usable.boxes.size(), distinct);

            Detail::SeededSequence sequence(options.seed);
            AnchorClustering clustering;
            clustering.anchors = seed_centres(usable.boxes, count, sequence);
            clustering.iterations =
                settle_centres(usable.boxes, clustering.anchors, options.max_iterations);
            clustering.mean_best_iou_before_refinement =
                measure_mean_best_iou(usable.boxes, clustering.anchors);

            const Refinement refinement =
                refine_anchors(usable.boxes, clustering.anchors,
                               clustering.mean_best_iou_before_refinement,
                               options.max_refinement_sweeps);
            clustering.mean_best_iou = refinement.fit;
            clustering.refinement_sweeps = refinement.sweeps;
            clustering.boxes_used = static_cast<std::int64_t>(usable.boxes.size());
            clustering.boxes_rejected = usable.rejected;

            sort_by_area(clustering.anchors);
            return clustering;
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
