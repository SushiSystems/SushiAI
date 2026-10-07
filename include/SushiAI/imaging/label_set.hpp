/**************************************************************************/
/* label_set.hpp                                                          */
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
 * @file label_set.hpp
 * @brief Declares the labels of one image and the rules that apply to all of them at once.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <SushiAI/imaging/boxes.hpp>
#include <SushiAI/imaging/image.hpp>
#include <SushiAI/imaging/polygon.hpp>

/** @brief Holds the units of the imaging module that no caller names. */
namespace SushiAI::Imaging::Detail
{
    /** @brief Throws Error unless a label set has one box and one polygon entry per class. */
    void check_label_counts(std::size_t classes, std::size_t boxes, std::size_t polygons);
} // namespace SushiAI::Imaging::Detail

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /**
     * @brief Holds the labels of one image: a class, a box and a polygon per object.
     *
     * The three lists always have the same length, so index i names one object in all of
     * them. An object without a polygon has an entry of no vertex. No member lets a caller
     * change one list's length alone.
     * @tparam Box NormalizedBox or PixelBox; the polygons are in the same units as the boxes.
     */
    template <typename Box>
    class LabelSet final
    {
        public:
            /** @brief Constructs a set of no object. */
            LabelSet() = default;

            /**
             * @brief Constructs a set of objects without polygons.
             * @param boxes One per class; otherwise throws Error.
             */
            LabelSet(std::vector<std::int32_t> classes, std::vector<Box> boxes)
                : classes_(std::move(classes)), boxes_(std::move(boxes)), polygons_(boxes_.size())
            {
                Detail::check_label_counts(classes_.size(), boxes_.size(), polygons_.size());
            }

            /**
             * @brief Constructs a set of objects with their polygons.
             * @param boxes One per class; otherwise throws Error.
             * @param polygons One per class, empty for an object without one; else throws Error.
             */
            LabelSet(std::vector<std::int32_t> classes, std::vector<Box> boxes,
                     std::vector<Polygon> polygons)
                : classes_(std::move(classes)), boxes_(std::move(boxes)),
                  polygons_(std::move(polygons))
            {
                Detail::check_label_counts(classes_.size(), boxes_.size(), polygons_.size());
            }

            /** @brief Reserves room for a number of objects in all three lists. */
            void reserve(std::size_t count)
            {
                classes_.reserve(count);
                boxes_.reserve(count);
                polygons_.reserve(count);
            }

            /**
             * @brief Appends one object.
             * @param polygon The object's outline; leave it empty for an object without one.
             */
            void add(std::int32_t class_id, const Box& box, Polygon polygon = Polygon{})
            {
                classes_.push_back(class_id);
                boxes_.push_back(box);
                polygons_.push_back(std::move(polygon));
            }

            /**
             * @brief Removes every object a predicate rejects, keeping the order of the rest.
             * @param is_kept Called once per index in rising order; it may read this set.
             */
            template <typename Predicate>
            void keep_if(Predicate is_kept)
            {
                std::size_t kept = 0;

                for (std::size_t i = 0; i < boxes_.size(); ++i)
                {
                    if (!is_kept(i))
                        continue;

                    // Entries from kept on are still unread or already moved from.
                    if (kept != i)
                    {
                        classes_[kept] = classes_[i];
                        boxes_[kept] = boxes_[i];
                        polygons_[kept] = std::move(polygons_[i]);
                    }

                    ++kept;
                }

                classes_.resize(kept);
                boxes_.resize(kept);
                polygons_.resize(kept);
            }

            /** @brief Returns the object count. */
            [[nodiscard]] std::size_t size() const noexcept { return boxes_.size(); }

            /** @brief Reports whether the image has no object, as a background image does. */
            [[nodiscard]] bool is_empty() const noexcept { return boxes_.empty(); }

            /** @brief Returns one class id per object. */
            [[nodiscard]] const std::vector<std::int32_t>& classes() const noexcept
            {
                return classes_;
            }

            /** @brief Returns one box per object. */
            [[nodiscard]] const std::vector<Box>& boxes() const noexcept { return boxes_; }

            /** @brief Returns one polygon per object, of no vertex where the object has none. */
            [[nodiscard]] const std::vector<Polygon>& polygons() const noexcept
            {
                return polygons_;
            }

            /**
             * @brief Returns one object's box for writing.
             * @param index In [0, size()); not checked.
             */
            [[nodiscard]] Box& box(std::size_t index) noexcept { return boxes_[index]; }

            /**
             * @brief Returns one object's polygon for writing.
             * @param index In [0, size()); not checked.
             */
            [[nodiscard]] Polygon& polygon(std::size_t index) noexcept { return polygons_[index]; }

        private:
            /// @brief Holds one class id per object, each in [0, class count).
            std::vector<std::int32_t> classes_;

            /// @brief Holds one box per object.
            std::vector<Box> boxes_;

            /// @brief Holds one polygon per object.
            std::vector<Polygon> polygons_;
    };

    /** @brief Names the label set a label file parses into, in fractions of the image. */
    using FileLabels = LabelSet<NormalizedBox>;

    /** @brief Names the label set every transform takes and returns, in pixels. */
    using Labels = LabelSet<PixelBox>;

    /**
     * @brief Resamples every polygon of one image to one point count.
     *
     * The count is resampled_point_count of the longest polygon, and each polygon goes
     * through resample_polygon, as the reference does on every load
     * (data/dataset.py:416-422). Boxes, classes and objects without a polygon are unchanged.
     */
    [[nodiscard]] FileLabels resample_polygons(FileLabels labels);

    /**
     * @brief Converts file labels to pixels on an image of the given size.
     *
     * Each box goes through to_pixel_box; each vertex has x multiplied by size.width and y
     * by size.height, in float. Classes and order are kept.
     */
    [[nodiscard]] Labels to_pixel_labels(const FileLabels& labels, ImageSize size);

    /**
     * @brief Converts pixel labels to fractions of an image of the given size.
     *
     * Each box goes through to_normalized_box; each vertex has x divided by size.width and
     * y by size.height, in float. Classes and order are kept.
     */
    [[nodiscard]] FileLabels to_normalized_labels(const Labels& labels, ImageSize size);

    /**
     * @brief Clips every box with clip_box and clamps every vertex to the image, removing none.
     *
     * A vertex's x is clamped to [0, size.width] and its y to [0, size.height]
     * (utils/instance.py:426-429).
     */
    void clip_labels(Labels& labels, ImageSize size) noexcept;

    /** @brief Removes every object whose box_area is not above zero, keeping the order. */
    void remove_empty_boxes(Labels& labels);
} // namespace SushiAI::Imaging
