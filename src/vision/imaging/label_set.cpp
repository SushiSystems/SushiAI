/**************************************************************************/
/* label_set.cpp                                                          */
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
 * @file label_set.cpp
 * @brief Implements the count check of a label set and the rules that apply to a whole set.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/label_set.hpp>

#include <algorithm>

#include <SushiAI/core/common.hpp>

namespace SushiAI::Vision::Imaging
{
    void Detail::check_label_counts(std::size_t classes, std::size_t boxes, std::size_t polygons)
    {
        SA_THROW_IF(classes != boxes || polygons != boxes,
                    "LabelSet: {} classes, {} boxes and {} polygons do not describe the same "
                    "objects; the three counts must match",
                    classes, boxes, polygons);
    }

    FileLabels resample_polygons(FileLabels labels)
    {
        std::size_t longest = 0;
        for (const Polygon& polygon : labels.polygons())
            longest = std::max(longest, polygon.size());

        if (longest == 0)
            return labels;

        const std::size_t count = resampled_point_count(longest);
        for (std::size_t i = 0; i < labels.size(); ++i)
            labels.polygon(i) = resample_polygon(labels.polygon(i), count);

        return labels;
    }

    Labels to_pixel_labels(const FileLabels& labels, ImageSize size)
    {
        const float width = static_cast<float>(size.width);
        const float height = static_cast<float>(size.height);

        Labels pixels;
        pixels.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            Polygon polygon = labels.polygons()[i];
            map_polygon(polygon, width, height, 0.0f, 0.0f);
            pixels.add(labels.classes()[i], to_pixel_box(labels.boxes()[i], size),
                       std::move(polygon));
        }

        return pixels;
    }

    FileLabels to_normalized_labels(const Labels& labels, ImageSize size)
    {
        const float width = static_cast<float>(size.width);
        const float height = static_cast<float>(size.height);

        FileLabels normalized;
        normalized.reserve(labels.size());

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            Polygon polygon = labels.polygons()[i];
            for (Point& point : polygon)
            {
                point.x /= width;
                point.y /= height;
            }

            normalized.add(labels.classes()[i], to_normalized_box(labels.boxes()[i], size),
                           std::move(polygon));
        }

        return normalized;
    }

    void clip_labels(Labels& labels, ImageSize size) noexcept
    {
        const PixelBox image{0.0f, 0.0f, static_cast<float>(size.width),
                             static_cast<float>(size.height)};

        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            labels.box(i) = clip_box(labels.box(i), size);
            clamp_polygon(labels.polygon(i), image);
        }
    }

    void remove_empty_boxes(Labels& labels)
    {
        labels.keep_if([&labels](std::size_t index)
        {
            return box_area(labels.boxes()[index]) > 0.0f;
        });
    }
} // namespace SushiAI::Vision::Imaging
