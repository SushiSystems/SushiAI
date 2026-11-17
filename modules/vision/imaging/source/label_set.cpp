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
 * @file label_set.cpp
 * @brief Implements the count check of a label set and the rules that apply to a whole set.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/label_set.hpp>

#include <algorithm>

#include <SushiAI/failure/check.hpp>

namespace SushiAI::Vision::Imaging
{
    void Detail::check_label_counts(std::size_t classes, std::size_t boxes, std::size_t outlines)
    {
        SA_FATAL_IF(classes != boxes || outlines != boxes,
                    "LabelSet: {} classes, {} boxes and {} outlines do not describe the same "
                    "objects; the three counts must match",
                    classes, boxes, outlines);
    }

    FileLabels resample_polygons(FileLabels labels)
    {
        std::size_t longest = 0;
        for (const Outline& outline : labels.outlines())
        {
            for (const Polygon& part : outline)
                longest = std::max(longest, part.size());
        }

        if (longest == 0)
            return labels;

        const std::size_t count = resampled_point_count(longest);
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            for (Polygon& part : labels.outline(i))
                part = resample_polygon(part, count);
        }

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
            Outline outline = labels.outlines()[i];
            map_outline(outline, width, height, 0.0f, 0.0f);
            pixels.add(labels.classes()[i], to_pixel_box(labels.boxes()[i], size),
                       std::move(outline));
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
            Outline outline = labels.outlines()[i];
            for (Polygon& part : outline)
            {
                for (Point& point : part)
                {
                    point.x /= width;
                    point.y /= height;
                }
            }

            normalized.add(labels.classes()[i], to_normalized_box(labels.boxes()[i], size),
                           std::move(outline));
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
            clamp_outline(labels.outline(i), image);
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
