/**************************************************************************/
/* polygon_mask.cpp                                                       */
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
 * @file polygon_mask.cpp
 * @brief Implements the painting of a fitted sample's polygons into a mask of class indices.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/polygon_mask.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

#include <SushiBLAS/core/format.hpp>
#include <SushiRuntime/core/span.hpp>

#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>

#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PAINTER = "paint_polygon_mask";
    } // namespace

    Status paint_polygon_mask(const Imaging::Labels& objects, Imaging::ImageSize size,
                              std::uint8_t background, Imaging::Image& mask)
    {
        return Detail::at_boundary(K_PAINTER, [&]() -> Status
        {
            std::vector<Segment::ClassOutline> outlines;
            outlines.reserve(objects.size());

            for (std::size_t object = 0; object < objects.size(); ++object)
            {
                const std::int32_t class_index = objects.classes()[object];
                const Imaging::Polygon& polygon = objects.polygons()[object];

                SA_FAIL_IF(polygon.empty(), StatusCode::InvalidDescription,
                           "{}: object {} has a box and no polygon; a mask is painted from "
                           "polygons",
                           K_PAINTER, object);
                SA_FAIL_IF(class_index < 0 || class_index >= K_MASK_VOID,
                           StatusCode::InvalidDescription,
                           "{}: object {} has the class {}; a mask of bytes names 0 to {}",
                           K_PAINTER, object, class_index, K_MASK_VOID - 1);

                // An object holds one polygon, so its outline has one part.
                outlines.push_back(Segment::ClassOutline{
                    static_cast<std::uint8_t>(class_index),
                    SushiRuntime::span<const Imaging::Polygon>(&polygon, 1)});
            }

            return Segment::rasterise_outlines(
                SushiRuntime::span<const Segment::ClassOutline>(outlines.data(),
                                                                outlines.size()),
                size, background, mask);
        });
    }
} // namespace SushiAI::Vision::Datasets
