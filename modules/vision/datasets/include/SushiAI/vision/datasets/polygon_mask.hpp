/**************************************************************************/
/* polygon_mask.hpp                                                       */
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
 * @file polygon_mask.hpp
 * @brief Declares the painting of a fitted sample's polygons into a mask of class indices.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Paints the polygon of each object with the object's class, in the set's order.
     *
     * The later of two objects that overlap wins; a pixel no polygon holds, by the rule
     * of Segment::fill_outline, takes @p background.
     * @param objects In pixels of a mask of @p size, as fit_sample leaves them.
     * @return InvalidDescription for an object with no polygon or a class above 254.
     */
    [[nodiscard]] Status paint_polygon_mask(const Imaging::Labels& objects,
                                            Imaging::ImageSize size, std::uint8_t background,
                                            Imaging::Image& mask) noexcept;
} // namespace SushiAI::Vision::Datasets
