/**************************************************************************/
/* fill_outline.hpp                                                       */
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
 * @file fill_outline.hpp
 * @brief Declares the fill of one outline, of one or more closed polygons, into a mask.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /**
     * @brief Writes @p value to every pixel whose centre the outline holds by the even-odd rule.
     *
     * The centre of pixel (x, y) is (x + 0.5, y + 0.5), and a part inside another cuts a hole.
     * @param parts Closed polygons in pixels of @p mask; the last vertex joins the first.
     * @param mask One channel, already sized; a pixel outside the outline keeps its value.
     * @return InvalidDescription for an unsized mask, several channels or a vertex not finite.
     */
    [[nodiscard]] Status fill_outline(SushiRuntime::span<const Imaging::Polygon> parts,
                                      std::uint8_t value, Imaging::Image& mask);

} // namespace SushiAI::Vision::Segment
