/**************************************************************************/
/* rasterise_outlines.hpp                                                 */
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
 * @file rasterise_outlines.hpp
 * @brief Declares the mask of class indices painted from the outlines of one image.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>

/** @brief Holds the mask geometry and the metrics of semantic segmentation. */
namespace SushiAI::Vision::Segment
{
    /** @brief Holds the outline of one object and the class its pixels take. */
    struct ClassOutline
    {
        /** @brief Holds the byte written to every pixel the outline covers. */
        std::uint8_t class_index = 0;

        /** @brief Views the closed polygons of the outline; fill_outline() states their rule. */
        SushiRuntime::span<const Imaging::Polygon> parts;
    };

    /**
     * @brief Paints the outlines into a mask of @p size, in order; a later outline wins an
     *        overlap, and a pixel no outline covers holds @p background.
     * @param mask Reset to @p size with one channel; unspecified on failure.
     * @return InvalidDescription for a size outside the bounds of an image, or the failure
     *         of fill_outline() with the position of its outline in front.
     */
    [[nodiscard]] Status rasterise_outlines(SushiRuntime::span<const ClassOutline> outlines,
                                            Imaging::ImageSize size, std::uint8_t background,
                                            Imaging::Image& mask);

} // namespace SushiAI::Vision::Segment
