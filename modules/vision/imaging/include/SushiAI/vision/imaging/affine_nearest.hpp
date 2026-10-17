/**************************************************************************/
/* affine_nearest.hpp                                                     */
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
 * @file affine_nearest.hpp
 * @brief Declares the affine warp that copies source pixels and blends none.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Warps an image into plan.output, each pixel taking the source pixel nearest it.
     *
     * Output pixel (x, y) reads the source pixel at ((x - m[2]) / m[0], (y - m[5]) / m[4])
     * rounded half up; one outside the source is plan.border_value. Nothing is blended.
     * @param destination Reset to plan.output with the source's channels.
     * @return A failure README.md, section affine_nearest.hpp, lists.
     */
    [[nodiscard]] Status apply_affine_nearest(ImageView source, const AffinePlan& plan,
                                              Image& destination);
} // namespace SushiAI::Vision::Imaging
