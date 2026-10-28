/**************************************************************************/
/* fit_mask.hpp                                                           */
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
 * @file fit_mask.hpp
 * @brief Declares the fit of one sample's mask to the input size by nearest neighbour.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds the byte of a mask that names no class: the padding of a fitted mask. */
    inline constexpr std::uint8_t K_MASK_VOID = 255;

    /**
     * @brief Fits a mask with the geometry fit_sample gives the image it belongs to.
     *
     * A fitted pixel takes the source pixel its centre falls in. SampleFit::Letterbox fills
     * what the mask does not cover with K_MASK_VOID.
     * @param mask One channel of the size of the image; must not be the pixels of @p fitted.
     * @return Incompatible for a mask of several channels or of another size than @p image.
     */
    [[nodiscard]] Status fit_mask(Imaging::ImageView mask, Imaging::ImageSize image,
                                  const SampleFitOptions& options, Imaging::Image& fitted) noexcept;
} // namespace SushiAI::Vision::Datasets
