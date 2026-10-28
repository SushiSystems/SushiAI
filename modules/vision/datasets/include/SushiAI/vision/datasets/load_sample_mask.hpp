/**************************************************************************/
/* load_sample_mask.hpp                                                   */
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
 * @file load_sample_mask.hpp
 * @brief Declares the loader of one sample's mask: one class index per pixel.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Decodes the mask file of one sample with Codecs::load_image.
     *
     * A palette image gives its stored indices; any other gives the channels it stores.
     * @param destination Holds the mask on success; unspecified on failure.
     * @return InvalidDescription for a sample the list lacks, a sample with no mask, or a
     *         mask file index the list lacks; otherwise the failure of Codecs::load_image.
     */
    [[nodiscard]] Status load_sample_mask(const SampleList& list, std::size_t sample,
                                          Imaging::Image& destination) noexcept;
} // namespace SushiAI::Vision::Datasets
