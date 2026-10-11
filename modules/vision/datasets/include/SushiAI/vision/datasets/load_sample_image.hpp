/**************************************************************************/
/* load_sample_image.hpp                                                  */
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
 * @file load_sample_image.hpp
 * @brief Declares the loader of one sample's pixels, stored raw or as an encoded image file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Loads one sample's pixels with read_stored_pixels or Codecs::load_image.
     * @param options Read by Codecs::load_image; a sample with stored pixels ignores them.
     * @param destination Holds the sample's pixels on success; unspecified on failure.
     * @return InvalidDescription for a sample or a file index the list lacks; otherwise the
     *         failure of the function that read the file.
     */
    [[nodiscard]] Status load_sample_image(const SampleList& list, std::size_t sample,
                                           const Codecs::DecodeOptions& options,
                                           Imaging::Image& destination);
} // namespace SushiAI::Vision::Datasets
