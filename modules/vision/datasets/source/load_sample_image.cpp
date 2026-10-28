/**************************************************************************/
/* load_sample_image.cpp                                                  */
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
 * @file load_sample_image.cpp
 * @brief Implements the loader of one sample's pixels.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/load_sample_image.hpp>

#include <filesystem>
#include <string_view>


#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_LOADER = "load_sample_image";
    } // namespace

    Status load_sample_image(const SampleList& list, std::size_t sample,
                             const Codecs::DecodeOptions& options, Imaging::Image& destination) noexcept
    {
        SA_FAIL_IF(sample >= list.samples.size(), StatusCode::InvalidDescription,
                   "{}: sample {} is outside a list of {} samples", K_LOADER, sample,
                   list.samples.size());

        const Sample& record = list.samples[sample];
        SA_FAIL_IF(record.file >= list.files.size(), StatusCode::InvalidDescription,
                   "{}: sample {} names file {} of a list of {} files", K_LOADER, sample,
                   record.file, list.files.size());

        const std::filesystem::path& file = list.files[record.file];
        if (record.stored.has_value())
            return read_stored_pixels(file, *record.stored, destination);

        SA_TRY_ASSIGN(destination, Codecs::load_image(file, options));
        return Status();
    }
} // namespace SushiAI::Vision::Datasets
