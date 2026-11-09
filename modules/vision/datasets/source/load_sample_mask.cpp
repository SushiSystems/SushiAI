/**************************************************************************/
/* load_sample_mask.cpp                                                   */
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
 * @file load_sample_mask.cpp
 * @brief Implements the loader of one sample's mask.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/load_sample_mask.hpp>

#include <string_view>


#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>

#include "decode_file.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_LOADER = "load_sample_mask";
    } // namespace

    Status load_sample_mask(const SampleList& list, std::size_t sample,
                            Imaging::Image& destination) noexcept
    {
        return load_sample_mask(list, sample, destination, Detail::file_system());
    }

    Status load_sample_mask(const SampleList& list, std::size_t sample,
                            Imaging::Image& destination, const ByteSource& source) noexcept
    {
        SA_FAIL_IF(sample >= list.samples.size(), StatusCode::InvalidDescription,
                   "{}: sample {} is outside a list of {} samples", K_LOADER, sample,
                   list.samples.size());

        const Sample& record = list.samples[sample];
        SA_FAIL_IF(!record.mask.has_value(), StatusCode::InvalidDescription,
                   "{}: sample {} has no mask", K_LOADER, sample);
        SA_FAIL_IF(*record.mask >= list.files.size(), StatusCode::InvalidDescription,
                   "{}: sample {} names the mask file {} of a list of {} files", K_LOADER, sample,
                   *record.mask, list.files.size());

        Codecs::DecodeOptions options;
        options.layout = Codecs::ChannelLayout::Stored;
        options.palette = Codecs::PaletteRead::Indices;

        SA_TRY_ASSIGN(destination,
                      Detail::decode_file(source, list.files[*record.mask], options, K_LOADER));
        return Status();
    }
} // namespace SushiAI::Vision::Datasets
