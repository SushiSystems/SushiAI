/**************************************************************************/
/* gray.cpp                                                               */
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
 * @file gray.cpp
 * @brief Implements the expansion of a one-channel image to three equal channels.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/gray.hpp>

#include <cstddef>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Imaging
{
    Status expand_gray(ImageView source, Image& destination) noexcept
    {
        check_source(source, destination, "expand_gray");
        SA_FAIL_IF(source.channels() != K_GRAY_CHANNELS, StatusCode::InvalidDescription,
                   "expand_gray: the source has {} channels; a grey image has {}",
                   source.channels(), K_GRAY_CHANNELS);

        destination.reset(source.size(), K_RGB_CHANNELS);

        const std::uint8_t* in = source.data();
        const std::uint8_t* const end = in + source.byte_count();
        std::uint8_t* out = destination.data();

        for (; in != end; ++in, out += K_RGB_CHANNELS)
        {
            out[0] = *in;
            out[1] = *in;
            out[2] = *in;
        }

        return Status();
    }
} // namespace SushiAI::Vision::Imaging
