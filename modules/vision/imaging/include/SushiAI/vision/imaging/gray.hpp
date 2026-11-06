/**************************************************************************/
/* gray.hpp                                                               */
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
 * @file gray.hpp
 * @brief Declares the expansion of a one-channel image to the three channels detection uses.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /** @brief Holds the channel count of a grey image. */
    inline constexpr std::int32_t K_GRAY_CHANNELS = 1;

    /**
     * @brief Copies a one-channel image into three equal channels.
     *
     * Red, green and blue of a pixel each take its grey level, which is what the reference's
     * colour load gives for a grey file (data/base.py:153).
     * @param destination Reset to the source's size with K_RGB_CHANNELS; must pass check_source.
     * @return InvalidDescription for a source of another channel count.
     */
    [[nodiscard]] Status expand_gray(ImageView source, Image& destination) noexcept;
} // namespace SushiAI::Vision::Imaging
