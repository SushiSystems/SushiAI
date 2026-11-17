/**************************************************************************/
/* darknet_resize.hpp                                                     */
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
 * @file darknet_resize.hpp
 * @brief Declares the bilinear resize of Darknet's resize_image, which maps corners to corners.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Resizes an image by Darknet's bilinear kernel, which maps corners to corners
     *        (resize_bilinear maps pixel centres), blends columns then rows in double and
     *        rounds half up; a destination side of 1 reads source position 0.
     * @param destination Reset to size with the source's channels; must pass check_source.
     * @return InvalidDescription unless @p size passes is_valid_size.
     */
    [[nodiscard]] Status resize_darknet(ImageView source, ImageSize size,
                                        Image& destination) noexcept;
} // namespace SushiAI::Vision::Imaging
