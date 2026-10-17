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
 * @file gray.cpp
 * @brief Implements the expansion of a one-channel image to three equal channels.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/gray.hpp>

#include <cstddef>

#include <SushiAI/failure/error.hpp>

namespace SushiAI::Vision::Imaging
{
    void expand_gray(ImageView source, Image& destination)
    {
        check_source(source, destination, "expand_gray");
        SA_THROW_IF(source.channels() != K_GRAY_CHANNELS,
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
    }
} // namespace SushiAI::Vision::Imaging
