/**************************************************************************/
/* rasterise_outlines.cpp                                                 */
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
 * @file rasterise_outlines.cpp
 * @brief Implements the painting of class outlines into one mask.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/segment/rasterise_outlines.hpp>

#include <cstddef>
#include <string_view>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>

namespace SushiAI::Vision::Segment
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "rasterise_outlines";

        /** @brief Holds the channel count of a mask. */
        constexpr std::int32_t K_MASK_CHANNELS = 1;
    } // namespace

    Status rasterise_outlines(SushiRuntime::span<const ClassOutline> outlines,
                              Imaging::ImageSize size, std::uint8_t background,
                              Imaging::Image& mask) noexcept
    {
        SA_FAIL_IF(!Imaging::is_valid_size(size), StatusCode::InvalidDescription,
                   "{}: a mask of {} by {} pixels is outside the bounds of an image", K_UNIT,
                   size.height, size.width);

        SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
        {
            mask.reset(size, K_MASK_CHANNELS);
            mask.fill(background);
            return Status();
        }));

        for (std::size_t index = 0; index < outlines.size(); ++index)
        {
            const Status filled =
                fill_outline(outlines[index].parts, outlines[index].class_index, mask);
            if (!filled.is_ok())
                return SA_ANNOTATE(filled, "{}: outline {}", K_UNIT, index);
        }

        return Status();
    }

} // namespace SushiAI::Vision::Segment
