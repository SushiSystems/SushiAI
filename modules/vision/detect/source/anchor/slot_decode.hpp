/**************************************************************************/
/* slot_decode.hpp                                                        */
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
 * @file slot_decode.hpp
 * @brief Defines the two codings of an anchor slot and the decode of its box.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cmath>
#include <cstddef>

#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace SushiAI::Vision::Detect::Anchor::Detail
{
    /** @brief Reads a slot whose offsets and scores are logits and sizes exponents. */
    struct LogitRule
    {
        /** @brief Returns the logistic of an offset, objectness or class entry. */
        static float activate(float value) noexcept
        {
            return 1.0f / (1.0f + std::exp(-value));
        }

        /** @brief Returns a box side: the exponential of @p value times @p anchor. */
        static float size(float value, float anchor) noexcept
        {
            return std::exp(value) * anchor;
        }
    };

    /** @brief Reads a slot whose entries all arrive in (0, 1). */
    struct ActivatedRule
    {
        /** @brief Returns an offset, objectness or class entry as it is. */
        static float activate(float value) noexcept
        {
            return value;
        }

        /** @brief Returns a box side: @p value squared, times four, times @p anchor. */
        static float size(float value, float anchor) noexcept
        {
            return value * value * 4.0f * anchor;
        }
    };

    /** @brief Holds the grid and the stretch every slot of one head decodes under. */
    struct SlotFrame
    {
        float grid_width = 1.0f;  /**< Grid columns W. */
        float grid_height = 1.0f; /**< Grid rows H. */
        float scale = 1.0f;       /**< Stretch of an offset. */
        float shift = 0.0f;       /**< `(scale - 1) / 2`, taken off a stretched offset. */
    };

    /** @brief Holds a slot's box in fractions of the input and the offsets it came from. */
    struct SlotBox
    {
        float x = 0.0f;        /**< Centre along the columns. */
        float y = 0.0f;        /**< Centre along the rows. */
        float width = 0.0f;    /**< Width. */
        float height = 0.0f;   /**< Height. */
        float offset_x = 0.0f; /**< Activated, stretched offset along the columns. */
        float offset_y = 0.0f; /**< Activated, stretched offset along the rows. */
    };

    /** @brief Returns the frame of a grid of @p height by @p width under @p scale_xy. */
    inline SlotFrame make_slot_frame(std::size_t height, std::size_t width,
                                     float scale_xy) noexcept
    {
        SlotFrame frame;
        frame.grid_width = static_cast<float>(width);
        frame.grid_height = static_cast<float>(height);
        frame.scale = scale_xy;
        frame.shift = (scale_xy - 1.0f) / 2.0f;
        return frame;
    }

    /**
     * @brief Decodes the box of the slot at @p slot, which lies in cell (@p row, @p col).
     * @tparam Rule Gives `activate(value)` and `size(value, anchor)` as static functions.
     * @param slot The slot's entries; the four box entries are read.
     */
    template<typename Rule>
    SlotBox decode_slot(const float* slot, const SlotFrame& frame, std::size_t row,
                        std::size_t col, AnchorSize anchor) noexcept
    {
        SlotBox box;
        box.offset_x = Rule::activate(slot[K_X_ENTRY]) * frame.scale - frame.shift;
        box.offset_y = Rule::activate(slot[K_Y_ENTRY]) * frame.scale - frame.shift;
        box.x = (static_cast<float>(col) + box.offset_x) / frame.grid_width;
        box.y = (static_cast<float>(row) + box.offset_y) / frame.grid_height;
        box.width = Rule::size(slot[K_WIDTH_ENTRY], anchor.width);
        box.height = Rule::size(slot[K_HEIGHT_ENTRY], anchor.height);
        return box;
    }

} // namespace SushiAI::Vision::Detect::Anchor::Detail
