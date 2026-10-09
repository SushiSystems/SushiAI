/**************************************************************************/
/* box_delta.hpp                                                          */
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
 * @file box_delta.hpp
 * @brief Declares what a box delta reads and the box deltas an anchor head can take.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/detect/anchor/anchor_size.hpp>

/** @brief Holds the units of the anchor part that no caller names. */
namespace SushiAI::Vision::Detect::Anchor::Detail
{
    /** @brief Holds a box as centre and size, in fractions of the network input. */
    struct CentreBox
    {
        float x = 0.0f;      /**< Centre along the columns. */
        float y = 0.0f;      /**< Centre along the rows. */
        float width = 0.0f;  /**< Width. */
        float height = 0.0f; /**< Height. */
    };

    /** @brief Holds what one assigned slot and its truth give a box delta. */
    struct BoxDeltaTerms
    {
        CentreBox predicted;        /**< The slot's decoded box. */
        CentreBox truth;            /**< The truth's box. */
        AnchorSize anchor;          /**< The slot's anchor. */
        float offset_x = 0.0f;      /**< The slot's activated offset X within its cell. */
        float offset_y = 0.0f;      /**< The slot's activated offset Y within its cell. */
        float raw_width = 0.0f;     /**< The slot's width entry as the head publishes it. */
        float raw_height = 0.0f;    /**< The slot's height entry as the head publishes it. */
        float target_x = 0.0f;      /**< The truth's centre within the cell, along columns. */
        float target_y = 0.0f;      /**< The truth's centre within the cell, along rows. */
        float box_weight = 0.0f;    /**< Factor of the box deltas. */
        float max_delta = 0.0f;     /**< Bound of a delta's magnitude; 0 sets none. */
    };

    /**
     * @brief Names a function that adds one truth's box delta to a slot's four box entries.
     * @param delta The slot's x, y, width and height deltas, in that order.
     */
    using BoxDeltaFunction = void (*)(const BoxDeltaTerms& terms, float* delta) noexcept;

    /**
     * @brief Adds the squared-error delta: each entry's target minus its value, weighted.
     * @param terms A truth and an anchor whose sides are positive.
     * @param delta The slot's x, y, width and height deltas, in that order.
     */
    void add_squared_error_box_delta(const BoxDeltaTerms& terms, float* delta) noexcept;

} // namespace SushiAI::Vision::Detect::Anchor::Detail
