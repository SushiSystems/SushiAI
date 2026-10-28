/**************************************************************************/
/* head_options.hpp                                                       */
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
 * @file head_options.hpp
 * @brief Declares the options, the extents and the channel layout of an anchor head.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the entry of the centre's x offset within an anchor slot. */
        inline constexpr std::int64_t K_X_ENTRY = 0;

        /** @brief Holds the entry of the centre's y offset within an anchor slot. */
        inline constexpr std::int64_t K_Y_ENTRY = 1;

        /** @brief Holds the entry of the width within an anchor slot. */
        inline constexpr std::int64_t K_WIDTH_ENTRY = 2;

        /** @brief Holds the entry of the height within an anchor slot. */
        inline constexpr std::int64_t K_HEIGHT_ENTRY = 3;

        /** @brief Holds the entry of the objectness within an anchor slot. */
        inline constexpr std::int64_t K_OBJECTNESS_ENTRY = 4;

        /** @brief Holds the entry of class 0 within an anchor slot, and the entries before it. */
        inline constexpr std::int64_t K_FIRST_CLASS_ENTRY = 5;

        /** @brief Names how a head's published values encode a box and its scores. */
        enum class HeadCoding : std::uint8_t
        {
            Logit,     /**< Offsets, objectness and classes are logits; sizes are exponents. */
            Activated, /**< Every entry arrives in (0, 1); a size is its square times four. */
        };

        /** @brief Names how a logit head turns its width and height entries into sides. */
        enum class SizeForm : std::uint8_t
        {
            Exponential,    /**< The exponential of the entry times the anchor. */
            SquaredSigmoid, /**< Twice the logistic of the entry, squared, times the anchor. */
        };

        /** @brief Holds what a head's decode needs besides its anchors. */
        struct HeadOptions
        {
            std::int64_t num_classes = 0;           /**< Class entries per anchor slot. */
            HeadCoding coding = HeadCoding::Logit;  /**< Encoding of the published values. */
            float scale_xy = 1.0f;                  /**< Stretch of an offset about 0.5. */
            /// @brief Holds the size form of a Logit head; an Activated head keeps the default.
            SizeForm size = SizeForm::Exponential;
        };

        /** @brief Holds the extents of a head's `[B, H, W, A * (5 + C)]` output. */
        struct HeadShape
        {
            std::int64_t batch = 0;  /**< Images B. */
            std::int64_t height = 0; /**< Grid rows H. */
            std::int64_t width = 0;  /**< Grid columns W. */
        };

        /** @brief Holds the size of the network input in pixels. */
        struct NetworkSize
        {
            std::int64_t height = 0; /**< Rows of the input. */
            std::int64_t width = 0;  /**< Columns of the input. */
        };

        /**
         * @brief Checks a head's options against the layout and the head they describe.
         * @return Success; the failure of validate_anchor_layout; or InvalidDescription for a
         *         head index outside the layout, a class count below 1, a scale that is not
         *         finite and above zero, or an Activated head with a size form other than
         *         Exponential.
         */
        [[nodiscard]] Status validate_head(const HeadOptions& options, const AnchorLayout& layout,
                                           std::int64_t head_index) noexcept;

        /**
         * @brief Returns the channels of a head's output, `A * (5 + C)`.
         * @pre validate_head succeeds on the same arguments; a head outside the layout is fatal.
         */
        [[nodiscard]] std::int64_t count_head_channels(const HeadOptions& options,
                                                       const AnchorLayout& layout,
                                                       std::int64_t head_index) noexcept;

        /**
         * @brief Returns the rows one image of a head decodes to, `H * W * A`.
         * @pre @p head_index names a head of @p layout; a head outside the layout is fatal.
         */
        [[nodiscard]] std::int64_t count_head_rows(const HeadShape& shape,
                                                   const AnchorLayout& layout,
                                                   std::int64_t head_index) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
