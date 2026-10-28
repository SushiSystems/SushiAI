/**************************************************************************/
/* level_rows.hpp                                                         */
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
 * @file level_rows.hpp
 * @brief Declares the copies between one level's rows and the anchor axis that joins the levels.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /** @brief Holds where one level's rows sit in a `[B, A, C]` array over all anchors. */
        struct LevelPlacement
        {
            std::int64_t batch = 0;         /**< Images, B. */
            std::int64_t anchors = 0;       /**< Anchors over all levels, A. */
            std::int64_t channels = 0;      /**< Values per anchor, C. */
            std::int64_t anchor_offset = 0; /**< Index of the level's first anchor. */
            std::int64_t level_anchors = 0; /**< Anchors of the level, its height times width. */
        };

        /**
         * @brief Copies one level's `[B, n, C]` rows into their place in a `[B, A, C]` array.
         * @param level `[B, H, W, C]` read as `[B, n, C]`.
         * @param joined `[B, A, C]`; only the level's rows are written.
         * @return InvalidDescription when the level does not fit its anchors, or when
         *         @p level and @p joined do not hold the sizes @p placement gives.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] Status join_level(SushiRuntime::span<const float> level,
                                        const LevelPlacement& placement,
                                        SushiRuntime::span<float> joined) noexcept;

        /**
         * @brief Copies one level's rows out of a `[B, A, C]` array into `[B, n, C]`.
         * @param joined `[B, A, C]`.
         * @param level Receives `[B, n, C]`, which is `[B, H, W, C]` in row-major order.
         * @return InvalidDescription when the level does not fit its anchors, or when
         *         @p joined and @p level do not hold the sizes @p placement gives.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] Status split_level(SushiRuntime::span<const float> joined,
                                         const LevelPlacement& placement,
                                         SushiRuntime::span<float> level) noexcept;

    } // namespace Vision::Detect::Free
} // namespace SushiAI
