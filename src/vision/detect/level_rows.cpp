/**************************************************************************/
/* level_rows.cpp                                                         */
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
 * @file level_rows.cpp
 * @brief Implements the copies between one level's rows and the joined anchor axis.
 * @author Mustafa Garip
 * @see include/SushiAI/vision/detect/README.md
 */

#include <SushiAI/vision/detect/level_rows.hpp>

#include <algorithm>
#include <cstddef>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        namespace
        {
            /** @brief Holds a placement's element counts, checked against two array sizes. */
            struct LevelExtents
            {
                std::size_t batch = 0;        /**< Images. */
                std::size_t joined_image = 0; /**< Values of one image in the joined array. */
                std::size_t level_image = 0;  /**< Values of one image in the level array. */
                std::size_t offset = 0;       /**< First value of the level in a joined image. */
            };

            /** @brief Returns the element counts of @p placement; throws on a size mismatch. */
            LevelExtents check_placement(const LevelPlacement& placement,
                                         std::size_t level_size,
                                         std::size_t joined_size)
            {
                SA_THROW_IF(placement.batch <= 0 || placement.anchors <= 0 ||
                                placement.channels <= 0 || placement.level_anchors <= 0 ||
                                placement.anchor_offset < 0 ||
                                placement.anchor_offset + placement.level_anchors >
                                    placement.anchors,
                            "a level of {} anchors at offset {} does not fit {} anchors with "
                            "batch {} and {} channels",
                            placement.level_anchors, placement.anchor_offset, placement.anchors,
                            placement.batch, placement.channels);

                LevelExtents extents;
                extents.batch = static_cast<std::size_t>(placement.batch);
                extents.joined_image =
                    static_cast<std::size_t>(placement.anchors * placement.channels);
                extents.level_image =
                    static_cast<std::size_t>(placement.level_anchors * placement.channels);
                extents.offset =
                    static_cast<std::size_t>(placement.anchor_offset * placement.channels);

                SA_THROW_IF(level_size != extents.batch * extents.level_image ||
                                joined_size != extents.batch * extents.joined_image,
                            "the placement needs {} level values and {} joined values; got {} "
                            "and {}",
                            extents.batch * extents.level_image,
                            extents.batch * extents.joined_image, level_size, joined_size);
                return extents;
            }
        } // namespace

        void join_level(SushiRuntime::span<const float> level,
                        const LevelPlacement& placement,
                        SushiRuntime::span<float> joined)
        {
            const LevelExtents extents = check_placement(placement, level.size(), joined.size());
            for (std::size_t image = 0; image < extents.batch; ++image)
            {
                const float* source = level.data() + image * extents.level_image;
                std::copy(source, source + extents.level_image,
                          joined.data() + image * extents.joined_image + extents.offset);
            }
        }

        void split_level(SushiRuntime::span<const float> joined,
                         const LevelPlacement& placement,
                         SushiRuntime::span<float> level)
        {
            const LevelExtents extents = check_placement(placement, level.size(), joined.size());
            for (std::size_t image = 0; image < extents.batch; ++image)
            {
                const float* source =
                    joined.data() + image * extents.joined_image + extents.offset;
                std::copy(source, source + extents.level_image,
                          level.data() + image * extents.level_image);
            }
        }

    } // namespace Vision::Detect
} // namespace SushiAI
