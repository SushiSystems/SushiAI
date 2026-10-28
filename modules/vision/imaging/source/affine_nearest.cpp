/**************************************************************************/
/* affine_nearest.cpp                                                     */
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
 * @file affine_nearest.cpp
 * @brief Implements the affine warp that copies source pixels and blends none.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/affine_nearest.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>


#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_WARP = "apply_affine_nearest";

        /** @brief Marks an output coordinate whose nearest source pixel lies outside. */
        constexpr std::int32_t K_OUTSIDE = -1;

        /**
         * @brief Returns the source index nearest one output index, or K_OUTSIDE.
         * @param offset The matrix translation of the axis.
         * @param scale The matrix scale of the axis, above zero.
         * @param extent The source length along the axis.
         */
        [[nodiscard]] std::int32_t nearest_index(std::int32_t output_index, double offset,
                                                 double scale, std::int32_t extent) noexcept
        {
            const double position = (static_cast<double>(output_index) - offset) / scale;
            const double nearest = std::floor(position + 0.5);

            if (!(nearest >= 0.0 && nearest <= static_cast<double>(extent - 1)))
                return K_OUTSIDE;

            return static_cast<std::int32_t>(nearest);
        }

        /** @brief Returns the failure of a plan whose matrix is no scale and translation. */
        [[nodiscard]] Status check_matrix(const AffinePlan& plan)
        {
            const std::array<float, 6>& m = plan.matrix;

            SA_FAIL_IF(!(m[0] > 0.0f) || !(m[4] > 0.0f) || !std::isfinite(m[0]) ||
                           !std::isfinite(m[4]),
                       StatusCode::InvalidDescription,
                       "{}: the matrix scales must be finite and above zero; got {} and {}",
                       K_WARP, m[0], m[4]);
            SA_FAIL_IF(m[0] != plan.scale || m[4] != plan.scale, StatusCode::InvalidDescription,
                       "{}: the matrix scales {} and {} must equal the plan's scale {}", K_WARP,
                       m[0], m[4], plan.scale);
            SA_FAIL_IF(m[1] != 0.0f || m[3] != 0.0f, StatusCode::InvalidDescription,
                       "{}: the warp is a scale and a translation only; the matrix has "
                       "off-diagonal terms {} and {}",
                       K_WARP, m[1], m[3]);
            SA_FAIL_IF(!std::isfinite(m[2]) || !std::isfinite(m[5]),
                       StatusCode::InvalidDescription,
                       "{}: the matrix needs a finite translation; got {} and {}", K_WARP, m[2],
                       m[5]);
            return Status();
        }

        /** @brief Writes the warped rows; destination has plan.output and the source's channels. */
        void warp_rows(ImageView source, const AffinePlan& plan,
                       const std::vector<std::int32_t>& columns, Image& destination) noexcept
        {
            const std::size_t channels = static_cast<std::size_t>(source.channels());

            for (std::int32_t y = 0; y < plan.output.height; ++y)
            {
                const std::int32_t source_row =
                    nearest_index(y, plan.matrix[5], plan.matrix[4], source.height());
                std::uint8_t* out = destination.row(y);

                if (source_row == K_OUTSIDE)
                {
                    std::fill_n(out, destination.row_bytes(), plan.border_value);
                    continue;
                }

                const std::uint8_t* const in = source.row(source_row);
                for (std::int32_t x = 0; x < plan.output.width; ++x, out += channels)
                {
                    const std::int32_t column = columns[static_cast<std::size_t>(x)];
                    if (column == K_OUTSIDE)
                        std::fill_n(out, channels, plan.border_value);
                    else
                        std::copy_n(in + static_cast<std::size_t>(column) * channels, channels,
                                    out);
                }
            }
        }
    } // namespace

    Status apply_affine_nearest(ImageView source, const AffinePlan& plan, Image& destination) noexcept
    {
        SA_FAIL_IF(source.data() == nullptr || !is_valid_size(source.size()) ||
                       !is_valid_channel_count(source.channels()),
                   StatusCode::InvalidDescription, "{}: the source holds no image", K_WARP);
        SA_FAIL_IF(overlaps(source, destination), StatusCode::InvalidDescription,
                   "{}: the source and the destination share pixels", K_WARP);
        SA_FAIL_IF(source.size() != plan.source, StatusCode::Incompatible,
                   "{}: the plan was made for a {}x{} source; the image is {}x{}", K_WARP,
                   plan.source.height, plan.source.width, source.height(), source.width());
        SA_FAIL_IF(!is_valid_size(plan.output), StatusCode::InvalidDescription,
                   "{}: an output size of {}x{} is outside [1, {}]", K_WARP, plan.output.height,
                   plan.output.width, K_MAX_IMAGE_SIDE);
        SA_TRY(check_matrix(plan));

        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            // Every output row reads the same source columns, so they are found once.
            std::vector<std::int32_t> columns(static_cast<std::size_t>(plan.output.width));
            for (std::int32_t x = 0; x < plan.output.width; ++x)
                columns[static_cast<std::size_t>(x)] =
                    nearest_index(x, plan.matrix[2], plan.matrix[0], source.width());

            destination.reset(plan.output, source.channels());
            warp_rows(source, plan, columns, destination);
            return Status();
        });
    }
} // namespace SushiAI::Vision::Imaging
