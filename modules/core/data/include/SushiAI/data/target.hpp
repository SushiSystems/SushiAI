/**************************************************************************/
/* target.hpp                                                             */
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
 * @file target.hpp
 * @brief Declares a named target of a batch, its description, and the lookups by role.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @struct BatchTarget
         * @brief Holds the values one batch carries under one role.
         */
        struct BatchTarget
        {
            /** @brief Names what the values are for, e.g. K_ROLE_LABELS. */
            std::string role;

            /** @brief Holds [N, sample...] values in storage the host and the device share. */
            SushiBLAS::Tensor values;
        };

        /**
         * @struct TargetDescription
         * @brief States the role, the per-sample shape and the element type of one target.
         */
        struct TargetDescription
        {
            /** @brief Names what the values are for, e.g. K_ROLE_LABELS. */
            std::string role;

            /** @brief Holds the shape of one sample's values, batch axis left out. */
            Shape sample_shape;

            /** @brief Names the element type of the values. */
            DType dtype = DType::Float32;

            /** @brief Counts the classes an index spans, or 0 when the values are no index. */
            int64_t classes = 0;
        };

        /**
         * @brief Returns the target of @p targets held under @p role.
         * @return A pointer into @p targets, or null when no target has that role.
         */
        [[nodiscard]] inline const BatchTarget* find_target(
            const std::vector<BatchTarget>& targets, std::string_view role) noexcept
        {
            for (const BatchTarget& target : targets)
            {
                if (target.role == role)
                    return &target;
            }

            return nullptr;
        }

        /**
         * @brief Returns the description of @p targets that states @p role.
         * @return A pointer into @p targets, or null when no description has that role.
         */
        [[nodiscard]] inline const TargetDescription* find_target(
            const std::vector<TargetDescription>& targets, std::string_view role) noexcept
        {
            for (const TargetDescription& target : targets)
            {
                if (target.role == role)
                    return &target;
            }

            return nullptr;
        }

    } // namespace Data
} // namespace SushiAI
