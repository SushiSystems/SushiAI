/**************************************************************************/
/* tensor_shape.hpp                                                       */
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
 * @file tensor_shape.hpp
 * @brief Defines shape_of, which reads the extents of a SushiBLAS tensor as a Shape.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    /**
     * @brief Returns the extents of @p tensor as a Shape.
     * @pre The tensor describes a legal shape: every extent is positive and the rank does not
     *      exceed K_MAX_RANK. A tensor a caller supplied goes through checked_shape_of().
     */
    [[nodiscard]] inline Shape shape_of(const SushiBLAS::Tensor& tensor) noexcept
    {
        return Shape(SushiRuntime::span<const int64_t>(tensor.shape.data(),
                                                       static_cast<std::size_t>(tensor.rank)));
    }

    /**
     * @brief Reads the extents of a tensor a caller supplied as a Shape.
     * @return The shape, or InvalidDescription for an extent that is not positive or a rank above
     *         K_MAX_RANK.
     */
    [[nodiscard]] inline Result<Shape> checked_shape_of(const SushiBLAS::Tensor& tensor) noexcept
    {
        return Shape::make(SushiRuntime::span<const int64_t>(
            tensor.shape.data(), static_cast<std::size_t>(tensor.rank)));
    }
} // namespace SushiAI
