/**************************************************************************/
/* dtype.hpp                                                              */
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
 * @file dtype.hpp
 * @brief Declares DType, Layout, K_MAX_RANK and the element-type helpers.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>

namespace SushiAI
{
    /**
     * @brief The element type of a tensor.
     *
     * An alias, not a parallel enum: SushiAI performs no arithmetic of its
     * own, so the set of types it can express is exactly the set SushiBLAS
     * can compute on.
     */
    using DType = SushiBLAS::Core::DataType;

    /** @brief The memory layout of a tensor, aliased from SushiBLAS for the same reason. */
    using Layout = SushiBLAS::Core::Layout;

    /** @brief Maximum tensor rank, fixed by SushiBLAS's own storage of shape and strides. */
    inline constexpr int32_t K_MAX_RANK = static_cast<int32_t>(SushiBLAS::Core::K_MAX_TENSOR_RANK);

    /**
     * @brief Returns the size in bytes occupied by one element of @p dtype.
     * @param dtype The element type to size.
     * @return The element's size in bytes; never zero.
     */
    constexpr std::size_t dtype_size(DType dtype) noexcept
    {
        switch (dtype)
        {
            case DType::Half:      return 2;
            case DType::Float32:   return 4;
            case DType::Float64:   return 8;
            case DType::Complex32: return 8;
            case DType::Complex64: return 16;
            case DType::Int32:     return 4;
            case DType::Int64:     return 8;
        }
        return 4;
    }

    /**
     * @brief A human-readable name for @p dtype, for diagnostics and error text.
     * @param dtype The element type to name.
     * @return A null-terminated literal owned by the library; never null.
     */
    constexpr const char* dtype_name(DType dtype) noexcept
    {
        switch (dtype)
        {
            case DType::Half:      return "half";
            case DType::Float32:   return "float32";
            case DType::Float64:   return "float64";
            case DType::Complex32: return "complex32";
            case DType::Complex64: return "complex64";
            case DType::Int32:     return "int32";
            case DType::Int64:     return "int64";
        }
        return "unknown";
    }

    /**
     * @brief Reports whether @p dtype is a real floating-point type.
     * @param dtype The element type to test.
     * @return True for HALF, FLOAT32, and FLOAT64; false otherwise.
     */
    constexpr bool is_real_float(DType dtype) noexcept
    {
        return dtype == DType::Half || dtype == DType::Float32 || dtype == DType::Float64;
    }
} // namespace SushiAI
