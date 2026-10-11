/**************************************************************************/
/* host_reals.hpp                                                         */
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
 * @file host_reals.hpp
 * @brief Defines the access to a float32 or float64 tensor from the host by its element type.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI::Vision::Training::Detail
{
    /** @brief Reports whether @p tensor holds float32 or float64 elements. */
    [[nodiscard]] inline bool holds_reals(const SushiBLAS::Tensor& tensor) noexcept
    {
        return tensor.dtype == DType::Float32 || tensor.dtype == DType::Float64;
    }

    /**
     * @brief Calls @p visit once with the elements of @p tensor as float* or as double*.
     * @tparam Visit Callable with either pointer; a generic lambda is.
     * @pre holds_reals(tensor), and no recorded task reads or writes the tensor.
     */
    template<typename Visit>
    void write_reals(SushiBLAS::Tensor& tensor, Visit&& visit)
    {
        if (tensor.dtype == DType::Float64)
            std::forward<Visit>(visit)(tensor.data_as<double>());
        else
            std::forward<Visit>(visit)(tensor.data_as<float>());
    }

    /**
     * @brief Calls @p visit once with the elements of @p tensor as const float* or const double*.
     * @tparam Visit Callable with either pointer; a generic lambda is.
     * @pre holds_reals(tensor), and no recorded task writes the tensor.
     */
    template<typename Visit>
    auto read_reals(const SushiBLAS::Tensor& tensor, Visit&& visit)
    {
        if (tensor.dtype == DType::Float64)
            return std::forward<Visit>(visit)(tensor.data_as<double>());

        return std::forward<Visit>(visit)(tensor.data_as<float>());
    }

} // namespace SushiAI::Vision::Training::Detail
