/**************************************************************************/
/* exact_sum.hpp                                                          */
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
 * @file exact_sum.hpp
 * @brief Declares sums_exactly, which types the replicas' results add as counts.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <SushiBLAS/core/common.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Returns whether results of @p dtype add as exact counts, each with weight 1.
         *
         * Int32, Int64 and UInt32 are counts; every floating and complex type is weighed.
         */
        [[nodiscard]] constexpr bool sums_exactly(SushiBLAS::Core::DataType dtype) noexcept
        {
            switch (dtype)
            {
                case SushiBLAS::Core::DataType::Int32:
                case SushiBLAS::Core::DataType::Int64:
                case SushiBLAS::Core::DataType::UInt32:
                    return true;
                case SushiBLAS::Core::DataType::Half:
                case SushiBLAS::Core::DataType::Float32:
                case SushiBLAS::Core::DataType::Float64:
                case SushiBLAS::Core::DataType::Complex32:
                case SushiBLAS::Core::DataType::Complex64:
                    return false;
            }
            return false;
        }

    } // namespace Distribution
} // namespace SushiAI
