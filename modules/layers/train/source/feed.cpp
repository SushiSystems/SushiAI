/**************************************************************************/
/* feed.cpp                                                               */
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
 * @file feed.cpp
 * @brief Implements check_feed.
 * @author Mustafa Garip
 */

#include <SushiAI/train/feed.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        Status check_feed(const SushiBLAS::Tensor& source, const SushiBLAS::Tensor& destination,
                          std::string_view what) noexcept
        {
            SA_TRY_ASSIGN(const Shape fed, checked_shape_of(source));
            SA_TRY_ASSIGN(const Shape expected, checked_shape_of(destination));

            SA_FAIL_IF(fed != expected, StatusCode::Incompatible,
                       "The batch's {} have shape {} but the compiled step expects {}",
                       what, fed.to_string(), expected.to_string());
            SA_FAIL_IF(source.dtype != destination.dtype, StatusCode::Incompatible,
                       "The batch's {} are {} but the compiled step expects {}",
                       what, dtype_name(source.dtype), dtype_name(destination.dtype));

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
