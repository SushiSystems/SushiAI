/**************************************************************************/
/* dense_targets.hpp                                                      */
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
 * @file dense_targets.hpp
 * @brief Declares the host fills that copy the real values of a batch role into a target leaf.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Copies every element of @p values into @p targets, in order.
         * @param values A float32 tensor the host can read.
         * @param targets A float32 or float64 tensor of as many elements the host can write.
         * @pre No recorded task reads or writes either tensor.
         * @return Success, or Incompatible, with @p targets unchanged, for another element
         *         type or another element count.
         */
        [[nodiscard]] Status fill_dense_targets(const SushiBLAS::Tensor& values,
                                                SushiBLAS::Tensor& targets) noexcept;

        /**
         * @brief Copies @p labels into @p targets after checking that each is 0 or 1.
         * @param labels A float32 tensor the host can read.
         * @param targets A float32 or float64 tensor of as many elements the host can write.
         * @pre No recorded task reads or writes either tensor.
         * @return Success, or Incompatible, with @p targets unchanged, for what
         *         fill_dense_targets() refuses or for a label that is neither 0 nor 1.
         */
        [[nodiscard]] Status fill_binary_targets(const SushiBLAS::Tensor& labels,
                                                 SushiBLAS::Tensor& targets) noexcept;

    } // namespace Train
} // namespace SushiAI
