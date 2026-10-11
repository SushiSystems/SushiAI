/**************************************************************************/
/* one_hot.hpp                                                            */
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
 * @file one_hot.hpp
 * @brief Declares the host fill that turns a batch's class indices into one-hot rows.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Writes one row per label into @p one_hot: 1 in the label's column, 0 elsewhere.
         * @param labels An [N] int32 tensor the host can read.
         * @param one_hot An [N, C] float32 or float64 tensor the host can write.
         * @pre No recorded task reads or writes either tensor.
         * @return Success, or Incompatible, with @p one_hot unchanged, when a shape or an
         *         element type is not the one stated or a label lies outside [0, C).
         */
        [[nodiscard]] Status fill_one_hot(const SushiBLAS::Tensor& labels,
                                          SushiBLAS::Tensor& one_hot);

    } // namespace Train
} // namespace SushiAI
