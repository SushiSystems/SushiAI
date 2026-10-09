/**************************************************************************/
/* feed.hpp                                                               */
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
 * @file feed.hpp
 * @brief Declares check_feed, which compares a batch tensor with the leaf a step copies it into.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiBLAS/tensor.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Returns Incompatible unless @p source has the rank, extents and dtype of
         *        @p destination.
         * @param what Names the fed tensors in the message, in the plural: "inputs", "targets".
         * @return Success, Incompatible whose text holds both shapes, or both dtypes when the
         *         shapes agree, or InvalidDescription for extents no Shape can hold.
         */
        [[nodiscard]] Status check_feed(const SushiBLAS::Tensor& source,
                                        const SushiBLAS::Tensor& destination,
                                        std::string_view what);

    } // namespace Train
} // namespace SushiAI
