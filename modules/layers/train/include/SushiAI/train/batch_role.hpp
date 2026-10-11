/**************************************************************************/
/* batch_role.hpp                                                         */
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
 * @file batch_role.hpp
 * @brief Declares the lookup of a role an objective reads from a batch.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Returns the target @p batch holds under @p role.
         * @return A pointer into @p batch, never null, or InvalidDescription naming the
         *         role when @p batch does not hold it.
         */
        [[nodiscard]] Result<const Data::BatchTarget*> require_batch_role(
            const Data::Batch& batch, std::string_view role);

    } // namespace Train
} // namespace SushiAI
