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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file batch_role.hpp
 * @brief Declares the lookup of a role an objective reads from a batch.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <SushiAI/failure/result.hpp>
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
            const Data::Batch& batch, std::string_view role) noexcept;

    } // namespace Train
} // namespace SushiAI
