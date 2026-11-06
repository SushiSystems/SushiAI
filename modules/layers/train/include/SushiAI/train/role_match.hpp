/**************************************************************************/
/* role_match.hpp                                                         */
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
 * @file role_match.hpp
 * @brief Declares the check of the roles a step reads against the roles a dataset publishes.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Checks that @p dataset publishes every role of @p required as described.
         * @param required The roles a step reads; a class count of 0 accepts any count.
         * @return Success; InvalidDescription naming a role @p dataset does not publish and
         *         the roles it does; or Incompatible naming a role whose per-sample shape,
         *         element type or class count differs, with both descriptions.
         */
        [[nodiscard]] Status match_roles(const std::vector<Data::TargetDescription>& required,
                                         const Data::Dataset& dataset) noexcept;

    } // namespace Train
} // namespace SushiAI
