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
