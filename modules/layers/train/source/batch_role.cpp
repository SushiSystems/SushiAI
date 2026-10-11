/**************************************************************************/
/* batch_role.cpp                                                         */
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
 * @file batch_role.cpp
 * @brief Implements the lookup of a role an objective reads from a batch.
 * @author Mustafa Garip
 */

#include <SushiAI/train/batch_role.hpp>

#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<const Data::BatchTarget*> require_batch_role(const Data::Batch& batch,
                                                            std::string_view role)
        {
            const Data::BatchTarget* const target = Data::find_target(batch.targets, role);
            SA_FAIL_IF(target == nullptr, StatusCode::InvalidDescription,
                       "The batch holds no '{}' role, which the objective reads", role);

            return target;
        }

    } // namespace Train
} // namespace SushiAI
