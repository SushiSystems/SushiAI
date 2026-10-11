/**************************************************************************/
/* parameter_groups.hpp                                                   */
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
 * @file parameter_groups.hpp
 * @brief Defines which parameters an optimizer's weight decay applies to, by their role.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Reports whether weight decay applies to a parameter of @p role.
         * @return True for a weight; false for a bias, a normalisation scale and an
         *         embedding table. An optimizer's decay_all option overrides a false.
         */
        [[nodiscard]] constexpr bool takes_weight_decay(NN::ParameterRole role) noexcept
        {
            return role == NN::ParameterRole::Weight;
        }

    } // namespace Train
} // namespace SushiAI
