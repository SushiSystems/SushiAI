/**************************************************************************/
/* input.hpp                                                              */
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
 * @file input.hpp
 * @brief Declares the description of the input a batch carries.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @struct InputDescription
         * @brief States the per-sample shape and the element type of the input of a batch.
         */
        struct InputDescription
        {
            /** @brief Holds the shape of one sample, batch axis left out. */
            Shape sample_shape;

            /** @brief Names the element type of the input. */
            DType dtype = DType::Float32;
        };

    } // namespace Data
} // namespace SushiAI
