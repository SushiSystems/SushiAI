/**************************************************************************/
/* darknet_fold.hpp                                                       */
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
 * @file darknet_fold.hpp
 * @brief Declares the batch-normalisation fold of a Darknet convolution.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /**
         * @brief Folds the batch norm of a normalised convolution into its weights and bias.
         * @param convolution The convolution to change; it is left with no scales or statistics.
         * @pre Each per-filter vector has one entry per filter, the weights hold whole filters,
         *      and rolling_variance + 1e-5 is positive; a violation stops the process.
         */
        void fold_darknet_normalisation(DarknetConvolution& convolution) noexcept;

    } // namespace Vision::Importers
} // namespace SushiAI
