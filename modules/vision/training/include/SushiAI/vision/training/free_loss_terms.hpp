/**************************************************************************/
/* free_loss_terms.hpp                                                  */
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
 * @file free_loss_terms.hpp
 * @brief Declares FreeLossTerms, the loss of an anchor-free step and its terms as numbers.
 * @author Mustafa Garip
 */

#pragma once

namespace SushiAI::Vision::Training
{
    /** @brief Holds the loss and its terms, each term before its gain. */
    struct FreeLossTerms
    {
        double total = 0.0;        /**< Gained terms times the batch. */
        double box = 0.0;          /**< Sum of (1 - CIoU) * weight. */
        double classes = 0.0;      /**< Cross-entropy over the score sum. */
        double distribution = 0.0; /**< DFL; 0 at reg_max 1. */
        double l1 = 0.0;           /**< L1; 0 at l1_gain 0. */
    };

} // namespace SushiAI::Vision::Training
