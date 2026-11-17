/**************************************************************************/
/* leaf_writes.hpp                                                        */
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
 * @file leaf_writes.hpp
 * @brief Declares the check that a feed wrote every leaf its objective declared.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/train/objective.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Fills every leaf of @p leaves that @p pool binds with a pattern no feed writes.
         *
         * The pattern repeats the four bytes 0xAD 0xDE 0xC0 0x7F, the float32 NaN 0x7FC0DEAD.
         * An objective's feed has to overwrite it; require_leaves_written() tells whether it did.
         * The fill costs one pass over the bytes of the leaves, which are the targets of one batch.
         * @pre No recorded task reads or writes a leaf; the host owns them between two steps.
         */
        void mark_leaves_unwritten(const ObjectiveLeaves& leaves, Graph::TensorPool& pool) noexcept;

        /**
         * @brief Returns InvalidDescription for the first leaf of @p leaves, bound in @p pool,
         *        in which any element still holds the pattern of mark_leaves_unwritten().
         *
         * An element holds it when every byte does; a value written as that pattern reads as
         * unwritten, and no objective writes one.
         * @return Success, or the failure naming the role and the first unwritten element.
         */
        [[nodiscard]] Status require_leaves_written(const ObjectiveLeaves& leaves,
                                                    Graph::TensorPool& pool) noexcept;

    } // namespace Train
} // namespace SushiAI
