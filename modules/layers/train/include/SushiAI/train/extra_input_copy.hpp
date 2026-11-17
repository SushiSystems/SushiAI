/**************************************************************************/
/* extra_input_copy.hpp                                                   */
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
 * @file extra_input_copy.hpp
 * @brief Declares copy_extra_inputs, which copies the extra inputs of a batch into a graph's leaves.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/input_leaves.hpp>
#include <SushiAI/train/extra_inputs.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Checks each extra input against its leaf and records its copy on @p blas.
         * @param leaves The leaves a graph builder declared.
         * @param pool Holds the tensor of each leaf.
         * @param feeds Scratch the call clears and fills; the caller keeps it to reuse its memory.
         * @return Success, what extra_feeds returns, or what check_feed returns for the first
         *         role whose shape or element type differs from its leaf.
         */
        [[nodiscard]] Status copy_extra_inputs(SushiRuntime::span<const NN::InputLeaf> leaves,
                                               const Data::Batch& batch,
                                               Graph::TensorPool& pool, SushiBLAS::BLASOps& blas,
                                               std::vector<NamedFeed>& feeds) noexcept;

    } // namespace Train
} // namespace SushiAI
