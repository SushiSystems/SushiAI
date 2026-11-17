/**************************************************************************/
/* extra_inputs.hpp                                                       */
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
 * @file extra_inputs.hpp
 * @brief Declares NamedFeed and extra_feeds, which read the extra inputs from the roles of a
 *        batch.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/nn/input_leaves.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Pairs one tensor of a run with the name of the leaf it is copied into. */
        struct NamedFeed
        {
            /** @brief Names an input leaf of the step. */
            std::string_view name;

            /** @brief Points at the tensor to copy; never null. */
            const SushiBLAS::Tensor* tensor = nullptr;
        };

        /**
         * @brief Appends to @p feeds one feed per extra input, from the batch role of its name.
         *
         * The trainer, the replica and the evaluation step read through this one function.
         * @param leaves The leaves a graph builder declared; the feeds follow their order.
         * @return Success, InvalidDescription naming the first leaf the batch has no role for,
         *         or OutOfMemory.
         */
        [[nodiscard]] Status extra_feeds(SushiRuntime::span<const NN::InputLeaf> leaves,
                                         const Data::Batch& batch,
                                         std::vector<NamedFeed>& feeds) noexcept;

    } // namespace Train
} // namespace SushiAI
