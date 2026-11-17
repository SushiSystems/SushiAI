/**************************************************************************/
/* buffer_update_recorder.hpp                                             */
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
 * @file buffer_update_recorder.hpp
 * @brief Declares bind_buffer_updates, which turns a training trace's updates into the slots
 *        Graph::BufferUpdateRecorder blends.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/buffer_update_recorder.hpp>

namespace SushiAI
{
    namespace NN
    {
        class BufferUpdateList;
        class ModelState;
    } // namespace NN

    namespace Graph
    {
        class TensorPool;
    } // namespace Graph

    namespace Train
    {
        /** @brief Names Graph::BufferUpdateSlot, the pair of tensors one update touches. */
        using Graph::BufferUpdateSlot;

        /** @brief Names Graph::BufferUpdateRecorder, the recorder that blends every slot. */
        using Graph::BufferUpdateRecorder;

        /**
         * @brief Resolves each update's target in @p state and its source in @p pool.
         * @param updates The updates a training trace requested, in trace order.
         * @param state The state that owns every target; read and not changed.
         * @param pool The pool that holds every source; it must have materialised them.
         * @return One slot per update, in order; Incompatible when the state has no entry for
         *         a target or when a target and its source differ in shape or dtype.
         */
        [[nodiscard]] Result<std::vector<BufferUpdateSlot>> bind_buffer_updates(
            const NN::BufferUpdateList& updates, NN::ModelState& state,
            const Graph::TensorPool& pool) noexcept;

    } // namespace Train
} // namespace SushiAI
