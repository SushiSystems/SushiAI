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
 * @file buffer_update_recorder.hpp
 * @brief Declares BufferUpdateSlot, bind_buffer_updates and BufferUpdateRecorder, which blend
 *        a training trace's running statistics into their buffers on every replay.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/tensor.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/graph/step_recorder.hpp>

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
        /** @brief Holds one running-statistic update as the two tensors it touches. */
        struct BufferUpdateSlot
        {
            /** @brief The buffer the update writes; it shares storage with the model state. */
            SushiBLAS::Tensor target;

            /** @brief The batch value the update reads; it shares storage with the pool. */
            SushiBLAS::Tensor source;

            /** @brief The weight of the source against the buffer's current content. */
            double momentum = 0.0;

            /** @brief The factor the source is scaled by before it is blended in. */
            double source_scale = 1.0;
        };

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
            const Graph::TensorPool& pool);

        /**
         * @brief Records target = (1 - momentum) * target + momentum * source_scale * source
         *        for every slot, as device work behind a compiled step.
         */
        class BufferUpdateRecorder final : public Graph::StepRecorder
        {
            public:
                /**
                 * @brief Takes the slots that every replay blends.
                 * @param slots The slots from bind_buffer_updates(); an empty list records nothing.
                 */
                explicit BufferUpdateRecorder(std::vector<BufferUpdateSlot> slots) noexcept;

                /**
                 * @brief Records two tasks per slot: a scale of the target, then an axpy of the
                 *        source into it.
                 * @return Twice slot_count(), or the failure of a recorded call.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    SushiBLAS::TaskRecorder& recorder) override;

                /** @brief Returns how many slots this recorder blends. */
                [[nodiscard]] std::size_t slot_count() const noexcept;

            private:
                /** @brief The slots, in the order they were given; each replay visits them all. */
                std::vector<BufferUpdateSlot> slots_;
        };

    } // namespace Train
} // namespace SushiAI
