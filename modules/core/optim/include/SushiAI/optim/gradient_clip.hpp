/**************************************************************************/
/* gradient_clip.hpp                                                      */
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
 * @file gradient_clip.hpp
 * @brief Declares the clip of a step's gradients to a global norm, recorded as device tasks.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/optim/step_scalars.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @brief Computes, on the device, the factor that brings a step's gradients to a
         *        global norm of at most max_norm().
         *
         * The factor is max_norm / norm when the norm over every gradient is above max_norm,
         * and 1 otherwise. Non-copyable: kernels capture the addresses of its buffers.
         */
        class GradientClip final
        {
            public:
                /**
                 * @brief Allocates the device buffers of a clip over @p gradient_count gradients.
                 * @param dtype The element type of every gradient; Float32 or Float64.
                 * @return The clip, or InvalidDescription for no gradient, another element
                 *         type, or a norm that is not positive and finite; or OutOfMemory.
                 */
                [[nodiscard]] static Result<std::unique_ptr<GradientClip>> make(
                    SushiBLAS::Engine& engine, DType dtype, std::size_t gradient_count,
                    double max_norm);

                ~GradientClip() = default;

                GradientClip(const GradientClip&) = delete;
                GradientClip& operator=(const GradientClip&) = delete;
                GradientClip(GradientClip&&) = delete;
                GradientClip& operator=(GradientClip&&) = delete;

                /**
                 * @brief Records one norm per gradient and the task that folds them into scale().
                 * @param gradients One tensor per gradient, in slot order; a null entry, another
                 *        count or another element type than make() was given is fatal.
                 * @return The number of tasks recorded, gradient_count() + 1, or the seam's
                 *         failure.
                 */
                [[nodiscard]] Result<std::size_t> record(
                    SushiBLAS::TaskRecorder& recorder,
                    const std::vector<SushiBLAS::Tensor*>& gradients);

                /**
                 * @brief Writes the norm and the gradients' scale into device memory.
                 * @param inverse_loss_scale What a stored gradient is multiplied by to give the
                 *        true one; 1 without loss scaling.
                 */
                void before_replay(float inverse_loss_scale);

                /** @brief Returns the one-element factor the updates read. */
                [[nodiscard]] const SushiBLAS::Tensor& scale() const noexcept { return scale_; }

                /** @brief Returns the norm the next replay clips to. */
                [[nodiscard]] double max_norm() const noexcept { return max_norm_; }

                /**
                 * @brief Sets the norm the next replay clips to.
                 * @pre @p max_norm is positive and finite; any other value stops the process.
                 */
                void set_max_norm(double max_norm);

                /**
                 * @brief Reads from the host the factor the last replay applied.
                 * @pre The replay has been waited on; before the first one the factor is 1.
                 */
                [[nodiscard]] double applied_scale() const;

                /** @brief Returns how many gradients the norm is taken over. */
                [[nodiscard]] std::size_t gradient_count() const noexcept
                {
                    return norm_views_.size();
                }

            private:
                /**
                 * @brief Takes ownership of the buffers make() allocated.
                 * @param arena Storage of gradient_count + 1 elements; moved in.
                 * @param norm_views One element per gradient, in slot order; moved in.
                 * @param norms The view over every element of @p norm_views.
                 * @param scale The view of the last element of @p arena.
                 */
                GradientClip(SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                             std::vector<SushiBLAS::Tensor> norm_views, SushiBLAS::Tensor norms,
                             SushiBLAS::Tensor scale, StepScalars scalars,
                             double max_norm) noexcept;

                /// @brief The storage every view below carves from.
                SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena_;

                /// @brief One element per gradient: its Euclidean norm after a replay.
                std::vector<SushiBLAS::Tensor> norm_views_;

                /// @brief Every element of norm_views_ as one tensor, read by the folding task.
                SushiBLAS::Tensor norms_;

                /// @brief The factor the updates multiply their gradient by.
                SushiBLAS::Tensor scale_;

                /// @brief The norm and the inverse loss scale, rewritten before every replay.
                StepScalars scalars_;

                /// @brief The norm the next replay clips to.
                double max_norm_ = 0.0;
        };

    } // namespace Optim
} // namespace SushiAI
