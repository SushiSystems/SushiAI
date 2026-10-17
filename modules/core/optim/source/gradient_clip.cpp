/**************************************************************************/
/* gradient_clip.cpp                                                      */
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
 * @file gradient_clip.cpp
 * @brief Implements the clip of a step's gradients to a global norm.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/gradient_clip.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <SushiBLAS/elementwise/dispatch.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /** @brief The norm the gradients are clipped to. */
            constexpr std::size_t K_MAX_NORM = 0;

            /** @brief 1 / the gradient loss scale; exactly 1 when unscaled. */
            constexpr std::size_t K_INVERSE_LOSS_SCALE = 1;

            /** @brief Holds the number of per-step scalars in the device buffer. */
            constexpr std::size_t K_SCALAR_COUNT = 2;

            /** @brief Names the task that folds the norms into the factor. */
            constexpr const char* K_TASK_NAME = "sushiai.optim.clip_scale";

            static_assert(Ops::K_CLIP_SCALE.name == std::string_view(K_TASK_NAME),
                          "the folding task's name must be the string its OpID hashes");

            /** @brief Reports whether @p max_norm is a norm a gradient can be clipped to. */
            [[nodiscard]] bool is_usable_norm(double max_norm) noexcept
            {
                // The comparison is false for a NaN, which is refused with the other values.
                return max_norm > 0.0 && std::isfinite(max_norm);
            }

            /** @brief Writes @p value into the one Float32 or Float64 element of @p tensor. */
            void write_element(SushiBLAS::Tensor& tensor, double value) noexcept
            {
                if (tensor.dtype == DType::Float64)
                    tensor.data_as<double>()[0] = value;
                else
                    tensor.data_as<float>()[0] = static_cast<float>(value);
            }

            /**
             * @brief Carves one element per gradient out of @p arena, in slot order.
             * @param arena Storage of at least @p gradient_count elements of @p dtype.
             * @return The views, or the failure of the allocation seam.
             */
            [[nodiscard]] Result<std::vector<SushiBLAS::Tensor>> carve_norm_views(
                SushiBLAS::Engine& engine,
                const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena, DType dtype,
                std::size_t gradient_count)
            {
                return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
                {
                    std::vector<SushiBLAS::Tensor> views;
                    views.reserve(gradient_count);
                    for (std::size_t i = 0; i < gradient_count; ++i)
                        views.push_back(view_tensor(engine, arena, i * dtype_size(dtype),
                                                    Shape{1}, dtype));
                    return views;
                });
            }
        } // namespace

        Result<std::unique_ptr<GradientClip>> GradientClip::make(SushiBLAS::Engine& engine,
                                                                 DType dtype,
                                                                 std::size_t gradient_count,
                                                                 double max_norm)
        {
            SA_FAIL_IF(gradient_count == 0, StatusCode::InvalidDescription,
                       "A gradient clip needs at least one gradient to take a norm over");
            SA_FAIL_IF(dtype != DType::Float32 && dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "A gradient clip folds float32 or float64 norms; the gradients are {}",
                       dtype_name(dtype));
            SA_FAIL_IF(!is_usable_norm(max_norm), StatusCode::InvalidDescription,
                       "A clip norm must be positive and finite; got {}", max_norm);

            const std::size_t element = dtype_size(dtype);
            SA_TRY_ASSIGN(auto arena, allocate_arena(engine, (gradient_count + 1) * element));
            SA_TRY_ASSIGN(auto norm_views,
                          carve_norm_views(engine, arena, dtype, gradient_count));
            SA_TRY_ASSIGN(StepScalars scalars, StepScalars::make(engine, K_SCALAR_COUNT));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                SushiBLAS::Tensor norms = view_tensor(
                    engine, arena, 0, Shape{static_cast<int64_t>(gradient_count)}, dtype);
                SushiBLAS::Tensor scale =
                    view_tensor(engine, arena, gradient_count * element, Shape{1}, dtype);

                // The arena is shared device memory; the host fills it before the first replay.
                for (SushiBLAS::Tensor& view : norm_views)
                    write_element(view, 0.0);
                write_element(scale, 1.0);

                return std::unique_ptr<GradientClip>(
                    new GradientClip(std::move(arena), std::move(norm_views), std::move(norms),
                                     std::move(scale), std::move(scalars), max_norm));
            });
        }

        GradientClip::GradientClip(SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                                   std::vector<SushiBLAS::Tensor> norm_views,
                                   SushiBLAS::Tensor norms, SushiBLAS::Tensor scale,
                                   StepScalars scalars, double max_norm) noexcept
            : arena_(std::move(arena)), norm_views_(std::move(norm_views)),
              norms_(std::move(norms)), scale_(std::move(scale)), scalars_(std::move(scalars)),
              max_norm_(max_norm)
        {

        }

        Result<std::size_t> GradientClip::record(SushiBLAS::TaskRecorder& recorder,
                                                 const std::vector<SushiBLAS::Tensor*>& gradients)
        {
            SA_FATAL_IF(gradients.size() != norm_views_.size(),
                        "This gradient clip takes a norm over {} gradients but was handed {}",
                        norm_views_.size(), gradients.size());

            return Detail::guard_value(Detail::SeamPhase::Record, [&]()
            {
                SushiBLAS::ReductionOps reductions(recorder);
                for (std::size_t i = 0; i < norm_views_.size(); ++i)
                {
                    SA_FATAL_IF(gradients[i] == nullptr,
                                "Gradient {} is null; every parameter must have one to clip", i);
                    SA_FATAL_IF(gradients[i]->dtype != scale_.dtype,
                                "Gradient {} is {} but this clip was made for {} gradients", i,
                                dtype_name(gradients[i]->dtype), dtype_name(scale_.dtype));

                    reductions.nrm2(*gradients[i], norm_views_[i]);
                }

                // The kernel reads the scalars at execution, so a replay sees the last write.
                const float* const s = scalars_.device_pointer();
                const std::size_t count = norm_views_.size();

                SushiBLAS::Dispatch::execute_unary_inplace(
                    recorder, scale_, K_TASK_NAME, Ops::K_CLIP_SCALE.id,
                    SushiBLAS::Dispatch::bind_reads(
                        [s, count](auto previous, auto norms)
                        {
                            using T = std::decay_t<decltype(previous)>;

                            T squares = static_cast<T>(0);
                            for (std::size_t k = 0; k < count; ++k)
                                squares = squares + norms[k] * norms[k];

                            const T norm = SushiBLAS::Dispatch::safe_sqrt(squares)
                                         * static_cast<T>(s[K_INVERSE_LOSS_SCALE]);
                            const T limit = static_cast<T>(s[K_MAX_NORM]);

                            // A norm that is not a number compares false and leaves the factor 1.
                            return norm > limit ? limit / norm : static_cast<T>(1);
                        },
                        SushiBLAS::Dispatch::read_as_element(norms_)));

                return count + 1;
            });
        }

        void GradientClip::before_replay(float inverse_loss_scale)
        {
            std::array<float, K_SCALAR_COUNT> values{};
            values[K_MAX_NORM] = static_cast<float>(max_norm_);
            values[K_INVERSE_LOSS_SCALE] = inverse_loss_scale;

            scalars_.write(SushiRuntime::span<const float>(values.data(), values.size()));
        }

        void GradientClip::set_max_norm(double max_norm)
        {
            SA_FATAL_IF(!is_usable_norm(max_norm),
                        "A clip norm of {} is not positive and finite", max_norm);

            max_norm_ = max_norm;
        }

        double GradientClip::applied_scale() const
        {
            return scale_.dtype == DType::Float64
                 ? scale_.data_as<double>()[0]
                 : static_cast<double>(scale_.data_as<float>()[0]);
        }

    } // namespace Optim
} // namespace SushiAI
