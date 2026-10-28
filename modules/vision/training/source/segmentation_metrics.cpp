/**************************************************************************/
/* segmentation_metrics.cpp                                               */
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
 * @file segmentation_metrics.cpp
 * @brief Implements the metrics a per-pixel objective reports of a step that has run.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/segmentation_metrics.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/host_values.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>

#include "host_reals.hpp"

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds how many pixels are picked and counted at a time. */
        constexpr std::size_t K_PIXELS_PER_PASS = 4096;

        /**
         * @brief Adds every pixel of a step to @p matrix, K_PIXELS_PER_PASS at a time.
         * @tparam Real float or double, the element type of the logits.
         * @param scores The C logits of each pixel; @p truth The class of each pixel.
         * @return What Segment::ConfusionMatrix::add refuses.
         */
        template<typename Real>
        [[nodiscard]] Status count_pixels(const Real* scores, const std::int32_t* truth,
                                          std::size_t pixels, std::int32_t ignore_index,
                                          Segment::ConfusionMatrix& matrix)
        {
            const std::size_t classes = static_cast<std::size_t>(matrix.class_count());
            std::array<std::int32_t, K_PIXELS_PER_PASS> picked{};
            for (std::size_t first = 0; first < pixels; first += K_PIXELS_PER_PASS)
            {
                const std::size_t count = std::min(K_PIXELS_PER_PASS, pixels - first);
                const SushiRuntime::span<std::int32_t> classes_picked(picked.data(), count);
                Segment::argmax_classes(
                    SushiRuntime::span<const Real>(scores + first * classes, count * classes),
                    matrix.class_count(), classes_picked);
                SA_TRY(matrix.add(
                    SushiRuntime::span<const std::int32_t>(picked.data(), count),
                    SushiRuntime::span<const std::int32_t>(truth + first, count), ignore_index));
            }

            return Status();
        }
    } // namespace

    Status measure_segmentation(const Train::StepView& step, const PixelExtents& extents,
                                std::int32_t ignore_index, Train::Metrics& metrics) noexcept
    {
        const std::int64_t pixels = extents.pixels();
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const logits, Train::only_output(step));
        SA_TRY(Train::require_host_reals(*logits, pixels * extents.classes, "the logits"));
        SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                      Train::require_batch_role(step.batch, Data::K_ROLE_MASK));
        SA_FAIL_IF(mask->values.dtype != DType::Int32 || mask->values.num_elements != pixels,
                   StatusCode::Incompatible,
                   "The batch's mask holds {} elements of {}; the step's metrics read {} of {}",
                   mask->values.num_elements, dtype_name(mask->values.dtype), pixels,
                   dtype_name(DType::Int32));
        SA_FAIL_IF(extents.classes > Segment::K_MAX_CONFUSION_CLASSES,
                   StatusCode::InvalidDescription,
                   "The step scores {} classes; its metrics count at most {}", extents.classes,
                   Segment::K_MAX_CONFUSION_CLASSES);

        SA_TRY_ASSIGN(Segment::ConfusionMatrix matrix,
                      Segment::ConfusionMatrix::make(static_cast<std::int32_t>(extents.classes)));
        const std::int32_t* const truth = mask->values.data_as<std::int32_t>();
        SA_TRY(Detail::read_reals(*logits, [&](const auto* const scores)
        {
            return count_pixels(scores, truth, static_cast<std::size_t>(pixels), ignore_index,
                                matrix);
        }));

        SA_TRY(metrics.add(Train::Metric{K_MEAN_IOU_METRIC, Segment::mean_iou(matrix), true}));

        return metrics.add(Train::Metric{K_PIXEL_ACCURACY_METRIC,
                                         Segment::pixel_accuracy(matrix), true});
    }

} // namespace SushiAI::Vision::Training
