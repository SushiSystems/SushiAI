/**************************************************************************/
/* segmentation_collator.cpp                                              */
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
 * @file segmentation_collator.cpp
 * @brief Implements the collator of the semantic segmentation task.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/segmentation_collator.hpp>

#include <cstddef>
#include <string>

#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/polygon_mask.hpp>

#include "collate_support.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "SegmentationCollator";

        /** @brief Holds K_MASK_VOID as the type a class index and a message hold it in. */
        constexpr std::int32_t K_VOID = K_MASK_VOID;
    } // namespace

    Result<std::unique_ptr<SegmentationCollator>> SegmentationCollator::make(
        const SegmentationOptions& options)
    {
        SA_FAIL_IF(options.class_count < 1 || options.class_count > K_VOID,
                   StatusCode::InvalidDescription,
                   "{}: {} classes; a mask of bytes names 1 to {}", K_COLLATOR,
                   options.class_count, K_VOID);
        SA_FAIL_IF(!Imaging::is_valid_size(options.size), StatusCode::InvalidDescription,
                   "{}: an input of {} by {} pixels is outside the bounds of an image",
                   K_COLLATOR, options.size.height, options.size.width);
        SA_FAIL_IF(options.ignore_index >= 0 && options.ignore_index < options.class_count,
                   StatusCode::InvalidDescription,
                   "{}: the ignore index {} is one of the {} classes", K_COLLATOR,
                   options.ignore_index, options.class_count);
        SA_FAIL_IF(options.background.has_value() && options.origin == MaskOrigin::File,
                   StatusCode::InvalidDescription,
                   "{}: a background of {} is stated, and the masks come from files, which "
                   "name the class of every pixel",
                   K_COLLATOR, options.background.value_or(0));
        SA_FAIL_IF(options.background.has_value() &&
                       (*options.background < 0 || *options.background >= options.class_count),
                   StatusCode::InvalidDescription,
                   "{}: the background {} is none of the {} classes", K_COLLATOR,
                   options.background.value_or(0), options.class_count);

        return Detail::at_boundary(
            K_COLLATOR, [&]() -> Result<std::unique_ptr<SegmentationCollator>>
        {
            return std::unique_ptr<SegmentationCollator>(new SegmentationCollator(options));
        });
    }

    SegmentationCollator::SegmentationCollator(const SegmentationOptions& options)
        : options_(options),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_MASK),
                                           Shape{options.size.height, options.size.width},
                                           DType::Int32, options.class_count}}
    {

    }

    std::string_view SegmentationCollator::task() const noexcept
    {
        return K_TASK_SEGMENTATION;
    }

    const std::vector<Data::TargetDescription>& SegmentationCollator::targets() const noexcept
    {
        return targets_;
    }

    Status SegmentationCollator::collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const
    {
        const std::size_t width = static_cast<std::size_t>(options_.size.width);
        const std::size_t pixels = static_cast<std::size_t>(options_.size.height) * width;

        SA_TRY(write_batch_inputs(samples, batch.inputs));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const mask,
                      Detail::role_tensor(batch, Data::K_ROLE_MASK, DType::Int32,
                                          static_cast<std::int64_t>(samples.size() * pixels),
                                          K_COLLATOR));

        std::int32_t* out = mask->data_as<std::int32_t>();
        const std::uint8_t background =
            static_cast<std::uint8_t>(options_.background.value_or(K_VOID));
        Imaging::Image painted;

        for (std::size_t i = 0; i < samples.size(); ++i, out += pixels)
        {
            if (options_.origin == MaskOrigin::Polygons)
                SA_TRY(SA_ANNOTATE(paint_polygon_mask(samples[i].objects, options_.size,
                                                      background, painted),
                                   "'{}'", Detail::file_of(list, samples[i])));

            const Imaging::Image& fitted =
                options_.origin == MaskOrigin::Polygons ? painted : samples[i].mask;

            SA_FAIL_IF(fitted.size() != options_.size || fitted.channels() != 1,
                       StatusCode::InvalidDescription,
                       "'{}': {}: the sample's mask is {} by {} of {} channels; the task reads "
                       "{} by {} of 1",
                       Detail::file_of(list, samples[i]), K_COLLATOR, fitted.height(),
                       fitted.width(), fitted.channels(), options_.size.height,
                       options_.size.width);

            // A one-channel image has no padding between rows, so its bytes are row-major.
            const std::uint8_t* const bytes = fitted.data();

            for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            {
                const std::int32_t value = bytes[pixel];

                SA_FAIL_IF(value >= options_.class_count && value != K_VOID,
                           StatusCode::InvalidDescription,
                           "'{}': {}: the fitted mask holds {} at row {}, column {}; a class is "
                           "below {} and {} names none",
                           Detail::file_of(list, samples[i]), K_COLLATOR, value, pixel / width,
                           pixel % width, options_.class_count, K_VOID);

                out[pixel] = value == K_VOID ? options_.ignore_index : value;
            }
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
