/**************************************************************************/
/* detection_collator.cpp                                                 */
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
 * @file detection_collator.cpp
 * @brief Implements the collator of the detection task.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/detection_collator.hpp>

#include <algorithm>
#include <cstddef>
#include <string>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>

#include "collate_support.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "DetectionCollator";
    } // namespace

    Result<std::unique_ptr<DetectionCollator>> DetectionCollator::make(std::int32_t class_count,
                                                                       std::int32_t max_objects)
    {
        SA_FAIL_IF(class_count < 1, StatusCode::InvalidDescription,
                   "{}: {} classes; there must be at least 1", K_COLLATOR, class_count);
        SA_FAIL_IF(max_objects < 1, StatusCode::InvalidDescription,
                   "{}: {} objects per image; there must be room for at least 1", K_COLLATOR,
                   max_objects);

        return Detail::at_boundary(
            K_COLLATOR, [&]() -> Result<std::unique_ptr<DetectionCollator>>
        {
            return std::unique_ptr<DetectionCollator>(
                new DetectionCollator(class_count, max_objects));
        });
    }

    DetectionCollator::DetectionCollator(std::int32_t class_count, std::int32_t max_objects)
        : class_count_(class_count), max_objects_(max_objects),
          targets_{
              Data::TargetDescription{std::string(Data::K_ROLE_BOXES),
                                      Shape{max_objects, K_BOX_COORDINATES}, DType::Float32, 0},
              Data::TargetDescription{std::string(Data::K_ROLE_CLASSES), Shape{max_objects},
                                      DType::Int32, class_count},
              Data::TargetDescription{std::string(Data::K_ROLE_VALID), Shape{max_objects},
                                      DType::Float32, 0}}
    {

    }

    std::string_view DetectionCollator::task() const noexcept
    {
        return K_TASK_DETECTION;
    }

    const std::vector<Data::TargetDescription>& DetectionCollator::targets() const noexcept
    {
        return targets_;
    }

    Status DetectionCollator::collate(const SampleList& list,
                                      SushiRuntime::span<const FittedSample> samples,
                                      Data::Batch& batch) const
    {
        const std::size_t slots = static_cast<std::size_t>(max_objects_);
        const std::int64_t rows = static_cast<std::int64_t>(samples.size()) * max_objects_;

        SA_TRY(write_batch_inputs(samples, batch.inputs));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const boxes,
                      Detail::role_tensor(batch, Data::K_ROLE_BOXES, DType::Float32,
                                          rows * K_BOX_COORDINATES, K_COLLATOR));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const classes,
                      Detail::role_tensor(batch, Data::K_ROLE_CLASSES, DType::Int32, rows,
                                          K_COLLATOR));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const valid,
                      Detail::role_tensor(batch, Data::K_ROLE_VALID, DType::Float32, rows,
                                          K_COLLATOR));

        float* box_out = boxes->data_as<float>();
        std::int32_t* class_out = classes->data_as<std::int32_t>();
        float* valid_out = valid->data_as<float>();

        std::fill_n(box_out, static_cast<std::size_t>(rows * K_BOX_COORDINATES), 0.0f);
        std::fill_n(class_out, static_cast<std::size_t>(rows), 0);
        std::fill_n(valid_out, static_cast<std::size_t>(rows), 0.0f);

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const Imaging::Labels& objects = samples[i].objects;

            SA_FAIL_IF(objects.size() > slots, StatusCode::InvalidDescription,
                       "'{}': {}: the image has {} objects; a batch holds at most {} per image",
                       Detail::file_of(list, samples[i]), K_COLLATOR, objects.size(), slots);

            for (std::size_t m = 0; m < objects.size(); ++m)
            {
                const std::int32_t class_id = objects.classes()[m];
                const Imaging::PixelBox& box = objects.boxes()[m];
                const std::size_t slot = i * slots + m;
                float* corners = box_out + slot * static_cast<std::size_t>(K_BOX_COORDINATES);

                SA_FAIL_IF(class_id < 0 || class_id >= class_count_,
                           StatusCode::InvalidDescription,
                           "'{}': {}: object {} has the class {}, outside [0, {})",
                           Detail::file_of(list, samples[i]), K_COLLATOR, m, class_id,
                           class_count_);

                corners[0] = box.x1;
                corners[1] = box.y1;
                corners[2] = box.x2;
                corners[3] = box.y2;
                class_out[slot] = class_id;
                valid_out[slot] = 1.0f;
            }
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
