/**************************************************************************/
/* multi_label_collator.cpp                                               */
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
 * @file multi_label_collator.cpp
 * @brief Implements the collator of the multi-label task.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/multi_label_collator.hpp>

#include <algorithm>
#include <cstddef>
#include <string>

#include <SushiBLAS/core/format.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>

#include "collate_support.hpp"
#include "disk.hpp"

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "MultiLabelCollator";
    } // namespace

    Result<std::unique_ptr<MultiLabelCollator>> MultiLabelCollator::make(std::int32_t class_count)
    {
        SA_FAIL_IF(class_count < 1, StatusCode::InvalidDescription,
                   "{}: {} classes; there must be at least 1", K_COLLATOR, class_count);

        return Detail::at_boundary(
            K_COLLATOR, [&]() -> Result<std::unique_ptr<MultiLabelCollator>>
        {
            return std::unique_ptr<MultiLabelCollator>(new MultiLabelCollator(class_count));
        });
    }

    MultiLabelCollator::MultiLabelCollator(std::int32_t class_count)
        : class_count_(class_count),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{class_count},
                                           DType::Float32, 0}}
    {

    }

    std::string_view MultiLabelCollator::task() const noexcept
    {
        return K_TASK_MULTI_LABEL;
    }

    const std::vector<Data::TargetDescription>& MultiLabelCollator::targets() const noexcept
    {
        return targets_;
    }

    Status MultiLabelCollator::collate(const SampleList& list,
                                       SushiRuntime::span<const FittedSample> samples,
                                       Data::Batch& batch) const
    {
        const std::size_t classes = static_cast<std::size_t>(class_count_);
        const std::int64_t flags = static_cast<std::int64_t>(samples.size()) * class_count_;

        SA_TRY(write_batch_inputs(samples, batch.inputs));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const labels,
                      Detail::role_tensor(batch, Data::K_ROLE_LABELS, DType::Float32, flags,
                                          K_COLLATOR));

        float* out = labels->data_as<float>();
        std::fill_n(out, static_cast<std::size_t>(flags), 0.0f);

        for (std::size_t i = 0; i < samples.size(); ++i)
        {
            const FittedSample& sample = samples[i];
            float* const row = out + i * classes;

            // Entry objects.size() stands for the class of the whole image, when it has one.
            for (std::size_t m = 0; m <= sample.objects.size(); ++m)
            {
                const bool is_object = m < sample.objects.size();
                if (!is_object && !sample.class_index.has_value())
                    continue;

                const std::int32_t class_id =
                    is_object ? sample.objects.classes()[m] : *sample.class_index;

                SA_FAIL_IF(class_id < 0 || class_id >= class_count_,
                           StatusCode::InvalidDescription,
                           "'{}': {}: the sample has the class {}, outside [0, {})",
                           Detail::file_of(list, sample), K_COLLATOR, class_id, class_count_);

                row[class_id] = 1.0f;
            }
        }

        return Status();
    }
} // namespace SushiAI::Vision::Datasets
