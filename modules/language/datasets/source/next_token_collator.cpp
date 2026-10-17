/**************************************************************************/
/* next_token_collator.cpp                                                */
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
 * @file next_token_collator.cpp
 * @brief Implements the collator of the next-token task.
 * @author Mustafa Garip
 */

#include <SushiAI/language/datasets/next_token_collator.hpp>

#include <cstddef>
#include <string>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>

#include "disk.hpp"

namespace SushiAI::Language::Datasets
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_COLLATOR = "NextTokenCollator";

        /** @brief Fails unless a tensor of a batch holds @p elements int32 values. */
        [[nodiscard]] Status check_ids_tensor(const SushiBLAS::Tensor& tensor,
                                              std::string_view what, std::int64_t elements)
        {
            SA_FAIL_IF(tensor.dtype != DType::Int32 || tensor.num_elements != elements,
                       StatusCode::Incompatible,
                       "{}: the {} of the batch hold {} elements; the collator writes {} int32 "
                       "ids",
                       K_COLLATOR, what, tensor.num_elements, elements);
            return Status();
        }
    } // namespace

    Result<std::unique_ptr<NextTokenCollator>> NextTokenCollator::make(
        std::int64_t window_length, std::int32_t vocabulary_size)
    {
        SA_FAIL_IF(window_length < 1 || window_length > K_MAX_WINDOW_TOKENS - 1,
                   StatusCode::InvalidDescription,
                   "{}: a window of {} positions; it has 1 to {}", K_COLLATOR, window_length,
                   K_MAX_WINDOW_TOKENS - 1);
        SA_FAIL_IF(vocabulary_size < 1 || vocabulary_size > Text::K_MAX_VOCABULARY_SIZE,
                   StatusCode::InvalidDescription,
                   "{}: a vocabulary of {} tokens; it has 1 to {}", K_COLLATOR, vocabulary_size,
                   Text::K_MAX_VOCABULARY_SIZE);

        return Detail::at_boundary(K_COLLATOR, [&]() -> Result<std::unique_ptr<NextTokenCollator>>
        {
            return std::unique_ptr<NextTokenCollator>(
                new NextTokenCollator(window_length, vocabulary_size));
        });
    }

    NextTokenCollator::NextTokenCollator(std::int64_t window_length, std::int32_t vocabulary_size)
        : window_length_(window_length), vocabulary_size_(vocabulary_size),
          targets_{Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{window_length},
                                           DType::Int32, vocabulary_size}}
    {

    }

    std::string_view NextTokenCollator::task() const noexcept
    {
        return K_TASK_NEXT_TOKEN;
    }

    std::int64_t NextTokenCollator::window_length() const noexcept
    {
        return window_length_;
    }

    std::int64_t NextTokenCollator::tokens_per_window() const noexcept
    {
        return window_length_ + 1;
    }

    std::int32_t NextTokenCollator::vocabulary_size() const noexcept
    {
        return vocabulary_size_;
    }

    const std::vector<Data::TargetDescription>& NextTokenCollator::targets() const noexcept
    {
        return targets_;
    }

    Status NextTokenCollator::collate(SushiRuntime::span<const Text::TokenID> tokens,
                                      Data::Batch& batch) const
    {
        const std::size_t length = static_cast<std::size_t>(window_length_);
        const std::size_t per_window = length + 1;

        SA_FAIL_IF(tokens.empty() || tokens.size() % per_window != 0, StatusCode::Incompatible,
                   "{}: {} ids are no whole number of windows of {}", K_COLLATOR, tokens.size(),
                   per_window);

        const std::size_t rows = tokens.size() / per_window;
        const std::int64_t elements = static_cast<std::int64_t>(rows * length);

        SA_FAIL_IF(batch.targets.size() != 1 || batch.targets[0].role != Data::K_ROLE_LABELS,
                   StatusCode::Incompatible, "{}: the batch does not hold the one role '{}'",
                   K_COLLATOR, Data::K_ROLE_LABELS);
        SA_TRY(check_ids_tensor(batch.inputs, "inputs", elements));
        SA_TRY(check_ids_tensor(batch.targets[0].values, "labels", elements));

        Text::TokenID* const inputs = batch.inputs.data_as<Text::TokenID>();
        Text::TokenID* const labels = batch.targets[0].values.data_as<Text::TokenID>();

        for (std::size_t row = 0; row < rows; ++row)
        {
            const Text::TokenID* const window = tokens.data() + row * per_window;

            for (std::size_t t = 0; t < length; ++t)
            {
                inputs[row * length + t] = window[t];
                labels[row * length + t] = window[t + 1];
            }
        }

        return Status();
    }
} // namespace SushiAI::Language::Datasets
