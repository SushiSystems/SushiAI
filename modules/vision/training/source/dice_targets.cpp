/**************************************************************************/
/* dice_targets.cpp                                                       */
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
 * @file dice_targets.cpp
 * @brief Implements the fill of the five leaves of the soft Dice loss.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/dice_targets.hpp>

#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>

#include "host_reals.hpp"
#include "score_indices.hpp"

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the largest index K_GATHER_ROWS addresses a table row by. */
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();

        /** @brief Returns Incompatible unless @p leaf holds @p elements INT32 indices. */
        [[nodiscard]] Status check_index_leaf(const SushiBLAS::Tensor& leaf, std::int64_t elements,
                                              std::string_view name)
        {
            SA_FAIL_IF(leaf.dtype != DType::Int32 || leaf.num_elements != elements,
                       StatusCode::Incompatible,
                       "The step's Dice leaf '{}' holds {} elements of {}; the fill writes {} "
                       "of {}",
                       name, leaf.num_elements, dtype_name(leaf.dtype), elements,
                       dtype_name(DType::Int32));

            return Status();
        }

        /** @brief Returns Incompatible unless @p leaf holds @p elements real numbers. */
        [[nodiscard]] Status check_leaf(const SushiBLAS::Tensor& leaf, std::int64_t elements,
                                        std::string_view name)
        {
            SA_FAIL_IF(!Detail::holds_reals(leaf) || leaf.num_elements != elements,
                       StatusCode::Incompatible,
                       "The step's Dice leaf '{}' holds {} elements of {}; the fill writes {} "
                       "of a real floating type",
                       name, leaf.num_elements, dtype_name(leaf.dtype), elements);

            return Status();
        }

        /** @brief Writes the two indices of every pixel, and a 1 into @p kept unless ignored. */
        void mark_kept_pixels(const std::int32_t* label, std::int64_t pixels, std::int64_t classes,
                              std::int32_t ignore_index, const DiceLeaves& leaves)
        {
            Detail::write_score_indices(label, pixels, classes, ignore_index,
                                        leaves.score_indices->data_as<std::int32_t>());

            // An ignored pixel names class 0, a valid row; its kept of 0 adds nothing there.
            std::int32_t* const class_index = leaves.class_indices->data_as<std::int32_t>();
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
                class_index[pixel] = label[pixel] != ignore_index ? label[pixel] : 0;

            Detail::write_reals(*leaves.kept, [&](auto* const out)
            {
                for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
                    out[pixel] = label[pixel] != ignore_index ? 1 : 0;
            });
        }

        /** @brief Writes the pixel count of each class and the weight of each present class. */
        void write_class_leaves(const std::vector<std::int64_t>& counts, const DiceLeaves& leaves)
        {
            std::int64_t present = 0;
            for (const std::int64_t count : counts)
                present += count > 0 ? 1 : 0;

            const double share = present == 0 ? 0.0 : 1.0 / static_cast<double>(present);
            Detail::write_reals(*leaves.class_pixels, [&counts](auto* const out)
            {
                using Real = std::remove_reference_t<decltype(*out)>;
                for (std::size_t index = 0; index < counts.size(); ++index)
                    out[index] = static_cast<Real>(counts[index]);
            });
            Detail::write_reals(*leaves.class_weights, [&counts, share](auto* const out)
            {
                using Real = std::remove_reference_t<decltype(*out)>;
                for (std::size_t index = 0; index < counts.size(); ++index)
                    out[index] = static_cast<Real>(counts[index] > 0 ? share : 0.0);
            });
        }
    } // namespace

    Status fill_dice_targets(const SushiBLAS::Tensor& mask, std::int64_t classes,
                             std::int32_t ignore_index, const DiceLeaves& leaves)
    {
        SA_FATAL_IF(leaves.score_indices == nullptr || leaves.class_indices == nullptr
                    || leaves.kept == nullptr || leaves.class_pixels == nullptr
                    || leaves.class_weights == nullptr,
                    "fill_dice_targets was given a null leaf");
        SA_FAIL_IF(classes < 1, StatusCode::Incompatible,
                   "A Dice step counts the pixels of at least one class; got {}", classes);

        const std::int64_t pixels = leaves.kept->num_elements;
        SA_FAIL_IF(pixels > K_LARGEST_INDEX / classes, StatusCode::Incompatible,
                   "{} pixels over {} classes cannot be indexed in {}", pixels, classes,
                   dtype_name(DType::Int32));
        SA_TRY(check_leaf(*leaves.kept, pixels, "kept"));
        SA_TRY(check_index_leaf(*leaves.score_indices, pixels, "score_indices"));
        SA_TRY(check_index_leaf(*leaves.class_indices, pixels, "class_indices"));
        SA_TRY(check_leaf(*leaves.class_pixels, classes, "class_pixels"));
        SA_TRY(check_leaf(*leaves.class_weights, classes, "class_weights"));
        SA_TRY(validate_pixel_labels(mask, pixels, classes, ignore_index));

        const std::int32_t* const label = mask.data_as<std::int32_t>();
        SA_TRY_ASSIGN(const std::vector<std::int64_t> counts,
                      SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            std::vector<std::int64_t> pixels_of(static_cast<std::size_t>(classes), 0);
            for (std::int64_t pixel = 0; pixel < pixels; ++pixel)
            {
                if (label[pixel] != ignore_index)
                    ++pixels_of[static_cast<std::size_t>(label[pixel])];
            }

            return pixels_of;
        }));

        // Every label and every leaf was checked, so a refused batch leaves the leaves alone.
        mark_kept_pixels(label, pixels, classes, ignore_index, leaves);
        write_class_leaves(counts, leaves);

        return Status();
    }

} // namespace SushiAI::Vision::Training
