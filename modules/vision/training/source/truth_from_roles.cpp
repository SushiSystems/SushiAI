/**************************************************************************/
/* truth_from_roles.cpp                                                   */
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
 * @file truth_from_roles.cpp
 * @brief Implements the roles a detection step reads and their copy into a ground truth.
 * @author Mustafa Garip
 * @see modules/vision/training/README.md
 */

#include <SushiAI/vision/training/truth_from_roles.hpp>

#include <cstddef>
#include <string>
#include <string_view>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/roles.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds what the tensor of one role must be for the copy to read it. */
        struct RoleForm
        {
            std::string_view role; /**< Name the batch carries the tensor under. */
            int rank;              /**< Axes of the tensor, the batch axis counted. */
            DType dtype;           /**< Element type of the tensor. */
        };

        /** @brief Holds the form of the boxes role: [N, M, 4] float32. */
        constexpr RoleForm K_BOXES_FORM{Data::K_ROLE_BOXES, 3, DType::Float32};

        /** @brief Holds the form of the classes role: [N, M] int32. */
        constexpr RoleForm K_CLASSES_FORM{Data::K_ROLE_CLASSES, 2, DType::Int32};

        /** @brief Holds the form of the valid role: [N, M] float32. */
        constexpr RoleForm K_VALID_FORM{Data::K_ROLE_VALID, 2, DType::Float32};

        /** @brief Returns the extents of @p tensor as "[2, 3, 4]". */
        std::string list_extents(const SushiBLAS::Tensor& tensor)
        {
            std::string text = "[";
            for (int axis = 0; axis < tensor.rank; ++axis)
            {
                text += axis == 0 ? "" : ", ";
                text += std::to_string(tensor.shape[static_cast<std::size_t>(axis)]);
            }
            return text + "]";
        }

        /** @brief Returns the roles of @p batch in quotes, or "(none)". */
        std::string list_roles(const Data::Batch& batch)
        {
            std::string text;
            for (const Data::BatchTarget& target : batch.targets)
            {
                text += text.empty() ? "'" : ", '";
                text += target.role + "'";
            }
            return text.empty() ? std::string("(none)") : text;
        }

        /**
         * @brief Finds the tensor @p batch carries under the role of @p form.
         * @return The tensor; InvalidDescription when the batch lacks the role; Incompatible
         *         for another rank or element type, or values the host cannot read in order.
         */
        Result<const SushiBLAS::Tensor*> find_role(const Data::Batch& batch,
                                                   const RoleForm& form)
        {
            const Data::BatchTarget* const target = Data::find_target(batch.targets, form.role);
            SA_FAIL_IF(target == nullptr, StatusCode::InvalidDescription,
                       "The batch carries no role '{}', which a detection step reads; it "
                       "carries {}",
                       form.role, list_roles(batch));

            const SushiBLAS::Tensor& values = target->values;
            SA_FAIL_IF(values.rank != form.rank || values.dtype != form.dtype,
                       StatusCode::Incompatible,
                       "The role '{}' must hold {} of rank {}, but it holds {} {}", form.role,
                       dtype_name(form.dtype), form.rank, dtype_name(values.dtype),
                       list_extents(values));
            SA_FAIL_IF(values.data_or_null() == nullptr || !values.is_contiguous(),
                       StatusCode::Incompatible,
                       "The role '{}' holds {} whose values the host cannot read in order",
                       form.role, list_extents(values));
            return &values;
        }

        /** @brief Checks that the [N, M] role @p values has the rows of @p boxes. */
        Status check_rows(const SushiBLAS::Tensor& values, std::string_view role,
                          const SushiBLAS::Tensor& boxes)
        {
            SA_FAIL_IF(values.shape[0] != boxes.shape[0] || values.shape[1] != boxes.shape[1],
                       StatusCode::Incompatible,
                       "The role '{}' holds {}, but the role '{}' holds {}; both must hold the "
                       "same images and object slots",
                       role, list_extents(values), Data::K_ROLE_BOXES, list_extents(boxes));
            return Status();
        }

        /** @brief Checks that each of the @p rows values of the valid role is 0 or 1. */
        Status check_flags(const float* flags, std::size_t rows, std::size_t slots)
        {
            for (std::size_t row = 0; row < rows; ++row)
            {
                SA_FAIL_IF(flags[row] != 0.0f && flags[row] != 1.0f,
                           StatusCode::InvalidDescription,
                           "The role '{}' holds {} for slot {} of image {}; it holds 1 for a "
                           "slot in use and 0 for padding",
                           Data::K_ROLE_VALID, flags[row], row % slots, row / slots);
            }
            return Status();
        }
    } // namespace

    Result<std::vector<Data::TargetDescription>> describe_truth_roles(std::int64_t max_objects,
                                                                      std::int64_t classes)
    {
        SA_FAIL_IF(max_objects < 1, StatusCode::InvalidDescription,
                   "A detection step reads at least 1 object slot per image; got {}",
                   max_objects);
        SA_FAIL_IF(classes < 0, StatusCode::InvalidDescription,
                   "A detection step reads classes of a count of 0 or more; got {}", classes);

        return std::vector<Data::TargetDescription>{
            {std::string(K_BOXES_FORM.role), Shape{max_objects, K_TRUTH_BOX_VALUES},
             K_BOXES_FORM.dtype, 0},
            {std::string(K_CLASSES_FORM.role), Shape{max_objects}, K_CLASSES_FORM.dtype, classes},
            {std::string(K_VALID_FORM.role), Shape{max_objects}, K_VALID_FORM.dtype, 0}};
    }

    Status truth_from_roles(const Data::Batch& batch, Detect::GroundTruthBatch& truth)
    {
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const boxes, find_role(batch, K_BOXES_FORM));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const classes, find_role(batch, K_CLASSES_FORM));
        SA_TRY_ASSIGN(const SushiBLAS::Tensor* const valid, find_role(batch, K_VALID_FORM));

        SA_FAIL_IF(boxes->shape[2] != K_TRUTH_BOX_VALUES, StatusCode::Incompatible,
                   "The role '{}' holds {}; its last axis must hold the {} values of a box",
                   Data::K_ROLE_BOXES, list_extents(*boxes), K_TRUTH_BOX_VALUES);
        SA_TRY(check_rows(*classes, Data::K_ROLE_CLASSES, *boxes));
        SA_TRY(check_rows(*valid, Data::K_ROLE_VALID, *boxes));

        const std::size_t slots = static_cast<std::size_t>(boxes->shape[1]);
        const std::size_t rows = static_cast<std::size_t>(boxes->shape[0]) * slots;
        const float* const flags = valid->data_as<float>();
        SA_TRY(check_flags(flags, rows, slots));

        const float* const corners = boxes->data_as<float>();
        const std::int32_t* const indices = classes->data_as<std::int32_t>();

        // assign() keeps the storage of an array that already holds as many values.
        truth.batch = boxes->shape[0];
        truth.max_boxes = boxes->shape[1];
        truth.boxes.assign(corners, corners + rows * static_cast<std::size_t>(K_TRUTH_BOX_VALUES));
        truth.labels.assign(indices, indices + rows);
        truth.valid.resize(rows);
        for (std::size_t row = 0; row < rows; ++row)
            truth.valid[row] = static_cast<std::uint8_t>(flags[row] != 0.0f);

        return Status();
    }

} // namespace SushiAI::Vision::Training
