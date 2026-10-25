/**************************************************************************/
/* free_bias_init.cpp                                                     */
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
 * @file free_bias_init.cpp
 * @brief Implements the stride-derived start of an anchor-free head's last biases.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_bias_init.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the value the last box convolution's bias starts from. */
        constexpr double K_BOX_BIAS = 1.0;

        /** @brief Holds the expected object count per image the class bias assumes. */
        constexpr double K_OBJECTS_PER_IMAGE = 5.0;

        /** @brief Holds the image side, in pixels, the class bias assumes. */
        constexpr double K_REFERENCE_IMAGE_SIDE = 640.0;

        /** @brief Holds the box sides each position predicts a distribution for. */
        constexpr std::int64_t K_BOX_SIDES = 4;

        /** @brief Holds one validated write: a bias tensor and the value it starts from. */
        struct BiasWrite
        {
            SushiBLAS::Tensor* tensor; /**< The bias, float32 of the expected channels. */
            float value;               /**< The value every element is set to. */
        };

        /** @brief Returns "<prefix>.<branch>.<level>.2.bias", or without a prefix when empty. */
        std::string bias_name(std::string_view prefix, std::string_view branch, std::size_t level)
        {
            std::string name;
            if (!prefix.empty())
                name.append(prefix).append(".");
            name.append(branch).append(".").append(std::to_string(level)).append(".2.bias");
            return name;
        }

        /** @brief Checks entry @p name of @p state and records the write of @p value. */
        Status plan_write(NN::ModelState& state, const std::string& name, std::int64_t channels,
                          double value, std::vector<BiasWrite>& writes)
        {
            const Result<SushiBLAS::Tensor*> found = state.tensor(name);
            SA_FAIL_IF(!found.has_value(), StatusCode::InvalidDescription,
                       "the model state has no entry '{}', which the detect head's last "
                       "convolution declares", name);
            SushiBLAS::Tensor* const tensor = found.value();

            SA_FAIL_IF(tensor->dtype != SushiBLAS::Core::DataType::Float32,
                       StatusCode::InvalidDescription, "'{}' is not float32", name);
            const Shape actual = shape_of(*tensor);
            SA_FAIL_IF(actual != Shape{channels}, StatusCode::InvalidDescription,
                       "'{}' is {}, but the traced outputs give [{}]", name, actual.to_string(),
                       channels);
            SA_FAIL_IF(tensor->data_or_null() == nullptr, StatusCode::InvalidDescription,
                       "'{}' has no storage to receive its start", name);

            writes.push_back(BiasWrite{tensor, static_cast<float>(value)});
            return Status();
        }

        /** @brief Records the box and class writes of one copy's branches at @p level. */
        Status plan_copy(NN::ModelState& state, const FreeModelFacts& facts,
                         std::string_view prefix, std::string_view box_branch,
                         std::string_view class_branch, std::size_t level,
                         std::vector<BiasWrite>& writes)
        {
            SA_TRY(plan_write(state, bias_name(prefix, box_branch, level),
                              K_BOX_SIDES * facts.reg_max, K_BOX_BIAS, writes));
            return plan_write(state, bias_name(prefix, class_branch, level), facts.classes,
                              free_class_bias(facts.classes, facts.levels[level].stride),
                              writes);
        }
    } // namespace

    double free_class_bias(std::int64_t classes, std::int64_t stride)
    {
        const double cells = K_REFERENCE_IMAGE_SIDE / static_cast<double>(stride);
        return std::log(K_OBJECTS_PER_IMAGE / static_cast<double>(classes) / (cells * cells));
    }

    Status initialize_free_biases(const FreeModelFacts& facts, std::string_view head_prefix,
                                  NN::ModelState& state)
    {
        std::vector<BiasWrite> writes;
        for (std::size_t level = 0; level < facts.levels.size(); ++level)
        {
            SA_TRY(plan_copy(state, facts, head_prefix, "cv2", "cv3", level, writes));
            if (facts.has_one_to_one)
            {
                SA_TRY(plan_copy(state, facts, head_prefix, "one2one_cv2", "one2one_cv3", level,
                                 writes));
            }
        }

        // Every entry was checked above, so the writes below change all of them or none.
        for (const BiasWrite& write : writes)
        {
            float* const data = write.tensor->data_as<float>();
            std::fill(data, data + write.tensor->num_elements, write.value);
        }
        return Status();
    }

} // namespace SushiAI::Vision::Training
