/**************************************************************************/
/* assigner_config.cpp                                                    */
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
 * @file assigner_config.cpp
 * @brief Implements the two branch presets of the task-aligned label assigner.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/free/assigner_config.hpp>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        namespace
        {
            /** @brief Holds the candidate count of the one-to-many branch. */
            constexpr std::int64_t K_ONE_TO_MANY_TOPK = 10;

            /** @brief Holds the candidate count of the one-to-one branch. */
            constexpr std::int64_t K_ONE_TO_ONE_TOPK = 7;

            /** @brief Holds the candidate count the one-to-one branch keeps after the tie break. */
            constexpr std::int64_t K_ONE_TO_ONE_TOPK2 = 1;

            /** @brief Holds the score exponent the detection loss gives both branches. */
            constexpr float K_LOSS_ALPHA = 0.5f;

            /** @brief Holds the overlap exponent the detection loss gives both branches. */
            constexpr float K_LOSS_BETA = 6.0f;

            /** @brief Holds the stride of the second level when no strides are given. */
            constexpr float K_DEFAULT_STRIDE_VAL = 16.0f;

            /** @brief Returns the settings both branches share, with @p topk and @p topk2. */
            AssignerConfig make_branch_config(std::int64_t topk,
                                              std::int64_t topk2,
                                              std::int64_t num_classes,
                                              SushiRuntime::span<const float> strides)
            {
                SA_THROW_IF(num_classes <= 0, "an assigner needs a positive class count; got {}",
                            num_classes);

                AssignerConfig config;
                config.topk = topk;
                config.topk2 = topk2;
                config.num_classes = num_classes;
                config.alpha = K_LOSS_ALPHA;
                config.beta = K_LOSS_BETA;
                config.stride_val = K_DEFAULT_STRIDE_VAL;
                if (strides.size() > 1)
                    config.stride_val = strides[1];
                else if (strides.size() == 1)
                    config.stride_val = strides[0];
                return config;
            }
        } // namespace

        AssignerConfig one_to_many_assigner_config(std::int64_t num_classes,
                                                   SushiRuntime::span<const float> strides)
        {
            return make_branch_config(K_ONE_TO_MANY_TOPK, K_ONE_TO_MANY_TOPK, num_classes, strides);
        }

        AssignerConfig one_to_one_assigner_config(std::int64_t num_classes,
                                                  SushiRuntime::span<const float> strides)
        {
            return make_branch_config(K_ONE_TO_ONE_TOPK, K_ONE_TO_ONE_TOPK2, num_classes, strides);
        }

    } // namespace Vision::Detect::Free
} // namespace SushiAI
