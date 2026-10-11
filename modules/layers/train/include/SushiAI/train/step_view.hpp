/**************************************************************************/
/* step_view.hpp                                                          */
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
 * @file step_view.hpp
 * @brief Declares what an objective reads of a step that has run.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Pairs the name of a published model output with the tensor a step wrote. */
        struct StepOutput
        {
            std::string name;                          /**< Name the model publishes it under. */
            const SushiBLAS::Tensor* tensor = nullptr; /**< Its tensor; the pool owns it. */
        };

        /** @brief Holds what an objective reads of a step that has run. */
        struct StepView
        {
            const Data::Batch& batch;                     /**< The batch the step trained on. */
            double loss = 0.0;                            /**< The loss the step computed. */
            SushiRuntime::span<const StepOutput> outputs; /**< Outputs, in published order. */
        };

        /**
         * @brief Returns the tensor of a step that published exactly one output.
         * @pre No recorded task writes the tensor.
         * @return The tensor, never null, or InvalidDescription stating how many outputs
         *         the step published.
         */
        [[nodiscard]] Result<const SushiBLAS::Tensor*> only_output(const StepView& step);

    } // namespace Train
} // namespace SushiAI
