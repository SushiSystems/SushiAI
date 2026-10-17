/**************************************************************************/
/* step_view.cpp                                                          */
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
 * @file step_view.cpp
 * @brief Implements the lookup of the one output of a step that has run.
 * @author Mustafa Garip
 */

#include <SushiAI/train/step_view.hpp>

#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<const SushiBLAS::Tensor*> only_output(const StepView& step)
        {
            SA_FAIL_IF(step.outputs.size() != 1, StatusCode::InvalidDescription,
                       "The metric reads one model output and the step published {}",
                       step.outputs.size());
            SA_FAIL_IF(step.outputs[0].tensor == nullptr, StatusCode::InvalidDescription,
                       "The output '{}' of the step has no tensor", step.outputs[0].name);

            return step.outputs[0].tensor;
        }

    } // namespace Train
} // namespace SushiAI
