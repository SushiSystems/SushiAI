/**************************************************************************/
/* fusion_selector.cpp                                                    */
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
 * @file fusion_selector.cpp
 * @brief Implements DeviceFusionSelector policy evaluation for fused operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/fusion_selector.hpp>

namespace SushiAI
{
    namespace Graph
    {
        DeviceFusionSelector DeviceFusionSelector::for_engine(SushiBLAS::Engine& engine)
        {
            // Device profile queries are cached statically across engine bindings.
            return DeviceFusionSelector(
                SushiBLAS::Device::profile_for(engine.get_context().get_queue()),
                engine.get_mode());
        }

        FusionChoice DeviceFusionSelector::choose(const FusionQuery& query) const
        {
            if (query.matched == FusedForm::NONE)
                return FusionChoice{FusedForm::NONE, "nothing was matched"};

            // Fall back to plain kernels when target device lacks native double precision.
            if (query.dtype == DType::FLOAT64 && !profile_.supports_fp64)
                return FusionChoice{FusedForm::NONE,
                                    "the device has no native double precision, so the plain "
                                    "kernels report the problem rather than a fused form the "
                                    "caller never asked for"};

            if (is_gemm_epilogue(query.matched))
            {
                // Reference and verification execution modes execute epilogues on the host.
                if (mode_ == SushiBLAS::ExecutionMode::FORCE_REFERENCE
                    || mode_ == SushiBLAS::ExecutionMode::VERIFY)
                    return FusionChoice{FusedForm::NONE,
                                        "this execution mode runs the epilogue on the host, "
                                        "where an operand in device memory cannot be read"};

                // Fused epilogue access requires a contiguous broadcast operand.
                if (!query.operand_is_contiguous)
                    return FusionChoice{FusedForm::NONE,
                                        "the epilogue's broadcast operand is not contiguous, and "
                                        "the functor indexes it directly"};
            }

            // Accept the matched fusion form when device capabilities and layout invariants hold.
            return FusionChoice{query.matched, "no device fact argues against this form"};
        }

    } // namespace Graph
} // namespace SushiAI
