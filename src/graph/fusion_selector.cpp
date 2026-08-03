/**************************************************************************/
/* fusion_selector.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <SushiAI/graph/fusion_selector.hpp>

namespace SushiAI
{
    namespace Graph
    {
        DeviceFusionSelector DeviceFusionSelector::for_engine(SushiBLAS::Engine& engine)
        {
            // profile_for caches per device for the life of the process, so this
            // is not eight `get_info` round trips per bind - it is eight once.
            return DeviceFusionSelector(
                SushiBLAS::Device::profile_for(engine.get_context().get_queue()),
                engine.get_mode());
        }

        FusionChoice DeviceFusionSelector::choose(const FusionQuery& query) const
        {
            if (query.matched == FusedForm::NONE)
                return FusionChoice{FusedForm::NONE, "nothing was matched"};

            // --- The device has to be able to run the element type at all -----
            // Declining hands the work back to the plain kernels, which report
            // an unsupported dtype in SushiBLAS's own words naming SushiBLAS's
            // own operation. A fused kernel failing instead would name a
            // composition the caller never wrote and cannot look up.
            if (query.dtype == DType::FLOAT64 && !profile_.supports_fp64)
                return FusionChoice{FusedForm::NONE,
                                    "the device has no native double precision, so the plain "
                                    "kernels report the problem rather than a fused form the "
                                    "caller never asked for"};

            if (is_gemm_epilogue(query.matched))
            {
                // --- The epilogue must be safe wherever the GEMM runs ---------
                // Under FORCE_REFERENCE and VERIFY, Kernels::Reference::gemm
                // invokes the epilogue on the **host**, so every pointer the
                // functor carries has to be host-readable there. SushiAI's own
                // allocations are shared USM and would be, but a caller may bind
                // a device-only tensor as a parameter and nothing here can ask
                // which it got. Dereferencing device USM from the host is a
                // fault or garbage, so the two debugging modes get the unfused
                // spelling - which costs launches in a mode that is already
                // running everything twice, and changes no number.
                if (mode_ == SushiBLAS::ExecutionMode::FORCE_REFERENCE
                    || mode_ == SushiBLAS::ExecutionMode::VERIFY)
                    return FusionChoice{FusedForm::NONE,
                                        "this execution mode runs the epilogue on the host, "
                                        "where an operand in device memory cannot be read"};

                // --- The epilogue indexes its operand directly ----------------
                // `bias[col]` is not a strided access and cannot be made one
                // without the stride reaching the functor, which would cost a
                // register and a multiply on every element of C for a case that
                // does not arise inside this library.
                if (!query.operand_is_contiguous)
                    return FusionChoice{FusedForm::NONE,
                                        "the epilogue's broadcast operand is not contiguous, and "
                                        "the functor indexes it directly"};
            }

            // --- What was considered and deliberately not used ---------------
            // `has_dedicated_local_memory` picks between SushiBLAS's local-tiled
            // and no-local GEMM, and **both** apply the epilogue at the same
            // point - the store - so it does not bear on whether to fuse. Nor
            // does the problem size: every entry in today's catalog removes
            // whole read-modify-write passes over the result, which is a win
            // that grows with the tensor rather than shrinking. Both facts are
            // carried in the query so a future entry can use them; today the
            // honest answer is that neither changes the decision, and inventing
            // a threshold that cannot be measured on the single OpenCL CPU
            // device this machine exposes would be worse than saying so.
            return FusionChoice{query.matched, "no device fact argues against this form"};
        }

    } // namespace Graph
} // namespace SushiAI
