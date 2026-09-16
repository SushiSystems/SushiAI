/**************************************************************************/
/* precision.cpp                                                          */
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

/**
 * @file precision.cpp
 * @brief Implements precision policies and mixed-precision narrowing decisions.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/precision.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Reason string literals preserved for diagnostic reports.
             */
            constexpr std::string_view REASON_DECLARED =
                "stays in the graph's own precision";
            constexpr std::string_view REASON_FLOPS =
                "the matrix product is where the work is, so it is what low precision is for";
            constexpr std::string_view REASON_EPILOGUE =
                "a fused GEMM applies this inside the GEMM's kernel, in the GEMM's element "
                "type, so the unfused spelling has to agree or fusion would change the numbers";
            constexpr std::string_view REASON_ALREADY_NARROW =
                "already at or below the policy's low type";
        }

        bool narrows_under_mixed_precision(OpID op) noexcept
        {
            // Matrix products and their narrowing epilogue forms compute in lower precision.
            if (op == Ops::MATMUL.id
                || op == Ops::GEMM_BIAS.id
                || op == Ops::GEMM_BIAS_RELU.id
                || op == Ops::GEMM_BIAS_SIGMOID.id
                || op == Ops::GEMM_BIAS_TANH.id)
                return true;

            // Fused bias addition and surviving activations match GEMM precision.
            if (op == Ops::BIAS_ADD.id) return true;
            if (Ops::activation_survives_fusion(op)) return true;

            // Activation backward operations follow the precision of their forward pass.
            for (const Ops::Activation& activation : Ops::ACTIVATIONS)
                if (activation.backward == op)
                    return Ops::activation_survives_fusion(activation.forward);

            return false;
        }

        PrecisionChoice UniformPrecisionPolicy::choose(const PrecisionQuery& query) const
        {
            return PrecisionChoice{query.declared, REASON_DECLARED};
        }

        MixedPrecisionPolicy::MixedPrecisionPolicy(DType low) : low_(low)
        {
            SA_THROW_IF(!is_real_float(low),
                        "A mixed-precision policy narrows to a real floating type; {} is not "
                        "one. SushiBLAS's converting copy is bounded to HALF, FLOAT32 and "
                        "FLOAT64, and an integer conversion is a rounding policy nothing here "
                        "is entitled to choose.",
                        dtype_name(low));
        }

        PrecisionChoice MixedPrecisionPolicy::choose(const PrecisionQuery& query) const
        {
            if (!narrows_under_mixed_precision(query.op))
                return PrecisionChoice{query.declared, REASON_DECLARED};

            // Narrowing applies strictly downward to real floats preserving idempotence.
            if (!is_real_float(query.declared) || dtype_size(query.declared) <= dtype_size(low_))
                return PrecisionChoice{query.declared, REASON_ALREADY_NARROW};

            const bool is_product = query.op == Ops::MATMUL.id
                                 || query.op == Ops::GEMM_BIAS.id
                                 || query.op == Ops::GEMM_BIAS_RELU.id
                                 || query.op == Ops::GEMM_BIAS_SIGMOID.id
                                 || query.op == Ops::GEMM_BIAS_TANH.id;

            return PrecisionChoice{low_, is_product ? REASON_FLOPS : REASON_EPILOGUE};
        }

    } // namespace Graph
} // namespace SushiAI
