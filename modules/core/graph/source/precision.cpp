/**************************************************************************/
/* precision.cpp                                                          */
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
 * @file precision.cpp
 * @brief Implements precision policies and mixed-precision narrowing decisions.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/precision.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Reason string literals preserved for diagnostic reports.
             */
            constexpr std::string_view K_REASON_DECLARED =
                "stays in the graph's own precision";
            constexpr std::string_view K_REASON_FLOPS =
                "the matrix product is where the work is, so it is what low precision is for";
            constexpr std::string_view K_REASON_EPILOGUE =
                "a fused GEMM applies this inside the GEMM's kernel, in the GEMM's element "
                "type, so the unfused spelling has to agree or fusion would change the numbers";
            constexpr std::string_view K_REASON_ALREADY_NARROW =
                "already at or below the policy's low type";
        }

        bool narrows_under_mixed_precision(OpID op) noexcept
        {
            // Matrix products and their narrowing epilogue forms compute in lower precision.
            if (op == Ops::K_MATMUL.id
                || op == Ops::K_GEMM_BIAS.id
                || op == Ops::K_GEMM_BIAS_RELU.id
                || op == Ops::K_GEMM_BIAS_SIGMOID.id
                || op == Ops::K_GEMM_BIAS_TANH.id)
                return true;

            // Fused bias addition and surviving activations match GEMM precision.
            if (op == Ops::K_BIAS_ADD.id) return true;
            if (Ops::activation_survives_fusion(op)) return true;

            // Activation backward operations follow the precision of their forward pass.
            for (const Ops::Activation& activation : Ops::K_ACTIVATIONS)
                if (activation.backward == op)
                    return Ops::activation_survives_fusion(activation.forward);

            return false;
        }

        PrecisionChoice UniformPrecisionPolicy::choose(const PrecisionQuery& query) const
        {
            return PrecisionChoice{query.declared, K_REASON_DECLARED};
        }

        Result<MixedPrecisionPolicy> MixedPrecisionPolicy::make(DType low) noexcept
        {
            SA_FAIL_IF(!is_real_float(low), StatusCode::InvalidDescription,
                       "A mixed-precision policy narrows to a real floating type; {} is not "
                       "one. SushiBLAS's converting copy is bounded to HALF, FLOAT32 and "
                       "FLOAT64, and an integer conversion is a rounding policy nothing here "
                       "is entitled to choose.",
                       dtype_name(low));

            return MixedPrecisionPolicy(low);
        }

        PrecisionChoice MixedPrecisionPolicy::choose(const PrecisionQuery& query) const
        {
            if (!narrows_under_mixed_precision(query.op))
                return PrecisionChoice{query.declared, K_REASON_DECLARED};

            // Narrowing applies strictly downward to real floats preserving idempotence.
            if (!is_real_float(query.declared) || dtype_size(query.declared) <= dtype_size(low_))
                return PrecisionChoice{query.declared, K_REASON_ALREADY_NARROW};

            const bool is_product = query.op == Ops::K_MATMUL.id
                                 || query.op == Ops::K_GEMM_BIAS.id
                                 || query.op == Ops::K_GEMM_BIAS_RELU.id
                                 || query.op == Ops::K_GEMM_BIAS_SIGMOID.id
                                 || query.op == Ops::K_GEMM_BIAS_TANH.id;

            return PrecisionChoice{low_, is_product ? K_REASON_FLOPS : K_REASON_EPILOGUE};
        }

    } // namespace Graph
} // namespace SushiAI
