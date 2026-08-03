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

#include <SushiAI/graph/precision.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief The reason strings, once, so the two policies cannot drift.
             *
             * Static storage of literals rather than formatted text: a
             * @c PrecisionChoice carries a @c string_view it does not own, and
             * a choice is made once per node per pass, so allocating a sentence
             * for each would be both a lifetime question and a cost for
             * something nobody reads unless a report is printed.
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
            // Rule 1: the matrix product and every spelling of it this policy
            // is willing to produce. GEMM_BIAS_GELU is deliberately absent -
            // narrowing it would put the tanh approximation's `x*x*x` into
            // fp16, and rule 5 argues that at length. Nothing is lost by the
            // omission: under this policy a GELU layer never fuses into that
            // form in the first place, so the answer here only has to be the
            // one find_dtype_mismatches would agree with if it ever met one,
            // and "the graph's own precision" is that answer.
            if (op == Ops::MATMUL.id
                || op == Ops::GEMM_BIAS.id
                || op == Ops::GEMM_BIAS_RELU.id
                || op == Ops::GEMM_BIAS_SIGMOID.id
                || op == Ops::GEMM_BIAS_TANH.id)
                return true;

            // Rule 2: what a narrowed GEMM epilogue folds in. BIAS_ADD is in
            // every one of the catalog's GEMM entries, so it has to agree with
            // them. The activations are the ones whose gradient reads their
            // output - which is *not* the same set as "the ones the catalog can
            // fold" any more, because GEMM_BIAS_GELU folds GELU by spilling the
            // pre-activation. The two sets are allowed to differ, and rule 5
            // says why they must here: an activation this policy keeps wide
            // gets a CAST between it and the GEMM, which stops the chain
            // matching, so the fold this rule would have had to agree with
            // never occurs.
            if (op == Ops::BIAS_ADD.id) return true;
            if (Ops::activation_survives_fusion(op)) return true;

            // An activation's gradient follows its forward. Not because a
            // kernel folds it - none does - but because its primal operand *is*
            // that forward's result: differentiating a low-precision activation
            // in high precision would cast the primal back up on every step and
            // then cast the result back down for the matrix product below it,
            // for no gain in a rule that computes one multiply per element.
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

            // Narrow only downwards, and only between real floats. This is also
            // what makes choose() idempotent - see the header: asked again about
            // a node already at low_, the guard below answers low_ unchanged, so
            // running the pass twice inserts nothing the second time and the
            // invariant can be re-checked after fusion has rewritten the graph.
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
