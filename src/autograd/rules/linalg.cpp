/**************************************************************************/
/* linalg.cpp                                                             */
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
 * @file linalg.cpp
 * @brief Implements backward differentiation rules for linear algebra operations.
 * @author Mustafa Garip
 */

#include <SushiAI/autograd/rules/linalg.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void matmul_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId a = node.inputs.at(0);
            const Graph::ValueId b = node.inputs.at(1);
            const bool ta = node.attributes.flag(0);
            const bool tb = node.attributes.flag(1);

            // Matrix multiplication gradients derive from dU = dC V^T and dV = U^T dC.
            if (tape.input_requires_grad(0))
            {
                Graph::ValueId da;
                if (!ta && !tb)     da = builder.matmul(dc, b, false, true);
                else if (ta && !tb) da = builder.matmul(b, dc, false, true);
                else if (!ta)       da = builder.matmul(dc, b, false, false);
                else                da = builder.matmul(b, dc, true, true);
                tape.contribute_input(0, da);
            }

            if (tape.input_requires_grad(1))
            {
                Graph::ValueId db;
                if (!ta && !tb)     db = builder.matmul(a, dc, true, false);
                else if (ta && !tb) db = builder.matmul(a, dc, false, false);
                else if (!ta)       db = builder.matmul(dc, a, true, false);
                else                db = builder.matmul(dc, a, true, true);
                tape.contribute_input(1, db);
            }
        }

        void bias_add_backward(const GradientTape& tape)
        {
            const Graph::ValueId dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            // Bias addition passes gradient directly to input and sums across rows for bias.
            if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().sum_rows(dy));
        }

        void sum_rows_backward(const GradientTape& tape)
        {
            const Graph::ValueId ds = tape.output_grad(0);
            if (ds == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId x = tape.node().inputs.at(0);
            const Graph::Value& value = tape.graph().value(x);

            // Sum-rows adjoint broadcasts row gradients back over matrix shape via bias addition on zeros.
            const Graph::ValueId zeros = tape.builder().zeros_like(ds, value.shape, value.dtype);
            tape.contribute_input(0, tape.builder().bias_add(zeros, ds));
        }

    } // namespace Autograd
} // namespace SushiAI
