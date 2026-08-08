/**************************************************************************/
/* linalg.cpp                                                             */
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

#include <SushiAI/autograd/rules/linalg.hpp>

namespace SushiAI
{
    namespace Autograd
    {
        void matmul_backward(const GradientTape& tape)
        {
            const Graph::Node& node = tape.node();
            const Graph::ValueId dc = tape.output_grad(0);
            if (dc == Graph::NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueId a = node.inputs.at(0);
            const Graph::ValueId b = node.inputs.at(1);
            const bool ta = node.attributes.flag(0);
            const bool tb = node.attributes.flag(1);

            // Four transposition cases, each derived by treating the product
            // as U*V with U = op(A), V = op(B): dU = dC V^T, dV = U^T dC, and
            // then transposing back through whichever operand was flagged.
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
            if (dy == Graph::NO_VALUE) return;

            // Y = X + 1 b^T is the identity in X, so the gradient passes
            // straight through with nothing emitted, and it sums over the
            // broadcast axis in b.
            if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().sum_rows(dy));
        }

        void sum_rows_backward(const GradientTape& tape)
        {
            const Graph::ValueId ds = tape.output_grad(0);
            if (ds == Graph::NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueId x = tape.node().inputs.at(0);
            const Graph::Value& value = tape.graph().value(x);

            // Summing over rows broadcasts back over rows; adding the row
            // vector to a zero matrix is exactly that broadcast, and reuses
            // the bias lowering rather than inventing a second one.
            const Graph::ValueId zeros = tape.builder().zeros(value.shape, value.dtype);
            tape.contribute_input(0, tape.builder().bias_add(zeros, ds));
        }

    } // namespace Autograd
} // namespace SushiAI
