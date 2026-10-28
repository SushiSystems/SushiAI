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
            const Graph::ValueID dc = tape.output_grad(0);
            if (dc == Graph::K_NO_VALUE) return;

            const Graph::GraphBuilder& builder = tape.builder();
            const Graph::ValueID a = tape.input(0);
            const Graph::ValueID b = tape.input(1);
            const bool ta = node.attributes.flag(0);
            const bool tb = node.attributes.flag(1);

            // Matrix multiplication gradients derive from dU = dC V^T and dV = U^T dC.
            if (tape.input_requires_grad(0))
            {
                Graph::Traced da;
                if (!ta && !tb)     da = builder.matmul(dc, b, false, true);
                else if (ta && !tb) da = builder.matmul(b, dc, false, true);
                else if (!ta)       da = builder.matmul(dc, b, false, false);
                else                da = builder.matmul(b, dc, true, true);
                tape.contribute_input(0, da);
            }

            if (tape.input_requires_grad(1))
            {
                Graph::Traced db;
                if (!ta && !tb)     db = builder.matmul(a, dc, true, false);
                else if (ta && !tb) db = builder.matmul(a, dc, false, false);
                else if (!ta)       db = builder.matmul(dc, a, true, false);
                else                db = builder.matmul(dc, a, true, true);
                tape.contribute_input(1, db);
            }
        }

        void bias_add_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;

            // Bias addition passes gradient directly to input and sums across rows for bias.
            if (tape.input_requires_grad(0)) tape.contribute_input(0, dy);
            if (tape.input_requires_grad(1)) tape.contribute_input(1, tape.builder().sum_rows(dy));
        }

        void sum_rows_backward(const GradientTape& tape)
        {
            const Graph::ValueID ds = tape.output_grad(0);
            if (ds == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            const Graph::ValueID x = tape.input(0);
            const Graph::Value& value = tape.graph().value(x);

            // Sum-rows adjoint broadcasts row gradients back over matrix shape via bias addition on zeros.
            const Graph::Traced zeros = tape.builder().zeros_like(ds, value.shape, value.dtype);
            tape.contribute_input(0, tape.builder().bias_add(zeros, ds));
        }

        void gather_rows_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // The ids and the row count are copied out before the emission grows the graph.
            const Graph::ValueID table = tape.input(0);
            const Graph::ValueID indices = tape.input(1);
            const int64_t table_rows = tape.graph().value(table).shape[0];

            tape.contribute_input(0, tape.builder().scatter_add_rows(dy, indices, table_rows));
        }

        void scatter_add_rows_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // The id is copied out before the emission grows the graph.
            const Graph::ValueID indices = tape.input(1);

            tape.contribute_input(0, tape.builder().gather_rows(dy, indices));
        }

        void causal_mask_backward(const GradientTape& tape)
        {
            const Graph::ValueID dy = tape.output_grad(0);
            if (dy == Graph::K_NO_VALUE) return;
            if (!tape.input_requires_grad(0)) return;

            // A replaced entry does not depend on its input, so its gradient is zero.
            constexpr double K_NO_GRADIENT = 0.0;

            tape.contribute_input(0, tape.builder().causal_mask(dy, K_NO_GRADIENT));
        }

    } // namespace Autograd
} // namespace SushiAI
