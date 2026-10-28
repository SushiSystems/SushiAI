/**************************************************************************/
/* expert_projection.cpp                                                  */
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
 * @file expert_projection.cpp
 * @brief Implements ExpertProjection's two products and their sum.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/expert_projection.hpp>

#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        Graph::Traced ExpertProjection::mix(const Graph::GraphBuilder& builder,
                                            const Graph::Traced& gated,
                                            const Graph::Traced& gate)
        {
            SA_REFUSE_IF(builder, dim <= 0 || hidden <= 0 || experts <= 0,
                         "ExpertProjection needs positive extents; got dim={} hidden={} "
                         "experts={}",
                         dim, hidden, experts);

            SA_TRY_ASSIGN(const Graph::ValueID gated_id, gated);
            SA_TRY_ASSIGN(const Graph::ValueID gate_id, gate);
            const DType dtype = builder.graph().value(gated_id).dtype;

            // The fans are those of one expert, so each starts as a FeedForward does.
            SA_TRY_ASSIGN(const Graph::ValueID w,
                          weight.declare(builder, Shape{dim, experts * hidden},
                                         NN::ParameterRole::Weight, hidden, dim, "weight",
                                         dtype));
            SA_TRY_ASSIGN(const Graph::ValueID b,
                          bias.declare(builder, Shape{experts, dim}, NN::ParameterRole::Bias,
                                       hidden, dim, "bias", dtype));

            return builder.add(builder.matmul(gated_id, w, false, true),
                               builder.matmul(gate_id, b));
        }

    } // namespace Language::Blocks
} // namespace SushiAI
