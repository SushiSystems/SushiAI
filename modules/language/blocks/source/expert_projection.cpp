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
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        Result<Graph::ValueID> ExpertProjection::mix(const Graph::GraphBuilder& builder,
                                                     Graph::ValueID gated, Graph::ValueID gate)
        {
            SA_FAIL_IF(dim <= 0 || hidden <= 0 || experts <= 0, StatusCode::InvalidDescription,
                       "ExpertProjection needs positive extents; got dim={} hidden={} "
                       "experts={}",
                       dim, hidden, experts);

            return NN::Detail::bridge_throwing([&]()
            {
                const DType dtype = builder.graph().value(gated).dtype;

                // The fans are those of one expert, so each starts as a FeedForward does.
                const Graph::ValueID w = weight.declare(
                    builder, Shape{dim, experts * hidden}, NN::ParameterRole::Weight, hidden,
                    dim, "weight", dtype);
                const Graph::ValueID b = bias.declare(
                    builder, Shape{experts, dim}, NN::ParameterRole::Bias, hidden, dim, "bias",
                    dtype);

                return builder.add(builder.matmul(gated, w, false, true),
                                   builder.matmul(gate, b));
            });
        }

    } // namespace Language::Blocks
} // namespace SushiAI
