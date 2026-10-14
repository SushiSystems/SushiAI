/**************************************************************************/
/* two_output_model.hpp                                                   */
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
 * @file two_output_model.hpp
 * @brief Declares the model of two published outputs the tests of a trainer build.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Builds Linear{6, 8}, GELU and Linear{8, 3}; publishes "features" and "logits". */
        [[nodiscard]] inline Result<NN::ModelGraph> build_two_output_model()
        {
            NN::ModelGraph model(Shape{6});

            SA_TRY_ASSIGN(const NN::NodeRef body,
                          model.add("body", NN::Linear{6, 8}, model.input()));
            SA_TRY_ASSIGN(const NN::NodeRef features, model.add("activation", NN::GELU{}, body));
            SA_TRY_ASSIGN(const NN::NodeRef logits, model.add("head", NN::Linear{8, 3}, features));
            SA_TRY(model.output("features", features));
            SA_TRY(model.output("logits", logits));

            return model;
        }

    } // namespace Test
} // namespace SushiAI
