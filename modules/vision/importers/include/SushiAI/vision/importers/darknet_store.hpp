/**************************************************************************/
/* darknet_store.hpp                                                      */
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
 * @file darknet_store.hpp
 * @brief Declares the copy of Darknet convolution vectors into a model state.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /**
         * @brief Names the state entry that receives each vector of one convolution.
         *
         * An empty name leaves that vector unstored.
         */
        struct DarknetWeightTarget
        {
            /** @brief Holds the entry of the weights, which are stored kernel last. */
            std::string weight;

            /** @brief Holds the entry of the biases. */
            std::string bias;

            /** @brief Holds the entry of the batch-norm scales. */
            std::string scale;

            /** @brief Holds the entry of the rolling means. */
            std::string mean;

            /** @brief Holds the entry of the rolling variances. */
            std::string variance;
        };

        /**
         * @brief Copies each convolution's vectors into the state entries its targets name.
         * @param weights The convolutions in file order; shapes and targets match them by index.
         * @param state The state written; it is left untouched when a refusal is returned.
         * @return Incompatible naming the layer for a missing or repeated entry, a dtype or shape
         *         that differs, or a vector of another length; InvalidDescription for a shape.
         */
        [[nodiscard]] Status store_darknet_weights(
            const DarknetWeights& weights,
            SushiRuntime::span<const DarknetConvolutionShape> shapes,
            SushiRuntime::span<const DarknetWeightTarget> targets,
            NN::ModelState& state);

    } // namespace Vision::Importers
} // namespace SushiAI
