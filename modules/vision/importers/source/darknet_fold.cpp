/**************************************************************************/
/* darknet_fold.cpp                                                       */
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
 * @file darknet_fold.cpp
 * @brief Defines the batch-normalisation fold of a Darknet convolution.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_fold.hpp>

#include <cmath>
#include <cstddef>
#include <SushiAI/core/check.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the epsilon Darknet adds under every rolling variance. */
            constexpr double K_BATCH_NORM_EPSILON = 0.00001;

            /**
             * @brief Stops the process when the convolution cannot be folded.
             * @param convolution The convolution about to be folded.
             */
            void check_foldable(const DarknetConvolution& convolution)
            {
                const std::size_t filters = convolution.biases.size();
                SA_FATAL_IF(filters == 0 || convolution.scales.size() != filters
                                || convolution.rolling_mean.size() != filters
                                || convolution.rolling_variance.size() != filters,
                            "a convolution to fold has no filters, or lacks a scale, mean or "
                            "variance for each filter; it is not normalised, or was folded "
                            "already");
                SA_FATAL_IF(convolution.weights.size() < filters
                                || convolution.weights.size() % filters != 0,
                            "the weights of a convolution to fold are not a whole number per "
                            "filter");

                for (std::size_t f = 0; f < filters; ++f)
                {
                    const double variance = convolution.rolling_variance[f];
                    SA_FATAL_IF(!(variance + K_BATCH_NORM_EPSILON > 0.0),
                                "filter {} has no positive variance once the epsilon is added", f);
                }
            }
        } // namespace

        void fold_darknet_normalisation(DarknetConvolution& convolution)
        {
            check_foldable(convolution);

            const std::size_t filters = convolution.biases.size();
            const std::size_t per_filter = convolution.weights.size() / filters;
            for (std::size_t f = 0; f < filters; ++f)
            {
                const double scale = convolution.scales[f];
                const double mean = convolution.rolling_mean[f];
                const double root = std::sqrt(static_cast<double>(convolution.rolling_variance[f])
                                              + K_BATCH_NORM_EPSILON);

                // The filter is scaled by scale / root; the bias moves by - scale * mean / root.
                const double bias = convolution.biases[f];
                convolution.biases[f] = static_cast<float>(bias - scale * mean / root);

                const double factor = scale / root;
                float* filter_weights = convolution.weights.data() + f * per_filter;
                for (std::size_t j = 0; j < per_filter; ++j)
                    filter_weights[j] = static_cast<float>(static_cast<double>(filter_weights[j])
                                                           * factor);
            }

            convolution.scales.clear();
            convolution.rolling_mean.clear();
            convolution.rolling_variance.clear();
        }

    } // namespace Vision::Importers
} // namespace SushiAI
