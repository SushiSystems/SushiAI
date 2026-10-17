/**************************************************************************/
/* darknet_tensors.hpp                                                    */
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
 * @file darknet_tensors.hpp
 * @brief Declares the listing of Darknet convolution vectors as the tensors of a weight file.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /**
         * @brief Lists each convolution's vectors as the tensors its targets name, on the host.
         * @param weights The convolutions in file order; shapes and targets match them by index.
         * @return One float32 tensor per target name that is not empty: the weights as
         *         [out, kh, kw, in], every other vector as [out]. Incompatible naming the layer
         *         for a vector of another length or a name stated twice; InvalidDescription
         *         for a shape that is not positive.
         */
        [[nodiscard]] Result<std::vector<IO::ImportedTensor>> list_darknet_tensors(
            const DarknetWeights& weights,
            SushiRuntime::span<const DarknetConvolutionShape> shapes,
            SushiRuntime::span<const DarknetWeightTarget> targets);

    } // namespace Vision::Importers
} // namespace SushiAI
