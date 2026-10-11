/**************************************************************************/
/* darknet_model.hpp                                                      */
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
 * @file darknet_model.hpp
 * @brief Declares the translation of a parsed .cfg into a model spec and what its heads need.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Names how a normalised convolution is written into the model. */
        enum class DarknetNormalisation : std::uint8_t
        {
            Separate, /**< A convolution without bias, then a batch-norm row. */
            Folded,   /**< A convolution with bias; the caller folds the weights. */
        };

        /** @brief Holds the choices a translation takes. */
        struct DarknetTranslation
        {
            /** @brief Holds the form of every normalised convolution. */
            DarknetNormalisation normalisation = DarknetNormalisation::Separate;
        };

        /** @brief Describes one [yolo] section as a published head. */
        struct DarknetHead
        {
            /** @brief Holds the published name, "head.<k>" for the k-th [yolo] of the file. */
            std::string output;

            /** @brief Holds the index of the [yolo] section among the file's layers. */
            std::int64_t layer = 0;

            /** @brief Holds the class count, the coding and the offset stretch of the head. */
            Detect::Anchor::HeadOptions options;
        };

        /** @brief Holds a translated .cfg: the model, its heads and where its weights go. */
        struct DarknetModel
        {
            /** @brief Holds the rows and the published outputs, one output per head. */
            Config::ModelSpec spec;

            /** @brief Holds the size of the network input in pixels. */
            Detect::Anchor::NetworkSize network;

            /** @brief Holds the anchors as fractions of the input and each head's mask. */
            Detect::Anchor::AnchorLayout layout;

            /** @brief Holds the heads in file order; entry k matches head k of layout. */
            std::vector<DarknetHead> heads;

            /** @brief Holds the suppression measure the last [yolo] section names. */
            Detect::SuppressionCriterion criterion = Detect::SuppressionCriterion::IoU;

            /** @brief Holds the exponent of the distance term, read by DistanceIoU only. */
            float distance_exponent = Detect::K_DEFAULT_DISTANCE_EXPONENT;

            /** @brief Holds one shape per convolutional section, in file order. */
            std::vector<DarknetConvolutionShape> convolutions;

            /** @brief Holds the state entries of each convolution, matched by index. */
            std::vector<DarknetWeightTarget> targets;

            /** @brief Holds, per layer of the file, the row that carries its output. */
            std::vector<std::int64_t> row_of_layer;
        };

        /**
         * @brief Translates a parsed .cfg into the rows, heads and weight targets of a model.
         * @param config The sections as parse_darknet_config returns them.
         * @return The model; Unsupported naming the line of a section, a key or a value this
         *         importer does not translate; InvalidDescription naming the line of a value
         *         that is malformed, out of range or does not fit the layer it reads.
         */
        [[nodiscard]] Result<DarknetModel> translate_darknet_model(
            const DarknetConfig& config, const DarknetTranslation& translation);

    } // namespace Vision::Importers
} // namespace SushiAI
