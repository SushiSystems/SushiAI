/**************************************************************************/
/* fitted_sample.hpp                                                      */
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
 * @file fitted_sample.hpp
 * @brief Declares one sample as a collator takes it: pixels and labels at the input size.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <optional>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Holds one sample after its image and its labels were fitted to the input size.
     *
     * A sample of a tabular list has no image; its features and targets view the row of
     * the list's table, which outlives the fitted sample.
     */
    struct FittedSample
    {
        /** @brief Holds the pixels at the input size. */
        Imaging::Image image;

        /** @brief Holds the index, in SampleList::files, of the file the pixels came from. */
        std::uint32_t file = 0;

        /** @brief Holds the class of the whole image; absent when the layout labels objects. */
        std::optional<std::int32_t> class_index;

        /** @brief Holds a class, a box and a polygon per object, in pixels of the image. */
        Imaging::Labels objects;

        /** @brief Holds one class per pixel at the input size; no pixel without a mask. */
        Imaging::Image mask;

        /** @brief Views the features of a tabular sample; empty for an image. */
        SushiRuntime::span<const float> features;

        /** @brief Views the targets of a tabular sample; empty for an image. */
        SushiRuntime::span<const float> targets;
    };
} // namespace SushiAI::Vision::Datasets
