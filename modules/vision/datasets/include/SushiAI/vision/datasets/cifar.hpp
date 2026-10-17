/**************************************************************************/
/* cifar.hpp                                                              */
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
 * @file cifar.hpp
 * @brief Declares the reader of the CIFAR-10 and CIFAR-100 binary batch files.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Holds the height and the width of a CIFAR image, in pixels. */
    inline constexpr std::int32_t K_CIFAR_SIDE = 32;

    /** @brief Names the record form of a CIFAR file and the label a sample takes from it. */
    enum class CIFARVariant : std::uint8_t
    {
        CIFAR10,        /**< One label byte per record; 10 classes. */
        CIFAR100Fine,   /**< A coarse and a fine label byte; the fine one, of 100 classes. */
        CIFAR100Coarse, /**< A coarse and a fine label byte; the coarse one, of 20 classes. */
    };

    /** @brief Holds what read_cifar reads. */
    struct CIFAROptions
    {
        /** @brief Holds the directory of the binary files. */
        std::filesystem::path root;

        /** @brief Names the split, "train" or "test". */
        std::string split = "train";

        /** @brief Names the record form and the label taken. */
        CIFARVariant variant = CIFARVariant::CIFAR10;
    };

    /**
     * @brief Lists the records of the binary files of one split with the class of each.
     *
     * Each sample refers to its three planes, red first, inside its file. README.md,
     * section cifar.hpp, lists the files of each variant and every status.
     * @return The list; a failure that names the file and, for a label at or above the
     *         class count, the one-based record.
     */
    [[nodiscard]] Result<SampleList> read_cifar(const CIFAROptions& options);
} // namespace SushiAI::Vision::Datasets
