/**************************************************************************/
/* head_contract.hpp                                                      */
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
 * @file head_contract.hpp
 * @brief Declares the names and bias constants an anchor-free head and its training share.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /** @brief Names the copy whose ports come first; also the prefix of its branches. */
        inline constexpr std::string_view K_DETECT_ONE_TO_MANY = "one2many";

        /** @brief Names the copy whose ports come second, present with end2end. */
        inline constexpr std::string_view K_DETECT_ONE_TO_ONE = "one2one";

        /** @brief Names the role of a box port. */
        inline constexpr std::string_view K_DETECT_BOXES = "boxes";

        /** @brief Names the role of a class-logit port. */
        inline constexpr std::string_view K_DETECT_SCORES = "scores";

        /** @brief Names the one-to-many box branch of the checkpoint. */
        inline constexpr std::string_view K_DETECT_BOX_BRANCH = "cv2";

        /** @brief Names the one-to-many class branch of the checkpoint. */
        inline constexpr std::string_view K_DETECT_CLASS_BRANCH = "cv3";

        /** @brief Names the one-to-one box branch of the checkpoint. */
        inline constexpr std::string_view K_DETECT_ONE_TO_ONE_BOX_BRANCH = "one2one_cv2";

        /** @brief Names the one-to-one class branch of the checkpoint. */
        inline constexpr std::string_view K_DETECT_ONE_TO_ONE_CLASS_BRANCH = "one2one_cv3";

        /** @brief Holds the index of the biased 1x1 convolution that ends every branch. */
        inline constexpr std::size_t K_DETECT_LAST_CONV_INDEX = 2;

        /** @brief Holds the value the last box convolution's bias starts from, given strides. */
        inline constexpr double K_DETECT_BOX_BIAS = 1.0;

        /** @brief Holds the expected object count per image the class bias assumes. */
        inline constexpr double K_DETECT_OBJECTS_PER_IMAGE = 5.0;

        /** @brief Holds the image side, in pixels, the class bias assumes. */
        inline constexpr double K_DETECT_REFERENCE_IMAGE_SIDE = 640.0;

        /**
         * @brief Builds the name of a head port, "<copy>.level<level>.<role>".
         * @param copy K_DETECT_ONE_TO_MANY or K_DETECT_ONE_TO_ONE.
         * @param role K_DETECT_BOXES or K_DETECT_SCORES.
         */
        [[nodiscard]] std::string detect_port_name(std::string_view copy, std::size_t level,
                                                   std::string_view role) noexcept;

        /**
         * @brief Returns the start of a last class bias, log(5 / classes / (640 / stride)^2).
         * @pre @p classes and @p stride are positive.
         */
        [[nodiscard]] double detect_class_bias(std::int64_t classes, double stride) noexcept;
    } // namespace Vision::Detect::Free
} // namespace SushiAI
