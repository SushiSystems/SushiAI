/**************************************************************************/
/* run_state.hpp                                                          */
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
 * @file run_state.hpp
 * @brief Declares what an optimizer keeps across updates, as named tensors and scalars.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /** @brief Names the scalar that counts the updates an optimizer has applied. */
        inline constexpr std::string_view K_STEP_COUNT_STATE = "step_count";

        /** @brief Names the scalar that holds the scale of an attached loss scaler. */
        inline constexpr std::string_view K_LOSS_SCALE_STATE = "loss_scale";

        /** @brief Names the scalar that counts the clean steps since the scale last moved. */
        inline constexpr std::string_view K_LOSS_SCALE_GOOD_STEPS_STATE = "loss_scale.good_steps";

        /** @brief Holds what follows a parameter's name in the name of its momentum buffer. */
        inline constexpr std::string_view K_MOMENTUM_ROLE = ".momentum";

        /** @brief Holds what follows a parameter's name in the name of its first moment. */
        inline constexpr std::string_view K_FIRST_MOMENT_ROLE = ".first_moment";

        /** @brief Holds what follows a parameter's name in the name of its second moment. */
        inline constexpr std::string_view K_SECOND_MOMENT_ROLE = ".second_moment";

        /**
         * @struct StateTensor
         * @brief Holds one state tensor of an optimizer under its name.
         */
        struct StateTensor
        {
            /** @brief Holds the parameter's name and a role, such as "0.weight.momentum". */
            std::string name;

            /** @brief Shares the optimizer's own storage; a write through it restores it. */
            SushiBLAS::Tensor tensor;
        };

        /**
         * @struct StateScalar
         * @brief Holds one scalar of an optimizer's run state under its name.
         */
        struct StateScalar
        {
            /** @brief Holds the name, such as @ref K_STEP_COUNT_STATE. */
            std::string name;

            /** @brief Holds a count or a real number; the name fixes which. */
            std::variant<std::uint64_t, double> value;
        };

        /**
         * @struct RunState
         * @brief Holds everything an optimizer keeps across updates that its options omit.
         */
        struct RunState
        {
            /** @brief Holds the state tensors, in slot order and then in role order. */
            std::vector<StateTensor> tensors;

            /** @brief Holds the scalars; the step count is first. */
            std::vector<StateScalar> scalars;
        };

    } // namespace Optim
} // namespace SushiAI
