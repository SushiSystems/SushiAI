/**************************************************************************/
/* optimizer.cpp                                                          */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#include <utility>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/optim/optimizer.hpp>

namespace SushiAI
{
    namespace Optim
    {
        Optimizer::Optimizer(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters)
            : engine_(&engine), parameters_(std::move(parameters))
        {
            for (const ParameterSlot& slot : parameters_)
            {
                SA_THROW_IF(slot.parameter.num_elements != slot.gradient.num_elements,
                            "Parameter '{}' has {} elements but its gradient has {}",
                            slot.name, slot.parameter.num_elements, slot.gradient.num_elements);
                SA_THROW_IF(slot.parameter.dtype != slot.gradient.dtype,
                            "Parameter '{}' is {} but its gradient is {}",
                            slot.name, dtype_name(slot.parameter.dtype),
                            dtype_name(slot.gradient.dtype));
                SA_THROW_IF(!is_real_float(slot.parameter.dtype),
                            "Parameter '{}' is {}; optimizers update real floating tensors only",
                            slot.name, dtype_name(slot.parameter.dtype));
                SA_THROW_IF(slot.parameter.dtype == DType::HALF,
                            "Parameter '{}' is half precision; SushiBLAS's RNG cannot initialise "
                            "half state and a half accumulator loses the small updates that make "
                            "momentum work. Keep parameters in float32.",
                            slot.name);
            }
        }

        void Optimizer::flush()
        {
            engine_->execute();
            engine_->get_context().wait_all();
        }

        SushiBLAS::Tensor Optimizer::allocate_state(const SushiBLAS::Tensor& like) const
        {
            // Its own Storage, and matching the parameter's dtype and layout
            // exactly - see core/allocation.hpp for both reasons.
            return allocate_like(*engine_, like);
        }

    } // namespace Optim
} // namespace SushiAI
