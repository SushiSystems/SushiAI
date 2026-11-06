/**************************************************************************/
/* host_values.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file host_values.cpp
 * @brief Implements the host read of one element of a float32 or float64 tensor.
 * @author Mustafa Garip
 */

#include <SushiAI/train/host_values.hpp>

#include <SushiAI/dtype/dtype.hpp>

namespace SushiAI
{
    namespace Train
    {
        Status require_host_reals(const SushiBLAS::Tensor& tensor, std::int64_t elements,
                                  std::string_view what) noexcept
        {
            SA_FAIL_IF(tensor.dtype != DType::Float32 && tensor.dtype != DType::Float64,
                       StatusCode::InvalidDescription,
                       "A metric needs a host read of {} and they are {}; only {} and {} "
                       "can be read back",
                       what, dtype_name(tensor.dtype), dtype_name(DType::Float32),
                       dtype_name(DType::Float64));
            SA_FAIL_IF(tensor.num_elements != elements, StatusCode::Incompatible,
                       "A metric reads {} as {} values; the step holds {}", what, elements,
                       tensor.num_elements);

            return Status();
        }

        double host_real(const SushiBLAS::Tensor& tensor, std::int64_t index) noexcept
        {
            return tensor.dtype == DType::Float64
                 ? tensor.data_as<double>()[index]
                 : static_cast<double>(tensor.data_as<float>()[index]);
        }

    } // namespace Train
} // namespace SushiAI
