/**************************************************************************/
/* test_utils.hpp                                                         */
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

#pragma once

/**
 * @file test_utils.hpp
 * @brief The fixture and the host-side tensor helpers every device test uses.
 *
 * Reads and writes go straight through the shared-USM pointer rather than
 * through a queue memcpy. Every tensor the pool hands out is Shared-strategy
 * USM, and the fixture waits before it reads, so the pointer is valid and
 * coherent - and doing it this way keeps a test from accidentally queueing
 * work into the engine graph it is in the middle of inspecting.
 */

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include "test_environment.hpp"

namespace SushiAI
{
    namespace Test
    {
        /**
         * @class EngineTest
         * @brief A fresh Engine per test, over the shared RuntimeContext.
         */
        class EngineTest : public ::testing::Test
        {
            protected:
                void SetUp() override
                {
                    engine = std::make_unique<SushiBLAS::Engine>(TestEnvironment::context());
                }

                void TearDown() override
                {
                    engine.reset();
                }

                /** @brief Flushes anything recorded and waits for the device. */
                void flush()
                {
                    engine->execute();
                    TestEnvironment::context().wait_all();
                }

                std::unique_ptr<SushiBLAS::Engine> engine;
        };

        /** @brief Writes host doubles into a FLOAT32 or FLOAT64 tensor's shared USM. */
        inline void write_tensor(SushiBLAS::Tensor& tensor, SushiRuntime::span<const double> values)
        {
            EXPECT_EQ(static_cast<std::size_t>(tensor.num_elements), values.size());

            if (tensor.dtype == DType::FLOAT64)
            {
                double* data = tensor.data_as<double>();
                for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
                return;
            }

            float* data = tensor.data_as<float>();
            for (std::size_t i = 0; i < values.size(); ++i) data[i] = static_cast<float>(values[i]);
        }

        /** @brief Reads a FLOAT32 or FLOAT64 tensor's shared USM back as host doubles. */
        inline std::vector<double> read_tensor(const SushiBLAS::Tensor& tensor)
        {
            const std::size_t count = static_cast<std::size_t>(tensor.num_elements);
            std::vector<double> values(count);

            if (tensor.dtype == DType::FLOAT64)
            {
                const double* data = tensor.data_as<double>();
                for (std::size_t i = 0; i < count; ++i) values[i] = data[i];
                return values;
            }

            const float* data = tensor.data_as<float>();
            for (std::size_t i = 0; i < count; ++i) values[i] = static_cast<double>(data[i]);
            return values;
        }

        /**
         * @brief A tensor's element bytes, for a bit-for-bit comparison.
         *
         * @c read_tensor widens everything to double, which is lossless for
         * float and therefore fine for "are these equal" - but it cannot
         * distinguish -0.0 from 0.0, and it says nothing about a payload
         * difference between two NaNs. A determinism claim is about bits, so
         * the check that guards it reads bits.
         *
         * @param tensor The tensor to read.
         * @return Its @c num_elements elements, as raw bytes.
         */
        inline std::vector<unsigned char> read_tensor_bytes(const SushiBLAS::Tensor& tensor)
        {
            const std::size_t bytes = static_cast<std::size_t>(tensor.num_elements)
                                    * dtype_size(tensor.dtype);
            const auto* data = static_cast<const unsigned char*>(tensor.data());
            return std::vector<unsigned char>(data, data + bytes);
        }

        /**
         * @brief Reports the first byte at which two element buffers differ.
         * @param a One buffer.
         * @param b The other.
         * @return The index, or a.size() when they are identical.
         */
        inline std::size_t first_difference(const std::vector<unsigned char>& a,
                                            const std::vector<unsigned char>& b)
        {
            const std::size_t shared = a.size() < b.size() ? a.size() : b.size();
            for (std::size_t i = 0; i < shared; ++i)
                if (a[i] != b[i]) return i;
            return shared;
        }

        /**
         * @brief A cheap deterministic value stream, for filling test tensors.
         *
         * A 64-bit splitmix step mapped to (-1, 1). Not a good PRNG and not
         * meant to be one; it just has to be reproducible without dragging a
         * device RNG into a test that is checking something else.
         */
        class HostNoise
        {
            public:
                explicit HostNoise(std::uint64_t seed) noexcept : state_(seed) {}

                /** @brief The next value, uniform-ish in (-scale, scale). */
                double next(double scale = 1.0) noexcept
                {
                    state_ += 0x9E3779B97F4A7C15ull;
                    std::uint64_t z = state_;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                    z ^= (z >> 31);

                    const double unit = static_cast<double>(z >> 11)
                                      / static_cast<double>(1ull << 53);
                    return (unit * 2.0 - 1.0) * scale;
                }

                /** @brief Fills @p values with next(@p scale). */
                void fill(SushiRuntime::span<double> values, double scale = 1.0) noexcept
                {
                    for (double& v : values) v = next(scale);
                }

            private:
                std::uint64_t state_;
        };

    } // namespace Test
} // namespace SushiAI
