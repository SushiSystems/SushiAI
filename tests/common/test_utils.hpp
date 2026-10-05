/****************************************************************/
/* test_utils.hpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_utils.hpp
 * @brief Declares the per-test Engine fixture and the host-side tensor helpers device tests use.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_utils.hpp".
 */

#pragma once

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
        /** @brief Provides a fresh Engine per test, over the shared RuntimeContext. */
        class EngineTest : public ::testing::Test
        {
            protected:
                /** @brief Builds the test's Engine over the shared context. */
                void SetUp() override
                {
                    engine = std::make_unique<SushiBLAS::Engine>(TestEnvironment::context());
                }

                /** @brief Destroys the test's Engine. */
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

                /** @brief Holds the Engine the test records on. */
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
         * @brief Returns a tensor's element bytes, for a bit-for-bit comparison.
         * @see tests/common/README.md, "test_utils.hpp".
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
         * @brief Generates a reproducible value stream for filling test tensors.
         * @see tests/common/README.md, "test_utils.hpp".
         */
        class HostNoise
        {
            public:
                /** @brief Seeds the stream. */
                explicit HostNoise(std::uint64_t seed) noexcept : state_(seed) {}

                /** @brief Returns the next value, roughly uniform in (-scale, scale). */
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
                /** @brief Holds the splitmix state. */
                std::uint64_t state_;
        };

    } // namespace Test
} // namespace SushiAI
