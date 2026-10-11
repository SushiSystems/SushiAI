/**************************************************************************/
/* test_utils.hpp                                                         */
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
 * @file test_utils.hpp
 * @brief Declares the Engine fixture, the tensor helpers, the builders and the trace summary.
 * @author Mustafa Garip
 * @see tests/common/README.md, "test_utils.hpp".
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.h>
#include "model_text.hpp"
#include "test_environment.hpp"
#include "test_status.hpp"

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
                    TestEnvironment::context().devices().wait_all();
                }

                /** @brief Holds the Engine the test records on. */
                std::unique_ptr<SushiBLAS::Engine> engine;
        };

        /** @brief Writes host doubles into a FLOAT32 or FLOAT64 tensor's shared USM. */
        inline void write_tensor(SushiBLAS::Tensor& tensor, SushiRuntime::span<const double> values)
        {
            EXPECT_EQ(static_cast<std::size_t>(tensor.num_elements), values.size());

            if (tensor.dtype == DType::Float64)
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

            if (tensor.dtype == DType::Float64)
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

        /** @brief Names a file under the temporary directory and removes it on both ends. */
        class ScratchFile
        {
            public:
                /** @brief Removes any file a previous run left under @p name. */
                explicit ScratchFile(const char* name)
                    : path_(std::filesystem::temp_directory_path() / name)
                {
                    std::error_code ignored;
                    std::filesystem::remove(path_, ignored);
                }

                /** @brief Removes the file. */
                ~ScratchFile()
                {
                    std::error_code ignored;
                    std::filesystem::remove(path_, ignored);
                }

                ScratchFile(const ScratchFile&) = delete;
                ScratchFile& operator=(const ScratchFile&) = delete;

                /** @brief Returns the path this file removes on destruction. */
                const std::filesystem::path& path() const noexcept { return path_; }

            private:
                /** @brief Holds the path of the scratch file. */
                std::filesystem::path path_;
        };

        /** @brief Builds Linear{6, 8}, GELU and Linear{8, 3}; publishes "features" and "logits". */
        [[nodiscard]] inline Result<NN::ModelGraph> build_two_output_model()
        {
            NN::ModelGraph model(Shape{6});

            SA_TRY_ASSIGN(const NN::NodeRef body,
                          model.add("body", NN::Linear{6, 8}, model.input()));
            SA_TRY_ASSIGN(const NN::NodeRef features, model.add("activation", NN::GELU{}, body));
            SA_TRY_ASSIGN(const NN::NodeRef logits, model.add("head", NN::Linear{8, 3}, features));
            SA_TRY(model.output("features", features));
            SA_TRY(model.output("logits", logits));

            return model;
        }

        /** @brief Builds a fresh registry that holds the rows register_layers adds. */
        [[nodiscard]] inline NN::NodeRegistry make_layer_registry()
        {
            NN::NodeRegistry registry;
            SA_EXPECT_OK(NN::register_layers(registry));
            return registry;
        }

        /** @brief Describes one node registry row's minimal object and the shapes around it. */
        struct MinimalCase
        {
            std::string_view type;       /**< The registry key. */
            std::string_view attributes; /**< The row without its "type" and "from". */
            std::vector<Shape> inputs;   /**< The per-sample shapes the row reads. */
            Shape output;                /**< The per-sample shape of the node's one port. */
        };

        /** @brief Views @p shapes as the inputs of one row. */
        [[nodiscard]] inline NN::ShapeSpan span_of(const std::vector<Shape>& shapes)
        {
            return NN::ShapeSpan(shapes.data(), shapes.size());
        }

        /** @brief Lists the type of every row of @p rows, in order. */
        template<typename Entry>
        [[nodiscard]] std::vector<std::string_view> types_of(SushiRuntime::span<const Entry> rows)
        {
            std::vector<std::string_view> types;
            for (const Entry& entry : rows)
                types.push_back(entry.type);

            return types;
        }

        /** @brief Builds the node @p minimal describes from @p registry, as "layer 0 '<type>'". */
        [[nodiscard]] inline Result<std::unique_ptr<NN::Node>> make_minimal(
            const NN::NodeRegistry& registry, const MinimalCase& minimal,
            std::string_view attributes)
        {
            const std::string context = "layer 0 '" + std::string(minimal.type) + "'";
            return NN::make_node(registry, minimal.type, span_of(minimal.inputs),
                                 attributes_from(attributes), context);
        }

        /** @brief Lists the operation of every node of a graph, in order. */
        using OpList = std::vector<decltype(Graph::Node::op)>;

        /** @brief Holds the batch size trace_summary traces over. */
        inline constexpr int64_t K_TRACE_BATCH = 2;

        /** @brief Holds what one training trace of a model graph left behind. */
        struct TraceSummary
        {
            OpList ops;                       /**< The operation of every traced node. */
            std::vector<std::string> outputs; /**< Each published output as "name [extents]". */
            std::vector<std::string> state;   /**< Each piece of state, as state_record() wrote it. */
        };

        /** @brief Writes one piece of state as "name [extents] kind", for a readable mismatch. */
        [[nodiscard]] inline std::string state_record(const std::string& name, const Shape& shape,
                                                      bool is_buffer)
        {
            return name + " " + shape.to_string() + (is_buffer ? " buffer" : " parameter");
        }

        /** @brief Lists every piece of state of @p model under @p prefix, in walk order. */
        [[nodiscard]] inline std::vector<std::string> state_records(NN::ModelGraph& model,
                                                                    std::string_view prefix)
        {
            std::vector<std::string> records;
            const auto on_parameter = [&records](const std::string& name, NN::Parameter& state)
            {
                records.push_back(state_record(name, state.shape(), false));
            };
            const auto on_buffer = [&records](const std::string& name, NN::Buffer& state)
            {
                records.push_back(state_record(name, state.shape(), true));
            };
            const NN::StateVisitor visitor(on_parameter, on_buffer);
            model.walk_state(prefix, visitor);

            return records;
        }

        /**
         * @brief Traces @p model in Training mode over K_TRACE_BATCH samples and summarises it.
         * @param prefix The prefix the state names are listed under.
         */
        [[nodiscard]] inline Result<TraceSummary> trace_summary(NN::ModelGraph& model,
                                                                const Shape& sample,
                                                                std::string_view prefix)
        {
            std::vector<int64_t> extents{K_TRACE_BATCH};
            for (int32_t axis = 0; axis < sample.rank(); ++axis)
                extents.push_back(sample[axis]);

            Graph::Graph graph;
            const Graph::GraphBuilder builder(graph);
            const Graph::ValueID x = builder.input(
                Shape(SushiRuntime::span<const int64_t>(extents.data(), extents.size())), "x");
            NN::BufferUpdateList updates;

            SA_TRY_ASSIGN(const NN::ModelOutputs outputs,
                          model.trace(NN::Trace::training(builder, updates), x));

            TraceSummary summary;
            for (const Graph::Node& node : graph.nodes())
                summary.ops.push_back(node.op);

            for (const NN::ModelOutputs::Entry& output : outputs.all())
                summary.outputs.push_back(output.first + " " +
                                          graph.value(output.second).shape.to_string());

            summary.state = state_records(model, prefix);
            return summary;
        }

    } // namespace Test
} // namespace SushiAI
