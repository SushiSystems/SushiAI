/**************************************************************************/
/* model.hpp                                                              */
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

// Shared between heavy_mlp_train.cpp and heavy_mlp_infer.cpp: the model
// architecture, the nonlinear dataset both programs train/evaluate on, and a
// tiny checkpoint format so a set of trained weights can cross the process
// boundary between the two.
//
// THE PROBLEM
//
// Four concentric rings in a 2D plane (x, y), each ring a class. The decision
// boundary is a degree-2 polynomial in the inputs (class(x, y) is a function
// of x^2 + y^2), so no linear classifier can separate it - a single Linear
// layer tops out near chance accuracy on this data no matter how it is
// trained. That is the whole reason this demo's MLP is "heavy" (four hidden
// layers) rather than the one-hidden-layer network in demo_mlp.cpp: a deep
// stack of GELUs is what turns the radius into something the final Linear
// layer can cut with hyperplanes. A handful of extra, purely-Gaussian
// "distractor" features ride along with (x, y) so the network also has to
// learn which inputs are signal.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

namespace HeavyMLP
{
    /** @brief The architecture both programs must agree on, bit for bit. */
    using Model = sa::NN::Sequential<sa::NN::Linear, sa::NN::GELU,
                                     sa::NN::Linear, sa::NN::GELU,
                                     sa::NN::Linear, sa::NN::GELU,
                                     sa::NN::Linear, sa::NN::GELU,
                                     sa::NN::Linear>;

    /** @brief features -> 256 -> 256 -> 128 -> 64 -> classes. */
    inline Model make_model(int64_t features, int64_t classes)
    {
        return Model{
            sa::NN::Linear{features, 256}, sa::NN::GELU{},
            sa::NN::Linear{256, 256},      sa::NN::GELU{},
            sa::NN::Linear{256, 128},      sa::NN::GELU{},
            sa::NN::Linear{128, 64},       sa::NN::GELU{},
            sa::NN::Linear{64, classes},
        };
    }

    // ------------------------------------------------------------------
    // The nonlinear-rings dataset
    // ------------------------------------------------------------------

    /** @brief How to shape the rings problem. */
    struct RingsOptions
    {
        int64_t batch_size = 64;
        int64_t batches = 64;
        int64_t classes = 4;          /**< One class per ring. */
        int64_t distractor_features = 4; /**< Pure Gaussian noise, no signal. */
        double ring_width = 1.0;      /**< Ring c spans radius [c, c + ring_width). */
        double position_noise = 0.12; /**< Gaussian jitter added to x and y. */
        std::uint64_t seed = 0x0D15EA5E'0BADF00Dull;
    };

    /**
     * @class RingsDataset
     * @brief Concentric rings, labelled by which ring a point falls in.
     *
     * Generated entirely on the host with a seeded std::mt19937_64 - every
     * SushiBLAS tensor here is backed by shared USM (see dataset.cpp's MNIST
     * loader for the same idiom), so writing through data_as<float>() is
     * enough; no device RNG or kernel launch is needed for synthetic host
     * data like this.
     */
    class RingsDataset final : public sa::Data::Dataset
    {
        public:
            RingsDataset(SushiBLAS::Engine& engine, RingsOptions options = {})
                : options_(options)
            {
                SA_THROW_IF(options_.batch_size <= 0 || options_.batches <= 0
                            || options_.classes <= 0 || options_.distractor_features < 0,
                            "RingsDataset needs positive batch/batches/classes and a "
                            "non-negative distractor count");

                const int64_t features = input_dim();
                const int64_t classes = options_.classes;
                const int64_t rows = options_.batch_size;

                constexpr double TWO_PI = 6.283185307179586476925286766559;

                std::mt19937_64 rng(options_.seed);
                std::uniform_real_distribution<double> theta_dist(0.0, TWO_PI);
                std::uniform_real_distribution<double> band_dist(0.0, options_.ring_width);
                std::normal_distribution<double> jitter(0.0, options_.position_noise);
                std::normal_distribution<double> distractor(0.0, 1.0);

                batches_.reserve(static_cast<std::size_t>(options_.batches));

                int64_t running = 0;
                for (int64_t b = 0; b < options_.batches; ++b)
                {
                    sa::Data::Batch batch;
                    const int64_t sample_dims[2] = {rows, features};
                    const int64_t target_dims[2] = {rows, classes};
                    batch.inputs = sa::allocate_tensor(
                        engine, SushiRuntime::span<const int64_t>(sample_dims, 2));
                    batch.targets = sa::allocate_tensor(
                        engine, SushiRuntime::span<const int64_t>(target_dims, 2));
                    batch.labels.resize(static_cast<std::size_t>(rows));

                    float* x = batch.inputs.data_as<float>();
                    float* t = batch.targets.data_as<float>();
                    std::fill(t, t + batch.targets.num_elements, 0.0f);

                    for (int64_t r = 0; r < rows; ++r)
                    {
                        // Round-robin over classes, exactly like
                        // SyntheticClassification, so every batch is balanced
                        // and accuracy needs no class weighting to read right.
                        const int64_t label = (running++) % classes;
                        batch.labels[static_cast<std::size_t>(r)] = static_cast<int32_t>(label);

                        const double radius = static_cast<double>(label) + band_dist(rng);
                        const double theta = theta_dist(rng);

                        float* row = x + r * features;
                        row[0] = static_cast<float>(radius * std::cos(theta) + jitter(rng));
                        row[1] = static_cast<float>(radius * std::sin(theta) + jitter(rng));
                        for (int64_t f = 0; f < options_.distractor_features; ++f)
                            row[2 + f] = static_cast<float>(distractor(rng));

                        t[r * classes + label] = 1.0f;
                    }

                    batches_.push_back(std::move(batch));
                }
            }

            int64_t batch_size() const noexcept override { return options_.batch_size; }
            int64_t input_dim() const noexcept override { return 2 + options_.distractor_features; }
            int64_t num_classes() const noexcept override { return options_.classes; }
            std::size_t batch_count() const noexcept override { return batches_.size(); }
            std::string_view name() const noexcept override { return "nonlinear-rings"; }

            const sa::Data::Batch& batch(std::size_t index) const override
            {
                SA_THROW_IF(index >= batches_.size(),
                            "Batch {} is out of range; the dataset has {}", index, batches_.size());
                return batches_[index];
            }

        private:
            RingsOptions options_;
            std::vector<sa::Data::Batch> batches_;
    };

    // ------------------------------------------------------------------
    // Checkpointing
    // ------------------------------------------------------------------
    //
    // A minimal, framework-specific binary format - not portable, not
    // versioned, and that is fine: its only job is to move the values
    // sa::NN::for_each_parameter() enumerates from a training process to an
    // inference process that traces the identical architecture, matching
    // entries by their dotted parameter name ("0.weight", "0.bias", ...).

    inline constexpr std::uint32_t CHECKPOINT_MAGIC = 0x504C4D48u; // "HMLP"

    /**
     * @brief Writes every parameter @p module currently holds, by name.
     * @param path Where to write the checkpoint.
     * @param module The trained model; must already be traced (its
     *        Parameters must be declared) and initialised.
     * @param pool Where the parameters' tensors live.
     */
    template<typename M>
    void save_checkpoint(const std::filesystem::path& path, M& module,
                         sa::Graph::TensorPool& pool)
    {
        std::ofstream out(path, std::ios::binary);
        SA_THROW_IF(!out, "could not open '{}' for writing", path.string());

        std::vector<std::pair<std::string, SushiBLAS::Tensor>> entries;
        sa::NN::for_each_parameter(module, [&](const std::string& name, sa::NN::Parameter& parameter)
        {
            entries.emplace_back(name, pool.tensor(parameter.id()));
        });

        const std::uint32_t magic = CHECKPOINT_MAGIC;
        const std::uint32_t count = static_cast<std::uint32_t>(entries.size());
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& [name, tensor] : entries)
        {
            const std::uint32_t name_len = static_cast<std::uint32_t>(name.size());
            const std::uint32_t elements = static_cast<std::uint32_t>(tensor.num_elements);
            out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            out.write(name.data(), name_len);
            out.write(reinterpret_cast<const char*>(&elements), sizeof(elements));
            out.write(reinterpret_cast<const char*>(tensor.data_as<float>()),
                       static_cast<std::streamsize>(elements * sizeof(float)));
        }

        SA_THROW_IF(!out, "failed writing checkpoint to '{}'", path.string());
    }

    /**
     * @brief Loads a checkpoint into @p module's already-materialised parameters.
     * @param path The checkpoint written by save_checkpoint().
     * @param module A freshly traced model of the identical architecture.
     * @param pool Where the parameters' tensors were materialised.
     * @throws Error If a name is missing, an element count disagrees, or the
     *         file cannot be read.
     */
    template<typename M>
    void load_checkpoint(const std::filesystem::path& path, M& module,
                         sa::Graph::TensorPool& pool)
    {
        std::ifstream in(path, std::ios::binary);
        SA_THROW_IF(!in, "could not open '{}' for reading", path.string());

        std::uint32_t magic = 0, count = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        SA_THROW_IF(!in || magic != CHECKPOINT_MAGIC,
                    "'{}' is not a heavy_mlp checkpoint", path.string());

        std::vector<std::pair<std::string, float*>> targets;
        std::vector<std::size_t> target_sizes;
        sa::NN::for_each_parameter(module, [&](const std::string& name, sa::NN::Parameter& parameter)
        {
            SushiBLAS::Tensor& tensor = pool.tensor(parameter.id());
            targets.emplace_back(name, tensor.data_as<float>());
            target_sizes.push_back(static_cast<std::size_t>(tensor.num_elements));
        });

        std::size_t matched = 0;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            std::uint32_t name_len = 0;
            in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            in.read(name.data(), name_len);

            std::uint32_t elements = 0;
            in.read(reinterpret_cast<char*>(&elements), sizeof(elements));
            std::vector<float> payload(elements);
            in.read(reinterpret_cast<char*>(payload.data()),
                    static_cast<std::streamsize>(elements * sizeof(float)));
            SA_THROW_IF(!in, "'{}' ended early while reading '{}'", path.string(), name);

            const auto it = std::find_if(targets.begin(), targets.end(),
                                         [&](const auto& kv) { return kv.first == name; });
            SA_THROW_IF(it == targets.end(),
                        "checkpoint has parameter '{}', which this model does not", name);

            const std::size_t index = static_cast<std::size_t>(it - targets.begin());
            SA_THROW_IF(target_sizes[index] != elements,
                        "'{}' holds {} elements for '{}', the model expects {}",
                        path.string(), elements, name, target_sizes[index]);

            std::memcpy(it->second, payload.data(), elements * sizeof(float));
            ++matched;
        }

        SA_THROW_IF(matched != targets.size(),
                    "'{}' supplies {} of this model's {} parameters",
                    path.string(), matched, targets.size());
    }

} // namespace HeavyMLP
