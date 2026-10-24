/**************************************************************************/
/* dataset.cpp                                                            */
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
 * @file dataset.cpp
 * @brief Implements dataset loaders and synthetic batch generators.
 * @author Mustafa Garip
 */

#include <SushiAI/data/dataset.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /** @brief Every key a "rings" dataset accepts. */
            constexpr std::string_view K_RINGS_KEYS[] = {
                "batch_size", "batches", "classes", "distractors",
                "ring_width", "position_noise", "seed",
            };

            /** @brief Wraps a literal key list as a span, for reject_unknown_keys. */
            template<std::size_t N>
            SushiRuntime::span<const std::string_view> as_span(const std::string_view (&keys)[N])
            {
                return SushiRuntime::span<const std::string_view>(keys, N);
            }

            /**
             * @brief Reads an optional whole number from a dataset's attributes.
             * @param attributes The dataset's keys.
             * @param key Which one.
             * @param prefix What to call this dataset in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            int64_t read_int(const Config::Value& attributes, std::string_view key,
                             const std::string& prefix, int64_t fallback)
            {
                const Config::Value spelled = Config::Value::integer(fallback);
                return attributes.at_or(key, spelled)
                                 .as_int(prefix + " '" + std::string(key) + "'");
            }

            /**
             * @brief Reads an optional real from a dataset's attributes.
             * @param attributes The dataset's keys.
             * @param key Which one.
             * @param prefix What to call this dataset in a diagnostic.
             * @param fallback Returned when @p key is absent.
             * @return The value, or @p fallback.
             */
            double read_real(const Config::Value& attributes, std::string_view key,
                             const std::string& prefix, double fallback)
            {
                const Config::Value spelled = Config::Value::real(fallback);
                return attributes.at_or(key, spelled)
                                 .as_real(prefix + " '" + std::string(key) + "'");
            }

            /**
             * @brief Returns the one description a dataset of class indices publishes.
             * @return K_ROLE_LABELS as one int32 per sample over @p classes classes, or
             *         OutOfMemory when the list cannot be built.
             */
            [[nodiscard]] Result<std::vector<TargetDescription>> describe_labels(int64_t classes)
            {
                return Detail::guard_value(Detail::SeamPhase::Allocate, [classes]()
                {
                    return std::vector<TargetDescription>{TargetDescription{
                        std::string(K_ROLE_LABELS), Shape{}, DType::Int32, classes}};
                });
            }
        }

        SyntheticClassification::SyntheticClassification(SyntheticOptions options,
                                                         std::vector<TargetDescription> targets,
                                                         std::vector<Batch> batches) noexcept
            : options_(options), targets_(std::move(targets)), batches_(std::move(batches))
        {

        }

        Result<std::unique_ptr<SyntheticClassification>> SyntheticClassification::make(
            SushiBLAS::Engine& engine,
            SyntheticOptions options)
        {
            SA_FAIL_IF(options.batch_size <= 0 || options.batches <= 0
                       || options.features <= 0 || options.classes <= 0,
                       StatusCode::InvalidDescription,
                       "SyntheticClassification needs positive extents; got batch={} batches={} "
                       "features={} classes={}",
                       options.batch_size, options.batches, options.features, options.classes);

            const int64_t rows = options.batch_size;
            const int64_t features = options.features;
            const int64_t classes = options.classes;
            const InputDescription input{Shape{features}, DType::Float32};

            SA_TRY_ASSIGN(std::vector<TargetDescription> targets, describe_labels(classes));

            // Class means are shared across all batches to form a consistent task.
            const int64_t mean_dims[2] = {classes, features};
            SA_TRY_ASSIGN(SushiBLAS::Tensor means,
                          allocate_tensor(engine, SushiRuntime::span<const int64_t>(mean_dims, 2)));

            SA_TRY(Detail::record_and_run(engine, [&](SushiBLAS::TaskRecorder& recorder)
            {
                SushiBLAS::RandomOps random(recorder, engine.rng());
                random.set_seed(options.seed);
                random.normal(means, 0.0, 1.0);
                return Status();
            }));

            const int64_t noise_dims[2] = {rows, features};
            SA_TRY_ASSIGN(SushiBLAS::Tensor noise,
                          allocate_tensor(engine,
                                          SushiRuntime::span<const int64_t>(noise_dims, 2)));

            std::vector<Batch> batches;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                batches.reserve(static_cast<std::size_t>(options.batches));

                int64_t running = 0;
                for (int64_t b = 0; b < options.batches; ++b)
                {
                    SA_TRY_ASSIGN(Batch batch, allocate_batch(engine, rows, input, targets));

                    int32_t* labels = batch.targets.front().values.data_as<int32_t>();
                    const float* class_means = means.data_as<float>();
                    float* x = batch.inputs.data_as<float>();

                    for (int64_t r = 0; r < rows; ++r)
                    {
                        // Round-robin class assignment balances every batch evenly.
                        const int64_t label = (running++) % classes;
                        labels[r] = static_cast<int32_t>(label);
                        std::copy_n(class_means + label * features, features, x + r * features);
                    }

                    SA_TRY(Detail::record_and_run(engine,
                                                  [&](SushiBLAS::TaskRecorder& recorder)
                    {
                        SushiBLAS::RandomOps random(recorder, engine.rng());
                        SushiBLAS::BLASOps blas(recorder);
                        random.normal(noise, 0.0, options.noise);
                        blas.axpy(1.0f, noise, batch.inputs);
                        return Status();
                    }));

                    batches.push_back(std::move(batch));
                }
                return Status();
            }));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<SyntheticClassification>(
                    new SyntheticClassification(options, std::move(targets), std::move(batches)));
            });
        }

        Status SyntheticClassification::read(std::size_t index, Batch& batch) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return share_batch(batches_[index], batch);
        }

        Rings::Rings(RingsOptions options, std::vector<TargetDescription> targets,
                     std::vector<Batch> batches) noexcept
            : options_(options), targets_(std::move(targets)), batches_(std::move(batches))
        {

        }

        Result<std::unique_ptr<Rings>> Rings::make(SushiBLAS::Engine& engine, RingsOptions options)
        {
            SA_FAIL_IF(options.batch_size <= 0 || options.batches <= 0 || options.classes <= 0,
                       StatusCode::InvalidDescription,
                       "Rings needs a positive batch size, batch count and class count; "
                       "got {}, {} and {}",
                       options.batch_size, options.batches, options.classes);
            SA_FAIL_IF(options.distractor_features < 0, StatusCode::InvalidDescription,
                       "Rings needs a non-negative distractor count, got {}",
                       options.distractor_features);
            SA_FAIL_IF(options.ring_width <= 0.0, StatusCode::InvalidDescription,
                       "Rings needs a positive ring width, got {}", options.ring_width);
            SA_FAIL_IF(options.position_noise < 0.0, StatusCode::InvalidDescription,
                       "Rings needs a non-negative position noise, got {}",
                       options.position_noise);

            const int64_t features = 2 + options.distractor_features;
            const int64_t classes = options.classes;
            const int64_t rows = options.batch_size;
            const InputDescription input{Shape{features}, DType::Float32};

            SA_TRY_ASSIGN(std::vector<TargetDescription> targets, describe_labels(classes));

            constexpr double K_TWO_PI = 6.283185307179586476925286766559;

            // Host RNG ensures reproducible sequence generation from the seed alone.
            std::mt19937_64 rng(options.seed);
            std::uniform_real_distribution<double> theta_dist(0.0, K_TWO_PI);
            std::uniform_real_distribution<double> band_dist(0.0, options.ring_width);
            std::normal_distribution<double> jitter(0.0, options.position_noise);
            std::normal_distribution<double> distractor(0.0, 1.0);

            std::vector<Batch> batches;
            SA_TRY(Detail::guard(Detail::SeamPhase::Allocate, [&]() -> Status
            {
                batches.reserve(static_cast<std::size_t>(options.batches));

                int64_t running = 0;
                for (int64_t b = 0; b < options.batches; ++b)
                {
                    SA_TRY_ASSIGN(Batch batch, allocate_batch(engine, rows, input, targets));

                    int32_t* labels = batch.targets.front().values.data_as<int32_t>();
                    float* x = batch.inputs.data_as<float>();

                    for (int64_t r = 0; r < rows; ++r)
                    {
                        // Round-robin class assignment balances every batch evenly.
                        const int64_t label = (running++) % classes;
                        labels[r] = static_cast<int32_t>(label);

                        const double radius = static_cast<double>(label) + band_dist(rng);
                        const double theta = theta_dist(rng);

                        float* row = x + r * features;
                        row[0] = static_cast<float>(radius * std::cos(theta) + jitter(rng));
                        row[1] = static_cast<float>(radius * std::sin(theta) + jitter(rng));
                        for (int64_t f = 0; f < options.distractor_features; ++f)
                            row[2 + f] = static_cast<float>(distractor(rng));
                    }

                    batches.push_back(std::move(batch));
                }
                return Status();
            }));

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<Rings>(
                    new Rings(options, std::move(targets), std::move(batches)));
            });
        }

        Status Rings::read(std::size_t index, Batch& batch) const
        {
            SA_FATAL_IF(index >= batches_.size(),
                        "Batch {} is out of range; the dataset has {}", index, batches_.size());
            return share_batch(batches_[index], batch);
        }

        Result<std::unique_ptr<Dataset>> make_rings(SushiBLAS::Engine& engine,
                                                    const Config::Value& attributes,
                                                    std::string_view context)
        {
            attributes.reject_unknown_keys(context, as_span(K_RINGS_KEYS));

            RingsOptions options;
            const std::string prefix(context);

            options.batch_size = read_int(attributes, "batch_size", prefix, options.batch_size);
            options.batches = read_int(attributes, "batches", prefix, options.batches);
            options.classes = read_int(attributes, "classes", prefix, options.classes);
            options.distractor_features =
                read_int(attributes, "distractors", prefix, options.distractor_features);
            options.ring_width = read_real(attributes, "ring_width", prefix, options.ring_width);
            options.position_noise =
                read_real(attributes, "position_noise", prefix, options.position_noise);

            // Reinterpreting the signed integer preserves arbitrary 64-bit seed patterns.
            options.seed = static_cast<std::uint64_t>(
                read_int(attributes, "seed", prefix, static_cast<int64_t>(options.seed)));

            SA_TRY_ASSIGN(std::unique_ptr<Dataset> dataset, Rings::make(engine, options));
            return std::move(dataset);
        }

    } // namespace Data
} // namespace SushiAI
