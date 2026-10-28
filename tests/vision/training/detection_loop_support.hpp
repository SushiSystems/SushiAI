/**************************************************************************/
/* detection_loop_support.hpp                                             */
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
 * @file detection_loop_support.hpp
 * @brief Declares the truth datasets and the epoch recorder the detection loop tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

/** @brief Holds the datasets and the epoch recorder the detection loop tests share. */
namespace SushiAI::DetectionLoopTest
{
    /** @brief Holds the pixel corners of the slots of both images; the last slot is padding. */
    inline const std::vector<float> K_CORNERS = {1.0f, 1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 8.0f, 7.0f,
                                                 4.0f, 4.0f, 6.0f, 7.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    /** @brief Holds the class of each slot; the padding slot holds 0. */
    inline const std::vector<std::int32_t> K_INDICES = {0, 1, 1, 0};

    /** @brief Holds 1 for each slot in use and 0 for the padding slot. */
    inline const std::vector<float> K_FLAGS = {1.0f, 1.0f, 1.0f, 0.0f};

    /** @brief Writes the corners, classes and flags above into the three roles of @p batch. */
    inline void write_truth(Data::Batch& batch)
    {
        std::copy(K_CORNERS.begin(), K_CORNERS.end(), batch.targets[0].values.data_as<float>());
        std::copy(K_INDICES.begin(), K_INDICES.end(),
                  batch.targets[1].values.data_as<std::int32_t>());
        std::copy(K_FLAGS.begin(), K_FLAGS.end(), batch.targets[2].values.data_as<float>());
    }

    /**
     * @brief Returns a dataset of @p batches batches of @p rows seeded images of @p side pixels.
     * @param targets The roles every batch carries; the three truth roles get the values above
     *                when @p rows equals @p truth_rows and there are three roles.
     * @param seed Seeds the pixels.
     */
    inline Data::MaterialisedDataset make_truth_dataset(
        SushiBLAS::Engine& engine, std::size_t batches, std::int64_t rows,
        const std::vector<Data::TargetDescription>& targets, std::int64_t side,
        std::uint64_t seed, std::int64_t truth_rows)
    {
        const Shape sample{side, side, 3};
        std::vector<Data::Batch> held;
        Test::HostNoise noise(seed);

        for (std::size_t index = 0; index < batches; ++index)
        {
            Result<Data::Batch> batch = Data::allocate_batch(
                engine, rows, Data::InputDescription{sample, DType::Float32}, targets);
            EXPECT_TRUE(batch.status().is_ok());

            std::vector<double> pixels(
                static_cast<std::size_t>(batch.value().inputs.num_elements));
            noise.fill(pixels);
            Test::write_tensor(batch.value().inputs, pixels);

            if (rows == truth_rows && targets.size() == 3)
                write_truth(batch.value());
            held.push_back(std::move(batch).value());
        }

        return Data::MaterialisedDataset(std::move(held), rows,
                                         Data::InputDescription{sample, DType::Float32}, targets,
                                         "truths");
    }

    /** @brief Reads through another dataset and keeps the epoch of every begin_epoch(). */
    class EpochRecorder final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /** @brief Reads through @p inner, which must outlive this. */
            explicit EpochRecorder(const Data::Dataset& inner) : inner_(&inner)
            {

            }

            int64_t batch_size() const noexcept override { return inner_->batch_size(); }
            Shape sample_shape() const override { return inner_->sample_shape(); }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return inner_->targets();
            }
            std::size_t batch_count() const noexcept override { return inner_->batch_count(); }
            std::string_view name() const noexcept override { return inner_->name(); }

            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override
            {
                return inner_->read(index, batch);
            }

            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override
            {
                return inner_->read_rows(index, rows, batch);
            }

            void begin_epoch(std::int64_t epoch) const noexcept override
            {
                epochs_.push_back(epoch);
            }

            /** @brief Returns the epochs begun, in order. */
            [[nodiscard]] const std::vector<std::int64_t>& epochs() const noexcept
            {
                return epochs_;
            }

        private:
            /** @brief Points at the dataset the batches come from. */
            const Data::Dataset* inner_;

            /** @brief Holds the epoch of each begin_epoch(), in order. */
            mutable std::vector<std::int64_t> epochs_;
    };
} // namespace SushiAI::DetectionLoopTest
