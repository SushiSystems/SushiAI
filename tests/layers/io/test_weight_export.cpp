/**************************************************************************/
/* test_weight_export.cpp                                                 */
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
 * @file test_weight_export.cpp
 * @brief Tests the weight export: a convolution, a depthwise convolution, a batch norm and a
 *        linear written to safetensors and to PyTorch and read back by import_state_weights.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/safetensors_weight_source.hpp>
#include <SushiAI/io/safetensors_writer.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/torch_writer.hpp>
#include <SushiAI/io/weight_export.hpp>
#include <SushiAI/io/weight_import.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/shape/shape.hpp>

using namespace SushiAI;
using SushiAI::Test::require_value;

namespace
{
    /** @brief Returns an entry named @p name of @p shape and @p layout. */
    NN::StateEntry make_entry(const std::string& name, const Shape& shape,
                              NN::StateLayout layout = NN::StateLayout::Plain)
    {
        NN::StateEntry entry;
        entry.name = name;
        entry.shape = shape;
        entry.layout = layout;
        return entry;
    }

    /** @brief Returns the entries of a conv2d, a depthwise conv2d, a batch norm and a linear. */
    std::vector<NN::StateEntry> make_entries()
    {
        // The convolution kernel is [2, 2, 2, 2]: both layouts have the same shape.
        return {make_entry("conv.weight", Shape{2, 2, 2, 2}, NN::StateLayout::KernelLast),
                make_entry("conv.bias", Shape{2}),
                make_entry("dw.weight", Shape{3, 3, 2}, NN::StateLayout::DepthwiseKernelLast),
                make_entry("bn.weight", Shape{2}),
                make_entry("bn.bias", Shape{2}),
                make_entry("bn.running_mean", Shape{2}),
                make_entry("bn.running_var", Shape{2}),
                make_entry("fc.weight", Shape{3, 8}),
                make_entry("fc.bias", Shape{3})};
    }

    /** @brief Returns one float32 record per entry, holding seeded values. */
    std::vector<IO::ImportedTensor> make_records(const std::vector<NN::StateEntry>& entries)
    {
        std::vector<IO::ImportedTensor> records;
        std::size_t seed = 1;
        for (const NN::StateEntry& entry : entries)
        {
            std::vector<float> values(static_cast<std::size_t>(entry.shape.num_elements()));
            for (float& value : values)
            {
                seed = (seed * 1103515245u + 12345u) & 0x7FFFFFFFu;
                value = static_cast<float>(seed % 2001u) / 1000.0f - 1.0f;
            }
            IO::ImportedTensor record;
            record.record.name = entry.name;
            record.record.dtype = DType::Float32;
            record.record.shape = entry.shape;
            record.payload.resize(values.size() * sizeof(float));
            std::memcpy(record.payload.data(), values.data(), record.payload.size());
            records.push_back(std::move(record));
        }
        return records;
    }

    /** @brief Expects @p imported to hold @p records, name, shape and bytes alike. */
    void expect_same_records(const std::vector<IO::ImportedTensor>& imported,
                             const std::vector<IO::ImportedTensor>& records)
    {
        ASSERT_EQ(imported.size(), records.size());
        for (std::size_t i = 0; i < records.size(); ++i)
        {
            EXPECT_EQ(imported[i].record.name, records[i].record.name);
            EXPECT_EQ(imported[i].record.shape, records[i].record.shape);
            EXPECT_EQ(imported[i].payload, records[i].payload) << records[i].record.name;
        }
    }
} // namespace

/** @brief Tests that the kernels are written in PyTorch's layout and the rest as they are. */
TEST(Small_WeightExport, KernelsTakeTheSourceLayout)
{
    const std::vector<NN::StateEntry> entries = make_entries();
    const std::vector<IO::ImportedTensor> records = make_records(entries);

    SA_ASSERT_OK_AND_ASSIGN(const IO::ExportedWeights exported,
                            IO::export_state_weights(records, entries, IO::NameMap()));
    ASSERT_EQ(exported.tensors.size(), records.size());
    EXPECT_EQ(exported.tensors[2].record.shape, (Shape{2, 1, 3, 3}));
    EXPECT_EQ(exported.tensors[7].payload, records[7].payload);
    ASSERT_EQ(exported.notes.size(), 1u);
    EXPECT_NE(exported.notes[0].find("num_batches_tracked"), std::string::npos);
}

/** @brief Tests that safetensors written by the export import back bit for bit. */
TEST(Small_WeightExport, SafetensorsRoundTripsEveryRecord)
{
    const std::vector<NN::StateEntry> entries = make_entries();
    const std::vector<IO::ImportedTensor> records = make_records(entries);
    SA_ASSERT_OK_AND_ASSIGN(const IO::ExportedWeights exported,
                            IO::export_state_weights(records, entries, IO::NameMap()));
    SA_ASSERT_OK_AND_ASSIGN(std::vector<std::byte> file,
                            IO::write_safetensors(exported.tensors));

    IO::WeightSourceOptions options;
    options.origin = "written.safetensors";
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                            IO::open_safetensors_source(std::move(file), options));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            IO::import_state_weights(entries, *source, IO::NameMap()));
    expect_same_records(imported, records);
}

/** @brief Tests that a PyTorch file written by the export under renames imports back. */
TEST(Small_WeightExport, TorchRoundTripsEveryRecordThroughAMap)
{
    const std::vector<NN::StateEntry> entries = make_entries();
    const std::vector<IO::ImportedTensor> records = make_records(entries);
    SA_ASSERT_OK_AND_ASSIGN(const IO::NameMap map,
                            IO::NameMap::parse(R"({ "state_dict": "model",
                                                    "rename": [ { "from": "backbone.",
                                                                  "to": "conv." } ],
                                                    "drop": [ "aux.weight" ] })",
                                               "map"));
    SA_ASSERT_OK_AND_ASSIGN(const IO::ExportedWeights exported,
                            IO::export_state_weights(records, entries, map));
    EXPECT_EQ(exported.tensors[0].record.name, "backbone.weight");
    ASSERT_EQ(exported.notes.size(), 2u);
    EXPECT_NE(exported.notes[0].find("'aux.weight'"), std::string::npos);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<std::byte> file,
                            IO::write_torch_state_dict(exported.tensors, "model",
                                                       map.state_key()));

    IO::WeightSourceOptions options;
    options.origin = "written.pth";
    options.state_key = map.state_key();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                            IO::open_torch_source(std::move(file), options));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            IO::import_state_weights(entries, *source, map));
    expect_same_records(imported, records);
}

/** @brief Tests that a record with no entry and a record of another shape are named together. */
TEST(Small_WeightExport, RecordsThatDoNotFitAreNamedInOneMessage)
{
    const std::vector<NN::StateEntry> entries = {make_entry("fc.weight", Shape{3, 8})};
    std::vector<IO::ImportedTensor> records = make_records(
        {make_entry("fc.weight", Shape{8, 3}), make_entry("extra.bias", Shape{2})});

    SA_EXPECT_FAILURE(IO::export_state_weights(records, entries, IO::NameMap()),
                      StatusCode::Incompatible, "'fc.weight' is [8, 3]",
                      "'extra.bias' has no entry");
}
