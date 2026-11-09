/**************************************************************************/
/* test_import_weights.cpp                                                */
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
 * @file test_import_weights.cpp
 * @brief Tests the prefix rules of the strict loader and its refusals against a ModelState.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include "checkpoint_bytes.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_ImportWeights : public SaTest::EngineTest {};

    using SaTest::ScratchFile;

    /** @brief One record to write, with the tensor whose bytes its payload carries. */
    struct Source
    {
        /** @brief The header the record is written with. */
        IO::TensorRecord record;

        /** @brief The tensor whose bytes form the payload. */
        const SushiBLAS::Tensor* tensor;
    };

    /** @brief Returns a record called @p name with the extents @p shape, in float32. */
    IO::TensorRecord make_record(std::string name, Shape shape)
    {
        IO::TensorRecord record;
        record.name = std::move(name);
        record.shape = shape;
        return record;
    }

    /** @brief Describes an entry of a given name, identity and extents, for from_entries. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, Shape shape)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = shape;
        return entry;
    }

    /** @brief The two entries every integration case builds a state from. */
    std::vector<NN::StateEntry> two_entries()
    {
        return {make_entry("0.weight", 1, Shape{2, 3}), make_entry("0.bias", 2, Shape{3})};
    }

    /** @brief Returns the tensor of the entry @p name in @p state. */
    SushiBLAS::Tensor& tensor_of(NN::ModelState& state, std::string_view name)
    {
        return *state.tensor(name).value();
    }

    /** @brief Writes @p base + 0.5 * index into every element of the entry @p name. */
    void fill_tensor(NN::ModelState& state, std::string_view name, float base)
    {
        SushiBLAS::Tensor& tensor = tensor_of(state, name);
        float* values = tensor.data_as<float>();
        for (std::int64_t i = 0; i < tensor.num_elements; ++i)
            values[i] = base + static_cast<float>(i) * 0.5f;
    }

    /** @brief Writes a checkpoint of @p sources to @p path, in the order given. */
    void write_checkpoint(const std::filesystem::path& path, const std::vector<Source>& sources)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer,
                                IO::CheckpointWriter::create(
                                    out, static_cast<std::uint32_t>(sources.size())));
        for (const Source& source : sources)
            SA_EXPECT_OK(writer.write(source.record, source.tensor->data()));
        writer.finish();
    }
} // namespace

namespace
{
    /** @brief Holds a name and the name a rule renames it to. */
    struct RenameRow
    {
        std::string name;       /**< The row's name. */
        std::string from;       /**< The record name before the rule. */
        std::string to;         /**< The record name after the rule. */
    };

    /** @brief Runs the rename of one rule, which strips "model.", over each name. */
    class Small_RenameRuleStrip : public ::testing::TestWithParam<RenameRow>
    {
    };
} // namespace

/** @brief Tests that an empty to-prefix removes the matched prefix and keeps the rest. */
TEST_P(Small_RenameRuleStrip, AnEmptyToPrefixStripsTheMatchedPrefix)
{
    const IO::RenameRule rules[] = {{"model.", ""}};
    EXPECT_EQ(IO::renamed(GetParam().from, rules), GetParam().to);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RenameRuleStrip,
    ::testing::Values(RenameRow{"AWeight", "model.0.weight", "0.weight"},
                      RenameRow{"ABias", "model.1.bias", "1.bias"}),
    [](const ::testing::TestParamInfo<RenameRow>& info) { return info.param.name; });

/** @brief Tests that a name no rule's prefix starts is returned unchanged. */
TEST(Small_RenameRule, LeavesANameNoRuleMatches)
{
    const IO::RenameRule rules[] = {{"model.", ""}};
    EXPECT_EQ(IO::renamed("other.0.weight", rules), "other.0.weight");
}

/** @brief Tests that the first rule whose prefix matches decides the name, not the longest. */
TEST(Small_RenameRule, TheFirstMatchingRuleWins)
{
    const IO::RenameRule rules[] = {{"model.", "first."}, {"model.0", "second"}};
    EXPECT_EQ(IO::renamed("model.0.weight", rules), "first.0.weight");
}

/**
 * @brief Tests that a file written under foreign names loads through a rule, with every
 *        record landing in its entry bit for bit.
 */
TEST_F(Medium_ImportWeights, ARuleStripsAForeignPrefixAndEveryRecordLands)
{
    const ScratchFile file("sushiai_import_rule.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(source, "0.weight", 1.25f);
    fill_tensor(source, "0.bias", -3.0f);
    write_checkpoint(file.path(),
                     {{make_record("model.0.weight", Shape{2, 3}),
                       &tensor_of(source, "0.weight")},
                      {make_record("model.0.bias", Shape{3}), &tensor_of(source, "0.bias")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(target, "0.weight", 9.0f);
    fill_tensor(target, "0.bias", 9.0f);
    ASSERT_NE(SaTest::read_tensor_bytes(tensor_of(target, "0.weight")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.weight")));

    const IO::RenameRule rules[] = {{"model.", ""}};
    SA_EXPECT_OK(IO::import_weights(file.path(), target, rules));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.weight")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.weight")));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.bias")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.bias")));
}

/** @brief Tests that a path that cannot be opened is refused as FileNotFound. */
TEST_F(Medium_ImportWeights, AMissingFileIsFileNotFound)
{
    const ScratchFile file("sushiai_import_absent.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target,
                                         SushiRuntime::span<const IO::RenameRule>()),
                      StatusCode::FileNotFound, "could not open");
}

/** @brief Tests that bytes in memory load a state as the file of the same bytes loads it. */
TEST_F(Medium_ImportWeights, BytesInMemoryLoadWhatTheFileOfTheSameBytesLoads)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(source, "0.weight", 1.25f);
    fill_tensor(source, "0.bias", -3.0f);

    std::ostringstream out;
    SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer, IO::CheckpointWriter::create(out, 2));
    SA_EXPECT_OK(writer.write(make_record("0.weight", Shape{2, 3}),
                              tensor_of(source, "0.weight").data()));
    SA_EXPECT_OK(writer.write(make_record("0.bias", Shape{3}), tensor_of(source, "0.bias").data()));
    writer.finish();

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(target, "0.weight", 9.0f);
    fill_tensor(target, "0.bias", 9.0f);

    const std::string bytes = out.str();
    SA_EXPECT_OK(IO::import_weights(
        SushiRuntime::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
        "memory.sushi", target, {}));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.weight")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.weight")));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.bias")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.bias")));
}

namespace
{
    /** @brief The refusals of an import, decided on names and shapes, over bytes on the host. */
    class Small_ImportWeightsRefusal : public ::testing::Test
    {
        protected:
            /** @brief The host backend the state of a case is created on. */
            HostExecutionBackend backend;
    };

    /** @brief Holds one foreign record, whether a rule is given, and the words of its refusal. */
    struct UnmatchedRow
    {
        std::string name;       /**< The row's name. */
        std::string record;     /**< The name of the one record the file holds. */
        bool with_rule;         /**< Whether the "model." rule is given. */
        std::string first;      /**< A text the message holds. */
        std::string second;     /**< Another text the message holds. */
    };

    /** @brief Runs the refusal of a record matching no entry over each way it is renamed. */
    class Small_ImportWeightsUnmatched : public ::testing::TestWithParam<UnmatchedRow>
    {
        protected:
            /** @brief The host backend the state of a case is created on. */
            HostExecutionBackend backend;
    };
} // namespace

/** @brief Tests that a record whose name matches no entry is refused and named. */
TEST_P(Small_ImportWeightsUnmatched, ARecordTheStateLacksIsRefusedAndNamed)
{
    const UnmatchedRow& row = GetParam();
    const std::string file =
        SaTest::checkpoint_bytes({IO::TensorRecord{row.record, DType::Float32, Shape{2, 3}}});
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(backend.engine(), two_entries()));

    const IO::RenameRule rules[] = {{"model.", ""}};
    const SushiRuntime::span<const IO::RenameRule> given =
        row.with_rule ? SushiRuntime::span<const IO::RenameRule>(rules)
                      : SushiRuntime::span<const IO::RenameRule>();
    SA_EXPECT_FAILURE(IO::import_weights(SaTest::as_bytes(file), "foreign.sushi", target, given),
                      StatusCode::Incompatible, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_ImportWeightsUnmatched,
    ::testing::Values(UnmatchedRow{"WithoutTheRule", "model.0.weight", false,
                                   "'model.0.weight'", ""},
                      UnmatchedRow{"UnderBothNames", "model.9.weight", true,
                                   "holds 'model.9.weight'", "renames to '9.weight'"}),
    [](const ::testing::TestParamInfo<UnmatchedRow>& info) { return info.param.name; });

/** @brief Tests that an entry no record supplies is refused, and the refusal names it. */
TEST_F(Small_ImportWeightsRefusal, AnEntryTheFileLacksIsRefusedAndNamed)
{
    const std::string file = SaTest::checkpoint_bytes(
        {IO::TensorRecord{"model.0.weight", DType::Float32, Shape{2, 3}}});
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(backend.engine(), two_entries()));

    const IO::RenameRule rules[] = {{"model.", ""}};
    SA_EXPECT_FAILURE(IO::import_weights(SaTest::as_bytes(file), "partial.sushi", target, rules),
                      StatusCode::Incompatible, "does not supply", "'0.bias'");
}

/** @brief Tests that a record whose shape differs from its entry is refused with both shapes. */
TEST_F(Small_ImportWeightsRefusal, AShapeMismatchIsRefusedWithBothShapes)
{
    // The same six elements, under a shape the entry does not have.
    const std::string file = SaTest::checkpoint_bytes(
        {IO::TensorRecord{"model.0.weight", DType::Float32, Shape{3, 2}}});
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(backend.engine(), two_entries()));

    const IO::RenameRule rules[] = {{"model.", ""}};
    const std::string file_shape = Shape{3, 2}.to_string();
    const std::string model_shape = Shape{2, 3}.to_string();
    SA_EXPECT_FAILURE(IO::import_weights(SaTest::as_bytes(file), "wrong.sushi", target, rules),
                      StatusCode::Incompatible, file_shape, model_shape);
}
