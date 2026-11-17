/**************************************************************************/
/* test_yolox_x_import.cpp                                                */
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
 * @file test_yolox_x_import.cpp
 * @brief Tests that ByteTrack's mot17x checkpoint imports into the yolox_x model with nothing
 *        refused.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/weight_import.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/vision/registration/registration.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Names the variable that holds the folder of mot17x.pth.tar. */
    constexpr const char* K_DATA_VARIABLE = "SA_YOLOX_DATA_DIR";

    /** @brief Names the ByteTrack checkpoint in that folder. */
    constexpr const char* K_CHECKPOINT = "mot17x.pth.tar";

    /** @brief Returns the data folder, or an empty path when the variable is unset. */
    std::filesystem::path data_folder()
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        const char* const value = std::getenv(K_DATA_VARIABLE);
#pragma clang diagnostic pop
        return value == nullptr ? std::filesystem::path() : std::filesystem::path(value);
    }

    /** @brief Returns the folder of the yolox_x example. */
    std::filesystem::path example_folder()
    {
        return std::filesystem::path(SA_REPOSITORY_ROOT) / "applications" / "examples" / "yolox_x";
    }

    /** @brief Skips a test that needs the ByteTrack checkpoint when it is not there. */
    class Large_YoloxXImport : public SushiAI::Test::EngineTest
    {
        protected:
            /** @brief Skips the test when the variable or the checkpoint is missing. */
            void SetUp() override
            {
                SushiAI::Test::EngineTest::SetUp();
                folder_ = data_folder();
                if (folder_.empty())
                    GTEST_SKIP() << K_DATA_VARIABLE << " is not set";
                if (!std::filesystem::exists(folder_ / K_CHECKPOINT))
                    GTEST_SKIP() << (folder_ / K_CHECKPOINT).string() << " is missing";
            }

            std::filesystem::path folder_; /**< The data folder. */
    };
} // namespace

/** @brief Tests that the checkpoint fills every record and leaves no tensor without one. */
TEST_F(Large_YoloxXImport, TheCheckpointFillsEveryRecordAndNothingIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec spec,
                            Config::load_model(example_folder() / "model.json"));
    NN::NodeRegistry registry;
    SA_EXPECT_OK(NN::register_layers(registry));
    SA_EXPECT_OK(Vision::register_nodes(registry));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, spec));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::create(backend().engine(), graph));
    SA_ASSERT_OK_AND_ASSIGN(const IO::NameMap map,
                            IO::NameMap::read(example_folder() / "map.json"));

    IO::WeightSourceRegistry sources;
    SA_EXPECT_OK(sources.add(IO::K_TORCH_WEIGHT_SOURCE_ROW));
    const std::string origin = (folder_ / K_CHECKPOINT).string();
    IO::WeightSourceOptions options;
    options.origin = origin;
    options.state_key = map.state_key();
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                            IO::open_weight_file(sources, folder_ / K_CHECKPOINT, options));

    // A refusal is one Incompatible message that lists every record, tensor or shape at fault.
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<IO::ImportedTensor> imported,
                            IO::import_state_weights(state.entries(), *source, map));

    ASSERT_EQ(imported.size(), state.entries().size());
    for (std::size_t at = 0; at < imported.size(); ++at)
    {
        const NN::StateEntry& record = state.entries()[at];
        EXPECT_EQ(imported[at].record.name, record.name);
        EXPECT_EQ(imported[at].record.shape, record.shape) << record.name;
        EXPECT_EQ(imported[at].payload.size(),
                  static_cast<std::size_t>(record.shape.num_elements()) * dtype_size(record.dtype))
            << record.name;
    }
}
