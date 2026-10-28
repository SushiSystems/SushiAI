/**************************************************************************/
/* darknet_import.cpp                                                     */
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
 * @file darknet_import.cpp
 * @brief Implements the import of a Darknet model and the checks against its dump.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

namespace SushiAI
{
    namespace Test
    {
        namespace
        {
            using Vision::Detect::Anchor::AnchorTask;
            using Vision::Importers::DarknetNormalisation;

            /** @brief Returns why `<stem>.cfg` or `<stem>.weights` is missing, or nothing. */
            std::string find_missing_model_file(std::string_view stem)
            {
                for (const std::string_view extension : {".cfg", ".weights"})
                {
                    const DarknetFile file =
                        find_darknet_file(std::string(stem) + std::string(extension));
                    if (!file.found)
                        return file.reason;
                }

                return std::string();
            }

            /**
             * @brief Builds the model of @p spec over the rows the layers tier registers and
             *        traces it once, so every layer has declared the entries of its state.
             */
            Result<NN::ModelGraph> build_graph(const Config::ModelSpec& spec)
            {
                NN::NodeRegistry registry;
                SA_TRY(NN::register_layers(registry));
                SA_TRY_ASSIGN(NN::ModelGraph graph, NN::build_model(registry, spec));

                SA_TRY_ASSIGN(const Shape batch, Train::batch_shape(1, spec.input_shape));
                SA_TRY_ASSIGN([[maybe_unused]] const Train::TracedStep traced,
                              Train::build_evaluation_graph(graph, batch));
                return Result<NN::ModelGraph>(std::move(graph));
            }

            /** @brief Checks that each entry of @p found holds the bytes of its twin. */
            void expect_same_state(NN::ModelState& found, NN::ModelState& wanted)
            {
                const SushiRuntime::span<const NN::StateEntry> entries = wanted.entries();
                ASSERT_EQ(found.entries().size(), entries.size());
                for (std::size_t at = 0; at < entries.size(); ++at)
                {
                    const NN::StateEntry& entry = entries[at];
                    ASSERT_EQ(found.entries()[at].name, entry.name);
                    ASSERT_EQ(found.entries()[at].shape, entry.shape) << entry.name;

                    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const expected,
                                            wanted.tensor(entry.name));
                    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const actual,
                                            found.tensor(entry.name));
                    const std::size_t bytes =
                        static_cast<std::size_t>(expected->num_elements) * dtype_size(entry.dtype);
                    EXPECT_EQ(std::memcmp(actual->data(), expected->data(), bytes), 0)
                        << entry.name << " differs between the imported and the direct state";
                }
            }

        } // namespace

        Result<DarknetImportRun> import_darknet_stem(std::string_view stem)
        {
            DarknetImportRun run;
            SA_TRY_ASSIGN(run.config, load_darknet_config_file(stem));
            SA_TRY_ASSIGN(run.model, translate_darknet_file(stem, DarknetNormalisation::Separate));

            const DarknetFile file = find_darknet_file(std::string(stem) + ".weights");
            SA_FAIL_IF(!file.found, StatusCode::FileNotFound, "{}", file.reason);
            SA_TRY_ASSIGN(run.weights, Vision::Importers::read_darknet_weights(
                                           file.path, run.model.convolutions));

            SA_TRY_ASSIGN(run.imported, Vision::Importers::import_darknet_model(
                                            run.config, run.model, &run.weights));
            return run;
        }

        Status fill_imported_state(SushiRuntime::span<const IO::ImportedTensor> tensors,
                                   NN::ModelState& state)
        {
            std::unordered_set<std::string_view> supplied;
            for (const IO::ImportedTensor& tensor : tensors)
            {
                const IO::TensorRecord& record = tensor.record;
                const Result<SushiBLAS::Tensor*> found = state.tensor(record.name);
                SA_FAIL_IF(!found.has_value(), StatusCode::Incompatible,
                           "the imported model holds '{}', and the built model has no such entry",
                           record.name);
                SushiBLAS::Tensor* const entry = found.value();

                SA_FAIL_IF(!supplied.insert(record.name).second, StatusCode::Incompatible,
                           "the imported model supplies '{}' twice", record.name);
                SA_FAIL_IF(record.dtype != entry->dtype, StatusCode::Incompatible,
                           "the imported model holds '{}' as {}, but the built model's is {}",
                           record.name, dtype_name(record.dtype), dtype_name(entry->dtype));
                SA_FAIL_IF(record.shape != shape_of(*entry), StatusCode::Incompatible,
                           "the imported model holds '{}' with shape {}, but the built model's "
                           "is {}",
                           record.name, record.shape.to_string(), shape_of(*entry).to_string());
                SA_FAIL_IF(tensor.payload.size() != record.payload_bytes(),
                           StatusCode::Incompatible, "'{}' holds {} bytes, but its shape needs {}",
                           record.name, tensor.payload.size(), record.payload_bytes());

                std::memcpy(entry->data(), tensor.payload.data(), tensor.payload.size());
            }

            for (const NN::StateEntry& entry : state.entries())
            {
                SA_FAIL_IF(supplied.count(entry.name) == 0, StatusCode::Incompatible,
                           "the imported model does not supply the built model's '{}'",
                           entry.name);
            }

            return Status();
        }

        void check_imported_rows(const DarknetImportRun& run)
        {
            Config::ModelSpec rows = run.imported.model;
            rows.task = Config::Value();
            SA_ASSERT_OK_AND_ASSIGN(const std::string found, Config::write_model(rows));
            SA_ASSERT_OK_AND_ASSIGN(const std::string wanted, Config::write_model(run.model.spec));
            EXPECT_EQ(found, wanted);

            SA_ASSERT_OK_AND_ASSIGN(
                const AnchorTask task,
                Vision::Detect::Anchor::read_anchor_task(run.imported.model.task));
            EXPECT_EQ(task.layout.head_anchors, run.model.layout.head_anchors);
            EXPECT_EQ(task.criterion, run.model.criterion);

            ASSERT_EQ(task.layout.anchors.size(), run.model.layout.anchors.size());
            for (std::size_t at = 0; at < task.layout.anchors.size(); ++at)
            {
                EXPECT_EQ(task.layout.anchors[at].width, run.model.layout.anchors[at].width);
                EXPECT_EQ(task.layout.anchors[at].height, run.model.layout.anchors[at].height);
            }

            ASSERT_EQ(task.heads.size(), run.model.heads.size());
            for (std::size_t k = 0; k < task.heads.size(); ++k)
            {
                const Vision::Detect::Anchor::HeadOptions& wanted_head =
                    run.model.heads[k].options;
                EXPECT_EQ(task.heads[k].output, run.model.heads[k].output);
                EXPECT_EQ(task.heads[k].options.num_classes, wanted_head.num_classes);
                EXPECT_EQ(task.heads[k].options.coding, wanted_head.coding);
                EXPECT_EQ(task.heads[k].options.scale_xy, wanted_head.scale_xy);
                EXPECT_EQ(task.num_classes, wanted_head.num_classes);
            }
        }

        void check_imported_weights(SushiBLAS::Engine& engine, const DarknetImportRun& run)
        {
            SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph direct_graph, build_graph(run.model.spec));
            SA_ASSERT_OK_AND_ASSIGN(NN::ModelState direct,
                                    NN::ModelState::create(engine, direct_graph));
            ASSERT_TRUE(status_is_ok(Vision::Importers::store_darknet_weights(
                run.weights, run.model.convolutions, run.model.targets, direct)));

            SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph imported_graph,
                                    build_graph(run.imported.model));
            SA_ASSERT_OK_AND_ASSIGN(NN::ModelState imported,
                                    NN::ModelState::create(engine, imported_graph));
            ASSERT_TRUE(status_is_ok(fill_imported_state(run.imported.weights, imported)));

            expect_same_state(imported, direct);
        }

        void check_written_import(SushiBLAS::Engine& engine, const DarknetImportRun& run,
                                  const std::filesystem::path& directory)
        {
            SA_ASSERT_OK_AND_ASSIGN(const IO::WrittenImport written,
                                    IO::write_imported_model(run.imported, directory,
                                                             IO::ExistingFiles::Refuse));
            EXPECT_EQ(written.files.size(), 3u);

            SA_ASSERT_OK_AND_ASSIGN(const Config::ModelSpec model,
                                    Config::load_model(directory / IO::K_IMPORTED_MODEL_FILE));
            SA_ASSERT_OK_AND_ASSIGN(const std::string found, Config::write_model(model));
            SA_ASSERT_OK_AND_ASSIGN(const std::string wanted,
                                    Config::write_model(run.imported.model));
            EXPECT_EQ(found, wanted);

            SA_ASSERT_OK_AND_ASSIGN(
                const Config::Value training,
                Config::parse_json_file(directory / IO::K_IMPORTED_TRAINING_FILE));
            SA_ASSERT_OK_AND_ASSIGN(const Config::Value composed,
                                    IO::compose_training_file(run.imported));
            SA_ASSERT_OK_AND_ASSIGN(const std::string read_back, Config::write_json(training));
            SA_ASSERT_OK_AND_ASSIGN(const std::string stated, Config::write_json(composed));
            EXPECT_EQ(read_back, stated);

            SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph direct_graph, build_graph(run.model.spec));
            SA_ASSERT_OK_AND_ASSIGN(NN::ModelState direct,
                                    NN::ModelState::create(engine, direct_graph));
            ASSERT_TRUE(status_is_ok(Vision::Importers::store_darknet_weights(
                run.weights, run.model.convolutions, run.model.targets, direct)));

            SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph loaded_graph, build_graph(model));
            SA_ASSERT_OK_AND_ASSIGN(NN::ModelState loaded,
                                    NN::ModelState::create(engine, loaded_graph));
            ASSERT_TRUE(status_is_ok(
                IO::load_checkpoint(directory / IO::K_IMPORTED_WEIGHTS_FILE, loaded)));

            expect_same_state(loaded, direct);
        }

        void DarknetImportTest::SetUp()
        {
            EngineTest::SetUp();
            const std::string missing = find_missing_model_file(GetParam());
            if (!missing.empty())
                GTEST_SKIP() << missing;

            SA_ASSERT_OK_AND_ASSIGN(run, import_darknet_stem(GetParam()));
        }

    } // namespace Test
} // namespace SushiAI
