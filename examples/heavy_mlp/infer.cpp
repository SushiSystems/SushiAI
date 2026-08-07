/**************************************************************************/
/* infer.cpp                                                              */
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
/* permissions and limitations under the License.                        */
/**************************************************************************/

// Loads a checkpoint written by heavy_mlp_train and runs it forward-only:
// no loss, no differentiation, no optimizer. The graph is exactly the
// model's forward trace, compiled once with Graph::CompiledStep the same way
// tests/functional/integration/test_forward.cpp's OneShot helper does, and
// evaluated against a held-out batch of the same nonlinear-rings problem
// (a different seed than training used, so this is genuinely unseen data).

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/format.hpp>
#include "model.hpp"

namespace
{
    struct Settings
    {
        std::filesystem::path checkpoint = "heavy_mlp.ckpt";
        int64_t samples = 512;
        std::uint64_t seed = 0x0D15EA5E'0BADF00Full; // deliberately not the training seed
    };

    constexpr std::string_view USAGE = R"(heavy_mlp_infer - evaluate a heavy_mlp checkpoint on fresh rings samples.

Usage: heavy_mlp_infer [options]

  --checkpoint PATH   the file heavy_mlp_train wrote  (default heavy_mlp.ckpt)
  --samples N         held-out points to evaluate     (default 512)
  --seed N            seeds the held-out data (different from training's)
  -h, --help          print this and exit
)";

    Settings parse_command_line(int argc, char** argv, bool& wants_help)
    {
        Settings settings;
        wants_help = false;

        for (int i = 1; i < argc; ++i)
        {
            const std::string_view flag = argv[i];
            if (flag == "-h" || flag == "--help") { wants_help = true; return settings; }

            const auto value = [&]() -> std::string
            {
                SA_THROW_IF(i + 1 >= argc, "{} needs a value", flag);
                return argv[++i];
            };

            if (flag == "--checkpoint")     settings.checkpoint = value();
            else if (flag == "--samples")   settings.samples = std::stoll(value());
            else if (flag == "--seed")      settings.seed = static_cast<std::uint64_t>(std::stoll(value()));
            else SA_THROW_IF(true, "Unknown flag '{}'. Run with --help.", flag);
        }

        SA_THROW_IF(settings.samples <= 0, "--samples must be positive, got {}", settings.samples);
        return settings;
    }

    /** @brief The column index of the largest logit in one row. */
    int32_t argmax_row(const float* row, int64_t classes)
    {
        int32_t best = 0;
        for (int64_t c = 1; c < classes; ++c)
            if (row[c] > row[best]) best = static_cast<int32_t>(c);
        return best;
    }
}

int main(int argc, char** argv)
{
    try
    {
        bool wants_help = false;
        const Settings settings = parse_command_line(argc, argv, wants_help);
        if (wants_help) { std::cout << USAGE; return EXIT_SUCCESS; }

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        // One batch holding every held-out sample - inference has no notion
        // of an epoch, so there is nothing to gain from splitting it up.
        HeavyMLP::RingsOptions rings;
        rings.batch_size = settings.samples;
        rings.batches = 1;
        rings.seed = settings.seed;
        const HeavyMLP::RingsDataset held_out(engine, rings);
        const sa::Data::Batch& data = held_out.batch(0);

        const int64_t features = held_out.input_dim();
        const int64_t classes = held_out.num_classes();

        // --- Trace the forward pass only: no loss, no backward, no optimizer.
        sa::Graph::Graph graph;
        const sa::Graph::GraphBuilder builder(graph);
        const sa::Graph::ValueId input = builder.input(sa::Shape{settings.samples, features}, "inputs");

        HeavyMLP::Model model = HeavyMLP::make_model(features, classes);
        const sa::Graph::ValueId logits = model.forward(builder, input);

        sa::Graph::TensorPool pool(engine);
        pool.materialise_leaves(graph);
        HeavyMLP::load_checkpoint(settings.checkpoint, model, pool);

        std::memcpy(pool.tensor(input).data_as<float>(), data.inputs.data_as<float>(),
                    static_cast<std::size_t>(data.inputs.num_elements) * sizeof(float));

        sa::Graph::CompiledStep step(engine, graph, pool);
        step.run();

        const float* predicted = pool.tensor(logits).data_as<float>();
        std::size_t correct = 0;
        std::vector<int32_t> predicted_labels(static_cast<std::size_t>(settings.samples));

        for (int64_t r = 0; r < settings.samples; ++r)
        {
            const int32_t label = argmax_row(predicted + r * classes, classes);
            predicted_labels[static_cast<std::size_t>(r)] = label;
            if (label == data.labels[static_cast<std::size_t>(r)]) ++correct;
        }

        const double accuracy = static_cast<double>(correct) / static_cast<double>(settings.samples);

        std::cout << "heavy_mlp_infer\n"
                  << "----------------------------------------------------------------\n"
                  << "Checkpoint : " << settings.checkpoint << "\n"
                  << SushiBLAS::Core::format("Model      : {} -> 256 -> 256 -> 128 -> 64 -> {}\n",
                                             features, classes)
                  << SushiBLAS::Core::format("Held-out   : {} points, seed {} (unseen by training)\n\n",
                                             settings.samples, settings.seed);

        const int64_t preview = std::min<int64_t>(10, settings.samples);
        const float* x = data.inputs.data_as<float>();
        std::cout << "  x        y        true  predicted\n";
        for (int64_t r = 0; r < preview; ++r)
        {
            std::cout << SushiBLAS::Core::format(
                "  {:7.3f}  {:7.3f}  {:>4}  {:>9}{}\n",
                x[r * features], x[r * features + 1],
                data.labels[static_cast<std::size_t>(r)], predicted_labels[static_cast<std::size_t>(r)],
                predicted_labels[static_cast<std::size_t>(r)] == data.labels[static_cast<std::size_t>(r)]
                    ? "" : "  <-- miss");
        }

        std::cout << SushiBLAS::Core::format(
            "\n================================================================\n"
            " accuracy: {}/{} = {:.2f}%\n"
            "================================================================\n",
            correct, settings.samples, accuracy * 100.0);

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "heavy_mlp_infer failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
