/**************************************************************************/
/* train.cpp                                                              */
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

// Trains the heavy_mlp architecture (see model.hpp) on the nonlinear rings
// problem and writes a checkpoint that heavy_mlp_infer can load. Everything
// but the model and the dataset is the same shape as examples/example.cpp:
// trace, differentiate, compile once, replay per batch.

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
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
        int64_t epochs = 50;
        int64_t batches = 64;
        int64_t batch_size = 64;
        double learning_rate = 3e-3;
        std::uint64_t seed = 0x0D15EA5E'0BADF00Dull;
        std::filesystem::path checkpoint = "heavy_mlp.ckpt";
    };

    int64_t parse_int(std::string_view flag, const std::string& text)
    {
        try
        {
            std::size_t consumed = 0;
            const long long parsed = std::stoll(text, &consumed);
            if (consumed == text.size()) return static_cast<int64_t>(parsed);
        }
        catch (const std::logic_error&) {}
        SA_THROW_IF(true, "{} expects a whole number, got '{}'", flag, text);
        return 0;
    }

    double parse_real(std::string_view flag, const std::string& text)
    {
        try
        {
            std::size_t consumed = 0;
            const double parsed = std::stod(text, &consumed);
            if (consumed == text.size()) return parsed;
        }
        catch (const std::logic_error&) {}
        SA_THROW_IF(true, "{} expects a number, got '{}'", flag, text);
        return 0.0;
    }

    constexpr std::string_view USAGE = R"(heavy_mlp_train - train a 4-hidden-layer MLP on the nonlinear rings problem.

Usage: heavy_mlp_train [options]

  --epochs N       passes over the dataset          (default 30)
  --batches N      batches per epoch                (default 64)
  --batch-size N   samples per batch                (default 64)
  --lr F           AdamW learning rate               (default 3e-3)
  --seed N         seeds the data and the weights
  --out PATH       where to write the checkpoint    (default heavy_mlp.ckpt)
  -h, --help       print this and exit
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

            if (flag == "--epochs")          settings.epochs = parse_int(flag, value());
            else if (flag == "--batches")    settings.batches = parse_int(flag, value());
            else if (flag == "--batch-size") settings.batch_size = parse_int(flag, value());
            else if (flag == "--lr")         settings.learning_rate = parse_real(flag, value());
            else if (flag == "--seed")       settings.seed = static_cast<std::uint64_t>(parse_int(flag, value()));
            else if (flag == "--out")        settings.checkpoint = value();
            else SA_THROW_IF(true, "Unknown flag '{}'. Run with --help.", flag);
        }

        SA_THROW_IF(settings.epochs <= 0, "--epochs must be positive, got {}", settings.epochs);
        SA_THROW_IF(settings.batches <= 0, "--batches must be positive, got {}", settings.batches);
        SA_THROW_IF(settings.batch_size <= 0, "--batch-size must be positive, got {}", settings.batch_size);
        SA_THROW_IF(settings.learning_rate <= 0.0, "--lr must be positive, got {}", settings.learning_rate);
        return settings;
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

        // --- Data: four concentric rings, not linearly separable ----------
        HeavyMLP::RingsOptions rings;
        rings.batch_size = settings.batch_size;
        rings.batches = settings.batches;
        rings.seed = settings.seed;
        const HeavyMLP::RingsDataset dataset(engine, rings);

        const int64_t features = dataset.input_dim();
        const int64_t classes = dataset.num_classes();

        // --- Model: features -> 256 -> 256 -> 128 -> 64 -> classes --------
        HeavyMLP::Model model = HeavyMLP::make_model(features, classes);

        sa::Train::TrainingGraph built = sa::Train::build_classifier(model, settings.batch_size,
                                                                      features, classes);

        sa::Graph::TensorPool pool(engine);
        std::vector<sa::Optim::ParameterSlot> parameters = sa::Train::bind_parameters(built, pool);

        // Kaiming, since GELU is the rectifier family's nonlinear cousin and
        // every hidden layer here is GELU.
        sa::NN::InitPolicy init;
        init.seed = settings.seed;
        sa::NN::initialize(engine, model, pool, init);

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = settings.learning_rate;
        sa::Optim::AdamW optimizer(engine, parameters, adam);

        sa::Train::TrainerOptions options;
        options.epochs = 1;
        sa::Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        std::cout << "heavy_mlp_train - configuration\n"
                  << "----------------------------------------------------------------\n"
                  << "SushiAI  : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << "Data     : " << dataset.name() << " (4 rings, not linearly separable)\n"
                  << SushiBLAS::Core::format("Model    : {} -> 256 -> 256 -> 128 -> 64 -> {}  ({} learnable scalars)\n",
                                             features, classes, sa::NN::parameter_elements(model))
                  << SushiBLAS::Core::format("Batches  : {} x {}\n", dataset.batch_count(), settings.batch_size)
                  << SushiBLAS::Core::format("Graph    : {} IR nodes over {} values\n",
                                             built.graph.node_count(), built.graph.value_count())
                  << SushiBLAS::Core::format("Schedule : {} epochs, lr {:g}, seed {}\n\n",
                                             settings.epochs, settings.learning_rate, settings.seed);

        for (int64_t epoch = 1; epoch <= settings.epochs; ++epoch)
        {
            const sa::Train::RunReport report = trainer.fit(dataset);
            if (epoch == 1 || epoch % 5 == 0 || epoch == settings.epochs)
                std::cout << SushiBLAS::Core::format(
                    "  epoch {:>3}/{:<3} loss {:.6f}   accuracy {:6.2f}%\n",
                    epoch, settings.epochs, report.last_loss, report.last_accuracy * 100.0);
        }

        if (trainer.compile_count() != 1)
        {
            std::cerr << "\nThe step was compiled " << trainer.compile_count()
                      << " times; it must be compiled exactly once.\n";
            return EXIT_FAILURE;
        }

        HeavyMLP::save_checkpoint(settings.checkpoint, model, pool);
        std::cout << "\nWrote " << settings.checkpoint << " ("
                  << sa::NN::parameter_elements(model) << " scalars).\n"
                  << "Run heavy_mlp_infer --checkpoint " << settings.checkpoint
                  << " to evaluate it.\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "heavy_mlp_train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
