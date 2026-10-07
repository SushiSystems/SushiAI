/**************************************************************************/
/* inference.cpp                                                          */
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
 * @file inference.cpp
 * @brief Runs `sushiai inference`: evaluates a trained checkpoint over a dataset, forward pass only.
 * @author Mustafa Garip
 * @see apps/README.md, "inference.cpp".
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/format.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view USAGE =
        "sushiai inference - evaluate a trained checkpoint over a dataset.\n"
        "\n"
        "Usage: sushiai inference <model.json> <model.ckpt> <eval.json>\n"
        "\n"
        "  model.json  the architecture the checkpoint was trained under\n"
        "  model.ckpt  the checkpoint `sushiai train` wrote\n"
        "  eval.json   the data to evaluate on; the architecture file carries none\n";

    /** @brief Returns the index of the largest of @p count values. */
    int64_t argmax_row(const float* row, int64_t count)
    {
        int64_t best = 0;
        for (int64_t i = 1; i < count; ++i)
            if (row[i] > row[best]) best = i;
        return best;
    }
}

/** @brief Runs the inference driver and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << USAGE;
            return EXIT_SUCCESS;
        }

        SA_THROW_IF(argc != 4,
                    "usage: sushiai inference <model.json> <model.ckpt> <eval.json>");

        const std::filesystem::path model_path = argv[1];
        const std::filesystem::path checkpoint_path = argv[2];
        const std::filesystem::path data_path = argv[3];

        const sa::Config::ModelSpec model_spec = sa::Config::load_model(model_path);
        const sa::Config::DataSpec data_spec = sa::Config::load_data(data_path);

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        const std::unique_ptr<sa::Data::Dataset> dataset =
            sa::Data::make_dataset(engine, data_spec.type, data_spec.attributes,
                                   "the data block");

        SA_THROW_IF(dataset->input_dim() != model_spec.input_features,
                    "'{}' declares {} input features, but the '{}' data supplies {}",
                    model_path.string(), model_spec.input_features,
                    dataset->name(), dataset->input_dim());

        // Only the forward pass is traced: no loss, no backward pass, no optimizer; see apps/README.md.
        sa::NN::LayerStack model = sa::NN::build_layer_stack(model_spec);
        const int64_t rows = dataset->batch_size();
        const int64_t outputs = model.output_features();

        sa::Graph::Graph graph;
        const sa::Graph::GraphBuilder builder(graph);
        const sa::Graph::ValueId input =
            builder.input(sa::Shape{rows, dataset->input_dim()}, "inputs");
        const sa::Graph::ValueId output = model.forward(builder, input);

        sa::Graph::TensorPool pool(engine);
        pool.materialise_leaves(graph);
        sa::IO::load_checkpoint(checkpoint_path, model, pool);

        sa::Graph::CompiledStep step(engine, graph, pool);

        std::cout << "sushiai inference\n"
                  << "----------------------------------------------------------------\n"
                  << SushiBLAS::Core::format("Model      : {} ({} layers -> {} outputs)\n",
                                             model_path.string(), model.size(), outputs)
                  << SushiBLAS::Core::format("Checkpoint : {}\n", checkpoint_path.string())
                  << SushiBLAS::Core::format("Data       : {} ({} x {})\n\n",
                                             dataset->name(), dataset->batch_count(), rows);

        // Top-1 agreement is reported only when the output width equals the class count; see apps/README.md.
        const bool labelled = model.output_features() == dataset->num_classes();

        std::vector<std::size_t> predicted_counts(static_cast<std::size_t>(outputs), 0);
        std::size_t samples = 0;
        std::size_t agreeing = 0;

        for (std::size_t b = 0; b < dataset->batch_count(); ++b)
        {
            const sa::Data::Batch& batch = dataset->batch(b);
            SushiBLAS::Tensor& destination = pool.tensor(input);

            SA_THROW_IF(batch.inputs.num_elements != destination.num_elements,
                        "batch {} holds {} input elements but the compiled step expects {}",
                        b, batch.inputs.num_elements, destination.num_elements);

            std::memcpy(destination.data(), batch.inputs.data(),
                        static_cast<std::size_t>(batch.inputs.num_elements) * sizeof(float));

            step.run();

            const float* scores = pool.tensor(output).data_as<float>();
            for (int64_t r = 0; r < rows; ++r)
            {
                const int64_t predicted = argmax_row(scores + r * outputs, outputs);
                ++predicted_counts[static_cast<std::size_t>(predicted)];
                if (labelled && predicted == batch.labels[static_cast<std::size_t>(r)])
                    ++agreeing;
                ++samples;
            }
        }

        std::cout << SushiBLAS::Core::format("Evaluated {} samples over {} batches.\n\n",
                                             samples, dataset->batch_count());
        std::cout << "  output  predicted\n";
        for (std::size_t i = 0; i < predicted_counts.size(); ++i)
            std::cout << SushiBLAS::Core::format("  {:>6}  {:>9} ({:5.2f}%)\n", i,
                                                 predicted_counts[i],
                                                 100.0 * static_cast<double>(predicted_counts[i])
                                                       / static_cast<double>(samples));

        if (labelled)
            std::cout << SushiBLAS::Core::format(
                "\ntop-1 agreement with the labels: {}/{} = {:.2f}%\n",
                agreeing, samples,
                100.0 * static_cast<double>(agreeing) / static_cast<double>(samples));
        else
            std::cout << SushiBLAS::Core::format(
                "\nThe model has {} outputs and the data {} classes, so top-1 agreement "
                "is not comparable and is not reported.\n",
                outputs, dataset->num_classes());

        std::cout << SushiBLAS::Core::format("\ncompiles: {}\n", step.compile_count());
        if (step.compile_count() != 1)
        {
            std::cerr << "The step must be compiled exactly once.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai inference failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
