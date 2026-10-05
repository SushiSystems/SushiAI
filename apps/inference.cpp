/****************************************************************/
/* inference.cpp                                                */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// `sushiai inference <model.json> <model.ckpt> <eval.json>`
//
// Three positional arguments, and the data is one of them. The architecture
// file carries no data block - it describes a model - so without it there
// would be nothing to run on, and an optional flag that always errors when
// omitted would be a lie about optionality.
//
// This traces the FORWARD PASS ONLY: no loss, no backward, no optimizer. There
// is no objective here at all, and that is not an omission - inference does not
// minimise anything, and asking a config for a loss it will not use would be
// asking for a value with no meaning.
//
// Like apps/train.cpp, it names no layer type and no dataset type.

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

namespace
{
    constexpr std::string_view USAGE =
        "sushiai inference - evaluate a trained checkpoint over a dataset.\n"
        "\n"
        "Usage: sushiai inference <model.json> <model.ckpt> <eval.json>\n"
        "\n"
        "  model.json  the architecture the checkpoint was trained under\n"
        "  model.ckpt  the checkpoint `sushiai train` wrote\n"
        "  eval.json   the data to evaluate on; the architecture file carries none\n";

    /** @brief The index of the largest of @p count values. */
    int64_t argmax_row(const float* row, int64_t count)
    {
        int64_t best = 0;
        for (int64_t i = 1; i < count; ++i)
            if (row[i] > row[best]) best = i;
        return best;
    }
}

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

        // --- Trace the forward pass only ---------------------------------
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

        // Two summaries, and the second is the one that means anything.
        //
        // The predicted-class distribution alone cannot distinguish a perfect
        // model from a random one on class-balanced data - both give a uniform
        // spread - so it is reported beside top-1 agreement with the labels
        // the dataset supplies, not instead of it.
        //
        // Reading Batch::labels names no loss and no objective: labels are part
        // of the Dataset seam, and top-1 agreement is a property of the data
        // and the output, not of what the model was trained to minimise. It is
        // reported only when the output width matches the class count, because
        // otherwise the two are not comparable and a number would be fiction.
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
