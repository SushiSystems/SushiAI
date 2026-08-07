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

// `sushiai train <train.json>` - the driver that is written once and never
// edited again as the library grows.
//
// THE PROPERTY THIS FILE EXISTS TO HAVE
//
// It names no layer type, no loss, no dataset and no optimizer. Every one of
// those arrives through a registry lookup keyed by a string out of a JSON
// file, so adding a convolution, a detection loss or a COCO reader is a table
// row and a factory function in the library - and this file does not change.
//
// That is checkable rather than aspirational: grep this file for Linear, GELU,
// CrossEntropy, AdamW, SGD, Rings or Sequential and there is nothing to find.
// If a future edit adds one, the driver has stopped being the thing the
// config-driven design set out to build.
//
// It is deliberately not a flag-driven program. demo_mlp.cpp is that, and its
// flags are a contract the test suite pins; here every knob lives in the
// configuration file, because a knob that exists in two places drifts.

#include <cstdlib>
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
        "sushiai train - train a model described by a JSON configuration.\n"
        "\n"
        "Usage: sushiai train <train.json>\n"
        "\n"
        "  train.json names the architecture file, the objective, the data, the\n"
        "  optimizer, the epoch count and where to write the checkpoint. Paths\n"
        "  inside it resolve against its own directory.\n";
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

        SA_THROW_IF(argc != 2, "usage: sushiai train <train.json>");

        const sa::Config::TrainSpec spec = sa::Config::load_train(argv[1]);
        const sa::Config::ModelSpec model_spec = sa::Config::load_model(spec.model_path);

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        // --- Everything below is selected by a string, not named by a type ---

        const std::unique_ptr<sa::Data::Dataset> dataset =
            sa::Data::make_dataset(engine, spec.data.type, spec.data.attributes, "the data block");

        SA_THROW_IF(dataset->input_dim() != model_spec.input_features,
                    "'{}' declares {} input features, but the '{}' data supplies {}",
                    spec.model_path.string(), model_spec.input_features,
                    dataset->name(), dataset->input_dim());

        sa::NN::LayerStack model = sa::NN::build_layer_stack(model_spec);

        const std::unique_ptr<sa::Train::IObjective> objective =
            sa::Train::make_objective(spec.objective_type, dataset->batch_size(),
                                      dataset->num_classes(), spec.objective_attributes,
                                      "the objective block");

        sa::Train::TrainingGraph built =
            sa::Train::build_graph(model, *objective, dataset->batch_size(),
                                   dataset->input_dim());

        sa::Graph::TensorPool pool(engine);
        std::vector<sa::Optim::ParameterSlot> parameters =
            sa::Train::bind_parameters(built, pool);
        sa::NN::initialize(engine, model, pool);

        const std::unique_ptr<sa::Optim::Optimizer> optimizer =
            sa::Optim::make_optimizer(engine, spec.optimizer_type, parameters,
                                      spec.optimizer_attributes, "the optimizer block");

        sa::Train::TrainerOptions options;
        options.epochs = 1;
        // The objective owns the metric now, so the trainer's own
        // classifier-shaped accuracy path is switched off: a driver that knows
        // no loss type must not be computing a loss-specific number.
        options.track_accuracy = false;

        sa::Train::Trainer trainer(engine, built.graph, pool, built.spec, *optimizer, options);

        std::cout << "sushiai train\n"
                  << "----------------------------------------------------------------\n"
                  << "SushiAI    : " << SushiAI::version_string()
                  << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
                  << SushiBLAS::Core::format("Config     : {}\n", argv[1])
                  << SushiBLAS::Core::format("Model      : {} ({} layers, {} learnable scalars)\n",
                                             spec.model_path.string(), model.size(),
                                             sa::NN::parameter_elements(model))
                  << SushiBLAS::Core::format("Objective  : {}\n", objective->name())
                  << SushiBLAS::Core::format("Data       : {} ({} x {})\n",
                                             dataset->name(), dataset->batch_count(),
                                             dataset->batch_size())
                  << SushiBLAS::Core::format("Optimizer  : {}\n", spec.optimizer_type)
                  << SushiBLAS::Core::format("Graph      : {} IR nodes over {} values\n",
                                             built.graph.node_count(), built.graph.value_count())
                  << SushiBLAS::Core::format("Schedule   : {} epochs\n\n", spec.epochs);

        // The loop drives run_step itself rather than fit(), so the METRIC it
        // prints is whatever the objective offered - std::nullopt prints the
        // loss alone. That is the whole point: a detector or a PINN reports
        // through this same loop without it learning a second vocabulary.
        for (int64_t epoch = 1; epoch <= spec.epochs; ++epoch)
        {
            double last_loss = 0.0;
            std::optional<sa::Train::Metric> last_metric;

            for (std::size_t b = 0; b < dataset->batch_count(); ++b)
            {
                const sa::Data::Batch& batch = dataset->batch(b);
                last_loss = trainer.run_step(batch).loss;
                last_metric = objective->score(batch, pool.tensor(built.spec.logits));
            }

            if (epoch == 1 || epoch % 5 == 0 || epoch == spec.epochs)
            {
                std::cout << SushiBLAS::Core::format("  epoch {:>3}/{:<3} loss {:.6f}",
                                                     epoch, spec.epochs, last_loss);
                if (last_metric)
                    std::cout << SushiBLAS::Core::format(
                        "   {} {}", last_metric->name,
                        last_metric->is_fraction
                            ? SushiBLAS::Core::format("{:6.2f}%", last_metric->value * 100.0)
                            : SushiBLAS::Core::format("{:.6g}", last_metric->value));
                std::cout << "\n";
            }
        }

        // The compile-once contract, printed in the same shape `sa demo mlp`
        // prints it, because a CTest case pins this line.
        std::cout << SushiBLAS::Core::format("\ncompiles: {}\n", trainer.compile_count());
        if (trainer.compile_count() != 1)
        {
            std::cerr << "The step must be compiled exactly once.\n";
            return EXIT_FAILURE;
        }

        if (!spec.checkpoint.empty())
        {
            sa::IO::save_checkpoint(spec.checkpoint, model, pool, engine);
            std::cout << SushiBLAS::Core::format("wrote {} ({} scalars)\n",
                                                 spec.checkpoint.string(),
                                                 sa::NN::parameter_elements(model));
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sushiai train failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
