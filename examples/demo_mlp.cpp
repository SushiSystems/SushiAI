/**************************************************************************/
/* demo_mlp.cpp                                                           */
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

// The program behind `sa demo mlp`: a configurable MLP classifier trained end
// to end, reporting the one number the whole architecture rests on.
//
// WHY THIS IS NOT examples/example.cpp
//
// The two programs look alike and are deliberately not the same binary:
//
//  - example.cpp is a teaching file. It is meant to be read top to bottom and
//    edited by whoever is learning the API. Its output has no contract, and
//    hardcoding its shape is the point - there is nothing to configure and
//    nothing to distract from the six steps of a training run.
//  - this file is a product surface. `sa demo mlp` is a supported command:
//    its flags and its printed lines are a contract, tests/CMakeLists.txt
//    pins the compile-count line so the demo cannot rot unnoticed, and
//    docs/README.md quotes its real output.
//
// Pointing the CLI at the example binary would make every edit to a teaching
// file a change to a shipped command, and would leave the demo with no flags
// at all. One extra 200-line translation unit is a cheap price for keeping
// those two lifecycles apart.
//
// Data: MNIST when the IDX files happen to be present, and the seeded
// synthetic problem otherwise. Both paths come from data/dataset.hpp; this
// file only picks between them and says out loud which one it got.

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/format.hpp>
#include <SushiRuntime/core/span.hpp>

namespace
{
    /** @brief Which update rule the demo trains with. */
    enum class OptimizerKind
    {
        ADAMW, /**< AdamW, the default; converges on this problem without tuning. */
        SGD,   /**< SGD with momentum, for comparison. */
    };

    /**
     * @struct Settings
     * @brief Everything the demo takes from the command line.
     *
     * Defaults reproduce the run quoted in docs/README.md, so `sa demo mlp`
     * with no flags is the run the documentation describes.
     */
    struct Settings
    {
        int64_t epochs = 8;             /**< Passes over the dataset. */
        int64_t batches = 16;           /**< Batches per epoch; steps = epochs * batches. */
        int64_t batch_size = 32;        /**< Samples per batch. */
        int64_t hidden = 128;           /**< Width of the single hidden layer. */
        int64_t features = 32;          /**< Input features (synthetic fallback only). */
        int64_t classes = 6;            /**< Classes (synthetic fallback only). */

        /**
         * @brief The learning rate, or nothing for the optimizer's own default.
         *
         * Optional rather than a 0 sentinel, so `--lr 0` is a rejected value
         * rather than a silent request for the default.
         */
        std::optional<double> learning_rate;

        std::uint64_t seed = 0x0D15EA5E'0BADF00Dull; /**< Seeds both the data and the weights. */
        OptimizerKind optimizer = OptimizerKind::ADAMW;
        std::filesystem::path mnist_directory = "data/mnist";

        /**
         * @brief Whether the fusion pass runs; on by default, as the library is.
         *
         * A flag rather than a rebuild because the two paths are supposed to
         * produce the *same numbers* and differ only in launch count, and the
         * only honest way to show that is to run both from one binary against
         * one SushiBLAS. It is also the measurement anyone reproducing the
         * figures in docs/README.md has to be able to take.
         */
        bool fusion = true;

        /**
         * @brief Catalog entries to leave unmatched, by name.
         *
         * The per-form half of `--fusion`. Fusion's whole payoff is a launch
         * count, and a launch count is only worth something if the wall clock
         * agrees - so the ability to switch one form off and re-measure is not
         * a debugging aid, it is how anyone finds out which entry is actually
         * paying on their model and their device.
         */
        std::vector<sa::Graph::FusedForm> without;
    };

    constexpr std::string_view USAGE = R"(sa demo mlp - train an MLP classifier end to end.

Usage: sushiai_demo_mlp [options]

  --epochs N        passes over the dataset            (default 8)
  --batches N       batches per epoch                  (default 16)
  --batch-size N    samples per batch                  (default 32)
  --hidden N        hidden layer width                 (default 128)
  --features N      input features, synthetic only     (default 32)
  --classes N       classes, synthetic only            (default 6)
  --lr F            learning rate      (default 3e-3 adamw / 0.2 sgd)
  --optimizer NAME  adamw | sgd                        (default adamw)
  --fusion on|off   run the operator fusion pass           (default on)
  --no-fuse NAME    leave one catalog entry unmatched; repeatable. NAME is a
                    form as the report names it: gemm+bias, gemm+bias+relu,
                    gemm+bias+sigmoid, gemm+bias+tanh, exp-sub-scale
  --seed N          seeds the data and the weights  (default: printed below)
  --mnist-dir PATH  where to look for the IDX files    (default data/mnist)
  -h, --help        print this and exit

Total steps are epochs x batches. MNIST is used when the training split's two
IDX files are present under --mnist-dir; otherwise the run falls back to a
seeded synthetic classification problem and says so. --features and --classes
are ignored on the MNIST path, which is fixed at 784 -> 10.
)";

    /**
     * @brief Parses @p text as a whole number, naming @p flag when it is not one.
     * @param flag The flag the value came from, for the error message.
     * @param text The value as typed.
     * @return The parsed value.
     * @throws SushiAI::Error If @p text is not entirely a whole number.
     */
    int64_t parse_int(std::string_view flag, const std::string& text)
    {
        try
        {
            std::size_t consumed = 0;
            const long long parsed = std::stoll(text, &consumed);
            if (consumed == text.size()) return static_cast<int64_t>(parsed);
        }
        catch (const std::logic_error&)
        {
            // Both "not a number" and "out of range" fall through to the one
            // message below: the distinction means nothing to whoever typed it.
        }
        SA_THROW_IF(true, "{} expects a whole number, got '{}'", flag, text);
        return 0;
    }

    /**
     * @brief Parses @p text as an unsigned seed, naming @p flag when it is not one.
     * @param flag The flag the value came from, for the error message.
     * @param text The value as typed.
     * @return The parsed value.
     * @throws SushiAI::Error If @p text is not entirely an unsigned number.
     */
    std::uint64_t parse_seed(std::string_view flag, const std::string& text)
    {
        try
        {
            std::size_t consumed = 0;
            const unsigned long long parsed = std::stoull(text, &consumed);
            if (consumed == text.size()) return static_cast<std::uint64_t>(parsed);
        }
        catch (const std::logic_error&)
        {
            // As above: malformed and out-of-range read the same to the user.
        }
        SA_THROW_IF(true, "{} expects an unsigned number, got '{}'", flag, text);
        return 0;
    }

    /**
     * @brief Parses @p text as a real number, naming @p flag when it is not one.
     * @param flag The flag the value came from, for the error message.
     * @param text The value as typed.
     * @return The parsed value.
     * @throws SushiAI::Error If @p text is not entirely a real number.
     */
    double parse_real(std::string_view flag, const std::string& text)
    {
        try
        {
            std::size_t consumed = 0;
            const double parsed = std::stod(text, &consumed);
            if (consumed == text.size()) return parsed;
        }
        catch (const std::logic_error&)
        {
            // As above: malformed and out-of-range read the same to the user.
        }
        SA_THROW_IF(true, "{} expects a number, got '{}'", flag, text);
        return 0.0;
    }

    /**
     * @brief Rejects a setting that cannot describe a training run.
     *
     * Checked here rather than deep inside the library so the message names
     * the flag the user typed instead of an internal extent.
     *
     * @param settings The parsed settings.
     * @throws SushiAI::Error If any extent is not positive.
     */
    void validate(const Settings& settings)
    {
        SA_THROW_IF(settings.epochs <= 0, "--epochs must be positive, got {}", settings.epochs);
        SA_THROW_IF(settings.batches <= 0, "--batches must be positive, got {}", settings.batches);
        SA_THROW_IF(settings.batch_size <= 0, "--batch-size must be positive, got {}",
                    settings.batch_size);
        SA_THROW_IF(settings.hidden <= 0, "--hidden must be positive, got {}", settings.hidden);
        SA_THROW_IF(settings.features <= 0, "--features must be positive, got {}",
                    settings.features);
        SA_THROW_IF(settings.classes < 2, "--classes must be at least 2, got {}", settings.classes);
        SA_THROW_IF(settings.learning_rate && *settings.learning_rate <= 0.0,
                    "--lr must be positive, got {}", *settings.learning_rate);
    }

    /**
     * @brief Turns the command line into @c Settings.
     *
     * A hand-written loop rather than a parsing library: the whole grammar is
     * "--flag value", every flag takes exactly one value, and a dependency
     * for that would cost more than it saves.
     *
     * @param argc Argument count, as handed to main().
     * @param argv Argument vector, as handed to main().
     * @param wants_help Set when --help was asked for; nothing else is filled in.
     * @return The parsed settings.
     * @throws SushiAI::Error On an unknown flag, a missing value, or a bad value.
     */
    Settings parse_command_line(int argc, char** argv, bool& wants_help)
    {
        Settings settings;
        wants_help = false;

        for (int i = 1; i < argc; ++i)
        {
            const std::string_view flag = argv[i];

            if (flag == "-h" || flag == "--help")
            {
                wants_help = true;
                return settings;
            }

            // Every flag below takes exactly one value, so the "you forgot the
            // value" check is written once here instead of at each branch.
            const auto value = [&]() -> std::string
            {
                SA_THROW_IF(i + 1 >= argc, "{} needs a value", flag);
                return argv[++i];
            };

            if (flag == "--epochs")           settings.epochs = parse_int(flag, value());
            else if (flag == "--batches")     settings.batches = parse_int(flag, value());
            else if (flag == "--batch-size")  settings.batch_size = parse_int(flag, value());
            else if (flag == "--hidden")      settings.hidden = parse_int(flag, value());
            else if (flag == "--features")    settings.features = parse_int(flag, value());
            else if (flag == "--classes")     settings.classes = parse_int(flag, value());
            else if (flag == "--lr")          settings.learning_rate = parse_real(flag, value());
            else if (flag == "--seed")        settings.seed = parse_seed(flag, value());
            else if (flag == "--mnist-dir")   settings.mnist_directory = value();
            else if (flag == "--no-fuse")
            {
                const std::string name = value();
                const sa::Graph::FusedForm form = sa::Graph::fused_form_from_name(name);
                SA_THROW_IF(form == sa::Graph::FusedForm::NONE,
                            "--no-fuse does not know the form '{}'. Run with --help for the "
                            "names; they are the ones the fusion report prints.", name);
                settings.without.push_back(form);
            }
            else if (flag == "--fusion")
            {
                const std::string state = value();
                if (state == "on") settings.fusion = true;
                else if (state == "off") settings.fusion = false;
                else SA_THROW_IF(true, "--fusion takes 'on' or 'off', got '{}'", state);
            }
            else if (flag == "--optimizer")
            {
                const std::string name = value();
                if (name == "adamw") settings.optimizer = OptimizerKind::ADAMW;
                else if (name == "sgd") settings.optimizer = OptimizerKind::SGD;
                else SA_THROW_IF(true, "--optimizer takes 'adamw' or 'sgd', got '{}'", name);
            }
            else SA_THROW_IF(true, "Unknown flag '{}'. Run with --help.", flag);
        }

        validate(settings);
        return settings;
    }

    /** @brief The momentum SGD runs with; stated in the report rather than hidden. */
    constexpr double SGD_MOMENTUM = 0.9;

    /**
     * @brief Builds the requested optimizer over @p parameters.
     *
     * The learning-rate default is per-optimizer, because 3e-3 is a sensible
     * AdamW step and a uselessly small SGD one - a shared default would make
     * `--optimizer sgd` look broken rather than merely different.
     *
     * @param engine The engine the update is recorded on.
     * @param parameters The slots to update.
     * @param settings The parsed settings.
     * @param learning_rate Receives the rate actually used, for the report.
     * @return The optimizer, owned by the caller.
     */
    std::unique_ptr<sa::Optim::Optimizer> make_optimizer(
        sb::Engine& engine,
        std::vector<sa::Optim::ParameterSlot> parameters,
        const Settings& settings,
        double& learning_rate)
    {
        if (settings.optimizer == OptimizerKind::SGD)
        {
            sa::Optim::SGDOptions sgd;
            sgd.learning_rate = settings.learning_rate.value_or(0.2);
            sgd.momentum = SGD_MOMENTUM;
            learning_rate = sgd.learning_rate;
            return std::make_unique<sa::Optim::SGD>(engine, std::move(parameters), sgd);
        }

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = settings.learning_rate.value_or(3e-3);
        learning_rate = adam.learning_rate;
        return std::make_unique<sa::Optim::AdamW>(engine, std::move(parameters), adam);
    }

    /**
     * @struct DataChoice
     * @brief Which dataset the run got, and how to say so out loud.
     *
     * The provenance travels with the dataset rather than being recomputed by
     * whoever prints: "which data did this run actually see" is the first
     * question anyone asks of a training number, and it must not be possible
     * to answer it from a different code path than the one that chose.
     */
    struct DataChoice
    {
        std::unique_ptr<sa::Data::Dataset> dataset; /**< Never null. */
        std::string provenance;                     /**< A human-readable note. */
        bool is_mnist = false;                      /**< False means the synthetic fallback. */
    };

    /**
     * @brief Loads MNIST when its files are there, else the synthetic problem.
     *
     * The fallback itself lives in data/dataset.hpp - absence of the IDX files
     * is not an error there, and this only chooses between the two and records
     * which it got.
     *
     * @param engine The engine that allocates and fills the batches.
     * @param settings The parsed settings.
     * @return The dataset and its provenance.
     */
    DataChoice load_dataset(sb::Engine& engine, const Settings& settings)
    {
        sa::Data::MnistOptions mnist;
        mnist.directory = settings.mnist_directory;
        mnist.batch_size = settings.batch_size;
        mnist.max_samples = settings.batch_size * settings.batches;

        if (std::unique_ptr<sa::Data::Dataset> loaded = sa::Data::try_load_mnist(engine, mnist))
        {
            return DataChoice{
                std::move(loaded),
                SushiBLAS::Core::format("MNIST (real IDX files) from {}", mnist.directory.string()),
                true,
            };
        }

        sa::Data::SyntheticOptions synthetic;
        synthetic.batch_size = settings.batch_size;
        synthetic.batches = settings.batches;
        synthetic.features = settings.features;
        synthetic.classes = settings.classes;
        synthetic.seed = settings.seed;

        return DataChoice{
            std::make_unique<sa::Data::SyntheticClassification>(engine, synthetic),
            SushiBLAS::Core::format("SYNTHETIC FALLBACK - no MNIST here.\n"
                                    "           Looked for {}.",
                                    sa::Data::mnist_expected_paths(mnist)),
            false,
        };
    }

    /**
     * @brief A byte count a reader can take in at a glance.
     *
     * Integer KiB alone reads as "0 KiB instead of 0 KiB" on the two-step run
     * the CTest smoke test uses, which makes the planner look broken when it is
     * merely working on a tiny model.
     *
     * @param bytes The count to render.
     * @return "N B" below a kibibyte, "N.N KiB" above it.
     */
    std::string human_bytes(std::size_t bytes)
    {
        if (bytes < 1024) return SushiBLAS::Core::format("{} B", bytes);
        return SushiBLAS::Core::format("{:.1f} KiB", static_cast<double>(bytes) / 1024.0);
    }

    /**
     * @brief The fusion pass's lines of the configuration report.
     *
     * Shown rather than asserted, and shown *including the refusals*, because
     * every way this pass can be wrong produces a run that still converges. A
     * reader who can see "gemm+bias+relu x1, gemm+bias x1" next to "declined
     * sushiai.gelu: its gradient differentiates at its input" can tell at a
     * glance whether the model in front of them got what it should have.
     *
     * @param report What the pass did.
     * @return One line plus one per decision; always newline-terminated.
     */
    std::string describe_fusion(const sa::Graph::FusionReport& report)
    {
        if (!report.enabled())
            return "Fusion   : off; the graph is lowered exactly as it was built\n";

        if (report.fused().empty() && report.declined().empty())
            return "Fusion   : nothing in this graph matches the catalog\n";

        std::string text = SushiBLAS::Core::format("Fusion   : {} node(s) -> {}\n",
                                                   report.nodes_before(), report.nodes_after());

        for (const sa::Graph::FusedSubtree& subtree : report.fused())
            text += SushiBLAS::Core::format("           fused    {} -> value {}\n",
                                            sa::Graph::fused_form_name(subtree.form), subtree.result);

        for (const sa::Graph::DeclinedFusion& decline : report.declined())
            text += SushiBLAS::Core::format("           declined {}: {}\n",
                                            decline.form != sa::Graph::FusedForm::NONE
                                                ? sa::Graph::fused_form_name(decline.form)
                                                : sa::Ops::name_of(decline.op),
                                            sa::Graph::fusion_decline_reason(decline.reason));

        return text;
    }

    /**
     * @brief The memory planner's line of the configuration report.
     *
     * The saving is shown rather than asserted: the "Buffers" line above prints
     * what was actually allocated next to what one allocation per tensor would
     * have cost, and this says which arenas account for the difference. A run
     * in which the planner declined to overlap live ranges - a graph too large
     * for the reachability closure, say - prints that reason instead of a
     * silent 0%.
     *
     * @param pool The pool whose plan to describe.
     * @return One line, or two when reuse was declined; always newline-terminated.
     */
    std::string describe_arenas(const sa::Graph::TensorPool& pool)
    {
        if (!pool.has_plan())
            return "Arenas   : none; no memory plan was adopted\n";

        const sa::Graph::MemoryPlan& plan = pool.arenas().plan();
        const SushiRuntime::span<const sa::Graph::Arena> arenas = plan.arenas();

        if (arenas.empty())
            return "Arenas   : none; this graph has no packable values\n";

        std::string classes;
        std::size_t buffers = 0;
        for (const sa::Graph::Arena& arena : arenas)
        {
            if (!classes.empty()) classes += " + ";
            classes += sa::Graph::lifetime_name(arena.lifetime);
            buffers += arena.buffers;
        }

        const std::size_t packed = plan.arena_bytes();
        const std::size_t loose = plan.unpacked_bytes();
        const double saved = loose == 0
            ? 0.0
            : 100.0 * (1.0 - static_cast<double>(packed) / static_cast<double>(loose));

        std::string text = SushiBLAS::Core::format(
            "Arenas   : {} ({}), {} buffers in {} instead of {} - {:.0f}% saved\n",
            arenas.size(), classes, buffers, human_bytes(packed), human_bytes(loose), saved);

        if (!plan.reuse_enabled())
            text += SushiBLAS::Core::format("           live ranges were not overlapped: {}\n",
                                            plan.reuse_note());

        return text;
    }

} // namespace

int main(int argc, char** argv)
{
    try
    {
        bool wants_help = false;
        const Settings settings = parse_command_line(argc, argv, wants_help);
        if (wants_help)
        {
            std::cout << USAGE;
            return EXIT_SUCCESS;
        }

        sr::Execution::RuntimeContext context;
        sb::Engine engine(context);

        // --- Data ---------------------------------------------------------
        const DataChoice data = load_dataset(engine, settings);
        const sa::Data::Dataset& dataset = *data.dataset;

        const int64_t features = dataset.input_dim();
        const int64_t classes = dataset.num_classes();

        // --- Model --------------------------------------------------------
        sa::NN::Sequential model{
            sa::NN::Linear{features, settings.hidden},
            sa::NN::ReLU{},
            sa::NN::Linear{settings.hidden, classes},
        };

        // Trace, add cross-entropy, differentiate, fuse. All IR-to-IR: no
        // device work has happened yet, and nothing has been compiled.
        sa::Graph::FusionOptions fusion;
        fusion.enabled = settings.fusion;
        fusion.disabled = SushiRuntime::span<const sa::Graph::FusedForm>(settings.without.data(),
                                                                         settings.without.size());

        sa::Train::TrainingGraph built = sa::Train::build_classifier(
            model, settings.batch_size, features, classes, sa::DType::FLOAT32, fusion);

        // --- Memory and initialisation --------------------------------------
        sa::Graph::TensorPool pool(engine);
        std::vector<sa::Optim::ParameterSlot> parameters = sa::Train::bind_parameters(built, pool);

        sa::NN::InitPolicy init;
        init.seed = settings.seed;
        sa::NN::initialize(engine, model, pool, init);

        double learning_rate = 0.0;
        const std::unique_ptr<sa::Optim::Optimizer> optimizer =
            make_optimizer(engine, std::move(parameters), settings, learning_rate);

        // --- The one compile ------------------------------------------------
        // Trainer's constructor lowers and compiles the step. Epochs are driven
        // from this file, one fit() call each, so the per-epoch trajectory is a
        // real measurement rather than a summary - and so the compile count
        // below is counted across every one of those calls.
        sa::Train::TrainerOptions options;
        options.epochs = 1;
        sa::Train::Trainer trainer(engine, built.graph, pool, built.spec, *optimizer, options);

        // From the dataset, not from --batches: the MNIST path caps itself at
        // whatever the files actually hold, so the two can differ and the
        // reported schedule has to be the one that will run.
        const int64_t total_steps =
            settings.epochs * static_cast<int64_t>(dataset.batch_count());

        std::cout
            << "SushiAI MLP demo - configuration\n"
            << "----------------------------------------------------------------\n"
            << "SushiAI  : " << SushiAI::version_string()
            << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
            << "Data     : " << data.provenance << "\n"
            << SushiBLAS::Core::format("Model    : {} -> {} -> {}  ({} learnable scalars)\n",
                                       features, settings.hidden, classes,
                                       sa::NN::parameter_elements(model))
            << SushiBLAS::Core::format("Optimizer: {} (lr {:g}{})\n",
                                       settings.optimizer == OptimizerKind::SGD ? "sgd" : "adamw",
                                       learning_rate,
                                       settings.optimizer == OptimizerKind::SGD
                                           ? SushiBLAS::Core::format(", momentum {:g}", SGD_MOMENTUM)
                                           : std::string{})
            << SushiBLAS::Core::format("Batches  : {} x {} from '{}'\n",
                                       dataset.batch_count(), settings.batch_size, dataset.name())
            << SushiBLAS::Core::format("Graph    : {} IR nodes over {} values\n",
                                       built.graph.node_count(), built.graph.value_count())
            << describe_fusion(built.fusion)
            << SushiBLAS::Core::format("Buffers  : {} tensors, {} (one allocation each: {})\n",
                                       pool.tensor_count(), human_bytes(pool.allocated_bytes()),
                                       human_bytes(pool.unpacked_bytes()))
            << describe_arenas(pool)
            << SushiBLAS::Core::format("Schedule : {} epochs x {} batches = {} steps, seed {}\n\n",
                                       settings.epochs, dataset.batch_count(), total_steps, settings.seed);

        // --- Train ------------------------------------------------------------
        // The trajectory is collected rather than printed as it happens.
        // SushiRuntime logs at INFO to stdout in a Release build (its level is
        // a compile-time constant, so a consumer cannot turn it down), and a
        // report emitted line by line comes out shredded through it. Held back
        // and printed as one block, the result survives intact.
        std::string trajectory;
        sa::Train::RunReport run;

        for (int64_t epoch = 1; epoch <= settings.epochs; ++epoch)
        {
            const sa::Train::RunReport epoch_report = trainer.fit(dataset);

            if (epoch == 1) run.first_loss = epoch_report.first_loss;
            run.last_loss = epoch_report.last_loss;
            run.last_accuracy = epoch_report.last_accuracy;
            run.operations_per_step = epoch_report.operations_per_step;
            run.steps += epoch_report.steps;
            run.duration_ms += epoch_report.duration_ms;

            trajectory += SushiBLAS::Core::format(
                "  epoch {:>3}/{:<3} loss {:.6f} -> {:.6f}   accuracy {:6.2f}%\n",
                epoch, settings.epochs, epoch_report.first_loss,
                epoch_report.last_loss, epoch_report.last_accuracy * 100.0);
        }
        run.compile_count = trainer.compile_count();

        // The last thing printed, in one write, ending on the load-bearing
        // claim of the whole design: every step above replayed one compiled
        // plan, and the number that says so is the last thing on screen.
        std::cout
            << "\n================================================================\n"
            << " SushiAI MLP demo - result\n"
            << " " << (data.is_mnist ? "MNIST" : "synthetic fallback")
            << ", " << (settings.optimizer == OptimizerKind::SGD ? "sgd" : "adamw")
            << SushiBLAS::Core::format(", {} -> {} -> {}\n", features, settings.hidden, classes)
            << "================================================================\n"
            << trajectory
            << "\n" << run.to_string()
            << "----------------------------------------------------------------\n"
            << SushiBLAS::Core::format("  compiles: {}  <- one lowering, one compile, {} replays.\n",
                                       run.compile_count, run.steps)
            << "================================================================\n"
            << std::flush;

        if (run.compile_count != 1)
        {
            std::cerr << "\nThe step was compiled " << run.compile_count
                      << " times; it must be compiled exactly once.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sa demo mlp failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
