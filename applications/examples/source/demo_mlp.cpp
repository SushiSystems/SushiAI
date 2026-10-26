/**************************************************************************/
/* demo_mlp.cpp                                                           */
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
 * @file demo_mlp.cpp
 * @brief Runs `sa demo mlp`: trains a configurable MLP classifier and reports its compile count.
 * @author Mustafa Garip
 * @see applications/examples/README.md, "demo_mlp.cpp".
 */

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
#include <utility>
#include <vector>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiLog/text/value_text.hpp>
#include <SushiRuntime/core/span.hpp>

/** @brief Shortens the SushiBLAS namespace. */
namespace sb = SushiBLAS;

namespace
{
    /** @brief Selects the update rule the demo trains with. */
    enum class OptimizerKind : std::uint8_t
    {
        AdamW, /**< AdamW, the default; converges on this problem without tuning. */
        SGD,   /**< SGD with momentum, for comparison. */
    };

    /**
     * @brief Selects the activation of the demo's one hidden layer.
     * @see applications/examples/README.md, "The two activations".
     */
    enum class ActivationKind : std::uint8_t
    {
        ReLU, /**< The default; folds with nothing spilled. */
        GELU, /**< Folds only via the auxiliary pre-activation spill. */
    };

    /**
     * @brief Holds everything the demo takes from the command line.
     * @see applications/examples/README.md, "Settings and their defaults".
     */
    struct Settings
    {
        int64_t epochs = 8;             /**< Passes over the dataset. */
        int64_t batches = 16;           /**< Batches per epoch; steps = epochs * batches. */
        int64_t batch_size = 32;        /**< Samples per batch. */
        int64_t hidden = 128;           /**< Width of the single hidden layer. */
        int64_t features = 32;          /**< Input features (synthetic fallback only). */
        int64_t classes = 6;            /**< Classes (synthetic fallback only). */

        /** @brief Holds the learning rate, or nothing for the optimizer's own default. */
        std::optional<double> learning_rate;

        std::uint64_t seed = 0x0D15EA5E'0BADF00Dull; /**< Seeds both the data and the weights. */
        /** @brief Holds the update rule the run trains with. */
        OptimizerKind optimizer = OptimizerKind::AdamW;
        /** @brief Holds the activation of the hidden layer. */
        ActivationKind activation = ActivationKind::ReLU;
        /** @brief Holds the directory the MNIST IDX files are looked for in. */
        std::filesystem::path mnist_directory = "data/mnist";

        /** @brief Holds whether the fusion pass runs; on by default, as in the library. */
        bool fusion = true;

        /** @brief Lists the catalog entries the fusion pass leaves unmatched. */
        std::vector<sa::Graph::FusedForm> without;

        /** @brief Holds whether the compute-bound operations run in fp16; off by default. */
        bool mixed_precision = false;

        /** @brief Holds the gradient loss scale, or 0 for no scaling. */
        double loss_scale = 0.0;
    };

    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE = R"(sa demo mlp - train an MLP classifier end to end.

Usage: sushiai_demo_mlp [options]

  --epochs N        passes over the dataset            (default 8)
  --batches N       batches per epoch                  (default 16)
  --batch-size N    samples per batch                  (default 32)
  --hidden N        hidden layer width                 (default 128)
  --features N      input features, synthetic only     (default 32)
  --classes N       classes, synthetic only            (default 6)
  --lr F            learning rate      (default 3e-3 adamw / 0.2 sgd)
  --optimizer NAME  adamw | sgd                        (default adamw)
  --activation A    relu | gelu                          (default relu)
                    The two kinds of fusable activation. ReLU's gradient reads
                    its own output, so it folds into the GEMM epilogue with
                    nothing else written; GELU's reads its input, so it folds
                    only because that input is spilled to a second buffer on
                    the way past (cuBLASLt calls it GELU_AUX_BIAS). The fusion
                    report names whichever form was matched.
  --fusion on|off   run the operator fusion pass           (default on)
  --precision P     fp32 | mixed                          (default fp32)
                    'mixed' runs the matrix products and the epilogues a fused
                    GEMM folds into them in fp16, keeping fp32 master weights,
                    fp32 gradients and every reduction in fp32. There is no
                    fp16 matrix unit on this machine, so this is a correctness
                    switch and not a speed one.
  --loss-scale S    gradient loss scaling, starting at S  (default off)
                    S must be a power of two, so the unscale before the update
                    is exact. Overflow halves it and throws the step away;
                    2000 clean steps double it. This workload does not overflow
                    at a sane scale and converges the same without it, so the
                    honest use is `--precision mixed --loss-scale 16777216`,
                    which overflows fp16 deliberately and shows the backoff.
  --no-fuse NAME    leave one catalog entry unmatched; repeatable. NAME is a
                    form as the report names it: gemm+bias, gemm+bias+relu,
                    gemm+bias+sigmoid, gemm+bias+tanh, gemm+bias+gelu,
                    exp-sub-scale
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
     * @return The parsed value, or InvalidDescription when @p text is not a whole number.
     */
    [[nodiscard]] sa::Result<int64_t> parse_int(std::string_view flag, const std::string& text)
    {
        long long parsed = 0;
        bool is_whole = false;

        try
        {
            std::size_t consumed = 0;
            parsed = std::stoll(text, &consumed);
            is_whole = consumed == text.size();
        }
        catch (const std::logic_error&)
        {
            // Malformed and out-of-range input share the one message below.
        }

        SA_FAIL_IF(!is_whole, sa::StatusCode::InvalidDescription,
                   "{} expects a whole number, got '{}'", flag, text);
        return static_cast<int64_t>(parsed);
    }

    /**
     * @brief Parses @p text as an unsigned seed, naming @p flag when it is not one.
     * @param flag The flag the value came from, for the error message.
     * @param text The value as typed.
     * @return The parsed value, or InvalidDescription when @p text is not an unsigned number.
     */
    [[nodiscard]] sa::Result<std::uint64_t> parse_seed(std::string_view flag,
                                                       const std::string& text)
    {
        unsigned long long parsed = 0;
        bool is_whole = false;

        try
        {
            std::size_t consumed = 0;
            parsed = std::stoull(text, &consumed);
            is_whole = consumed == text.size();
        }
        catch (const std::logic_error&)
        {
            // As above: malformed and out-of-range read the same to the user.
        }

        SA_FAIL_IF(!is_whole, sa::StatusCode::InvalidDescription,
                   "{} expects an unsigned number, got '{}'", flag, text);
        return static_cast<std::uint64_t>(parsed);
    }

    /**
     * @brief Parses @p text as a real number, naming @p flag when it is not one.
     * @param flag The flag the value came from, for the error message.
     * @param text The value as typed.
     * @return The parsed value, or InvalidDescription when @p text is not a real number.
     */
    [[nodiscard]] sa::Result<double> parse_real(std::string_view flag, const std::string& text)
    {
        double parsed = 0.0;
        bool is_whole = false;

        try
        {
            std::size_t consumed = 0;
            parsed = std::stod(text, &consumed);
            is_whole = consumed == text.size();
        }
        catch (const std::logic_error&)
        {
            // As above: malformed and out-of-range read the same to the user.
        }

        SA_FAIL_IF(!is_whole, sa::StatusCode::InvalidDescription,
                   "{} expects a number, got '{}'", flag, text);
        return parsed;
    }

    /**
     * @brief Rejects a setting that cannot describe a training run.
     * @return Success, or InvalidDescription when an extent or the rate is out of range.
     * @see applications/examples/README.md, "Command line parsing".
     */
    [[nodiscard]] sa::Status validate(const Settings& settings)
    {
        SA_FAIL_IF(settings.epochs <= 0, sa::StatusCode::InvalidDescription,
                   "--epochs must be positive, got {}", settings.epochs);
        SA_FAIL_IF(settings.batches <= 0, sa::StatusCode::InvalidDescription,
                   "--batches must be positive, got {}", settings.batches);
        SA_FAIL_IF(settings.batch_size <= 0, sa::StatusCode::InvalidDescription,
                   "--batch-size must be positive, got {}", settings.batch_size);
        SA_FAIL_IF(settings.hidden <= 0, sa::StatusCode::InvalidDescription,
                   "--hidden must be positive, got {}", settings.hidden);
        SA_FAIL_IF(settings.features <= 0, sa::StatusCode::InvalidDescription,
                   "--features must be positive, got {}", settings.features);
        SA_FAIL_IF(settings.classes < 2, sa::StatusCode::InvalidDescription,
                   "--classes must be at least 2, got {}", settings.classes);
        SA_FAIL_IF(settings.learning_rate && *settings.learning_rate <= 0.0,
                   sa::StatusCode::InvalidDescription, "--lr must be positive, got {}",
                   *settings.learning_rate);
        return sa::Status();
    }

    /**
     * @brief Turns the command line into @c Settings.
     * @param wants_help Set when --help was asked for; nothing else is filled in.
     * @return The settings, or InvalidDescription on an unknown flag, a missing value or a
     *         bad value.
     * @see applications/examples/README.md, "Command line parsing".
     */
    [[nodiscard]] sa::Result<Settings> parse_command_line(int argc, char** argv, bool& wants_help)
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

            // Every flag below takes exactly one value; this lambda checks that it is there.
            const auto value = [&]() -> sa::Result<std::string>
            {
                SA_FAIL_IF(i + 1 >= argc, sa::StatusCode::InvalidDescription,
                           "{} needs a value", flag);
                return std::string(argv[++i]);
            };

            if (flag == "--epochs")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.epochs, parse_int(flag, text));
            }
            else if (flag == "--batches")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.batches, parse_int(flag, text));
            }
            else if (flag == "--batch-size")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.batch_size, parse_int(flag, text));
            }
            else if (flag == "--hidden")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.hidden, parse_int(flag, text));
            }
            else if (flag == "--features")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.features, parse_int(flag, text));
            }
            else if (flag == "--classes")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.classes, parse_int(flag, text));
            }
            else if (flag == "--lr")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.learning_rate, parse_real(flag, text));
            }
            else if (flag == "--seed")
            {
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.seed, parse_seed(flag, text));
            }
            else if (flag == "--mnist-dir")
            {
                SA_TRY_ASSIGN(settings.mnist_directory, value());
            }
            else if (flag == "--no-fuse")
            {
                SA_TRY_ASSIGN(const std::string name, value());
                const sa::Graph::FusedForm form = sa::Graph::fused_form_from_name(name);
                SA_FAIL_IF(form == sa::Graph::FusedForm::None, sa::StatusCode::InvalidDescription,
                           "--no-fuse does not know the form '{}'. Run with --help for the "
                           "names; they are the ones the fusion report prints.", name);
                settings.without.push_back(form);
            }
            else if (flag == "--fusion")
            {
                SA_TRY_ASSIGN(const std::string state, value());
                if (state == "on") settings.fusion = true;
                else if (state == "off") settings.fusion = false;
                else SA_FAIL_IF(true, sa::StatusCode::InvalidDescription,
                                "--fusion takes 'on' or 'off', got '{}'", state);
            }
            else if (flag == "--loss-scale")
            {
                // Only positivity is checked here; LossScaleState owns the power-of-two rule.
                SA_TRY_ASSIGN(const std::string text, value());
                SA_TRY_ASSIGN(settings.loss_scale, parse_real(flag, text));
                SA_FAIL_IF(settings.loss_scale <= 0.0, sa::StatusCode::InvalidDescription,
                           "--loss-scale takes a positive power of two, or omit it for none; "
                           "got {}", settings.loss_scale);
            }
            else if (flag == "--precision")
            {
                SA_TRY_ASSIGN(const std::string name, value());
                if (name == "fp32") settings.mixed_precision = false;
                else if (name == "mixed") settings.mixed_precision = true;
                else SA_FAIL_IF(true, sa::StatusCode::InvalidDescription,
                                "--precision takes 'fp32' or 'mixed', got '{}'", name);
            }
            else if (flag == "--optimizer")
            {
                SA_TRY_ASSIGN(const std::string name, value());
                if (name == "adamw") settings.optimizer = OptimizerKind::AdamW;
                else if (name == "sgd") settings.optimizer = OptimizerKind::SGD;
                else SA_FAIL_IF(true, sa::StatusCode::InvalidDescription,
                                "--optimizer takes 'adamw' or 'sgd', got '{}'", name);
            }
            else if (flag == "--activation")
            {
                SA_TRY_ASSIGN(const std::string name, value());
                if (name == "relu") settings.activation = ActivationKind::ReLU;
                else if (name == "gelu") settings.activation = ActivationKind::GELU;
                else SA_FAIL_IF(true, sa::StatusCode::InvalidDescription,
                                "--activation takes 'relu' or 'gelu', got '{}'", name);
            }
            else SA_FAIL_IF(true, sa::StatusCode::InvalidDescription,
                            "Unknown flag '{}'. Run with --help.", flag);
        }

        SA_TRY(validate(settings));
        return settings;
    }

    /** @brief Holds the momentum SGD runs with, which the report prints. */
    constexpr double K_SGD_MOMENTUM = 0.9;

    /**
     * @brief Builds the requested optimizer over @p parameters.
     * @param learning_rate Receives the rate actually used, for the report.
     * @return The optimizer, owned by the caller, or the failure of its factory.
     * @see applications/examples/README.md, "Learning-rate defaults".
     */
    [[nodiscard]] sa::Result<std::unique_ptr<sa::Optim::Optimizer>> make_optimizer(
        sb::Engine& engine,
        std::vector<sa::Optim::ParameterSlot> parameters,
        const Settings& settings,
        double& learning_rate)
    {
        if (settings.optimizer == OptimizerKind::SGD)
        {
            sa::Optim::SGDOptions sgd;
            sgd.learning_rate = settings.learning_rate.value_or(0.2);
            sgd.momentum = K_SGD_MOMENTUM;
            learning_rate = sgd.learning_rate;

            SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::SGD> optimizer,
                          sa::Optim::SGD::make(engine, std::move(parameters), sgd));
            return std::unique_ptr<sa::Optim::Optimizer>(std::move(optimizer));
        }

        sa::Optim::AdamWOptions adam;
        adam.learning_rate = settings.learning_rate.value_or(3e-3);
        learning_rate = adam.learning_rate;

        SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::AdamW> optimizer,
                      sa::Optim::AdamW::make(engine, std::move(parameters), adam));
        return std::unique_ptr<sa::Optim::Optimizer>(std::move(optimizer));
    }

    /**
     * @brief Holds the dataset the run got and the note that names it.
     * @see applications/examples/README.md, "Data source".
     */
    struct DataChoice
    {
        std::unique_ptr<sa::Data::Dataset> dataset; /**< Never null. */
        std::string provenance;                     /**< A human-readable note. */
        bool is_mnist = false;                      /**< False means the synthetic fallback. */
    };

    /**
     * @brief Loads MNIST when its files are present, else the synthetic problem.
     * @return The dataset and its provenance, or the failure of the load or of the generation.
     * @see applications/examples/README.md, "Data source".
     */
    [[nodiscard]] sa::Result<DataChoice> load_dataset(sb::Engine& engine, const Settings& settings)
    {
        sa::Vision::Datasets::MNISTOptions mnist;
        mnist.directory = settings.mnist_directory;
        mnist.batch_size = settings.batch_size;
        mnist.max_samples = settings.batch_size * settings.batches;

        sa::Result<std::unique_ptr<sa::Data::Dataset>> loaded =
            sa::Vision::Datasets::try_load_mnist(engine, mnist);

        if (loaded.has_value())
        {
            return DataChoice{
                std::move(loaded).value(),
                SushiLog::format_text("MNIST (real IDX files) from {}", mnist.directory.string()),
                true,
            };
        }

        // Only an absent file selects the fallback; any other failure of the load is returned.
        if (loaded.status().code() != sa::StatusCode::FileNotFound)
            return loaded.status();

        sa::Data::SyntheticOptions synthetic;
        synthetic.batch_size = settings.batch_size;
        synthetic.batches = settings.batches;
        synthetic.features = settings.features;
        synthetic.classes = settings.classes;
        synthetic.seed = settings.seed;

        SA_TRY_ASSIGN(std::unique_ptr<sa::Data::SyntheticClassification> generated,
                      sa::Data::SyntheticClassification::make(engine, synthetic));

        return DataChoice{
            std::move(generated),
            SushiLog::format_text("SYNTHETIC FALLBACK - no MNIST here.\n"
                                    "           Looked for {}.",
                                    sa::Vision::Datasets::mnist_expected_paths(mnist)),
            false,
        };
    }

    /**
     * @brief Renders a byte count as "N B" below a kibibyte and "N.N KiB" above it.
     * @see applications/examples/README.md, "The configuration report".
     */
    std::string human_bytes(std::size_t bytes)
    {
        if (bytes < 1024) return SushiLog::format_text("{} B", bytes);
        return SushiLog::format_text("{} KiB", SushiLog::fixed(static_cast<double>(bytes) / 1024.0,
                                                               1));
    }

    /**
     * @brief Renders the fusion pass's lines of the configuration report.
     * @return One line plus one per decision; always newline-terminated.
     * @see applications/examples/README.md, "The configuration report".
     */
    std::string describe_fusion(const sa::Graph::FusionReport& report)
    {
        if (!report.enabled())
            return "Fusion   : off; the graph is lowered exactly as it was built\n";

        if (report.fused().empty() && report.declined().empty())
            return "Fusion   : nothing in this graph matches the catalog\n";

        std::string text = SushiLog::format_text("Fusion   : {} node(s) -> {}\n",
                                                   report.nodes_before(), report.nodes_after());

        for (const sa::Graph::FusedSubtree& subtree : report.fused())
        {
            text += SushiLog::format_text("           fused    {} -> value {}",
                                            sa::Graph::fused_form_name(subtree.form),
                                            subtree.result);

            // The one form that writes two values names the spilled value here.
            if (subtree.spilled != sa::Graph::K_NO_VALUE)
                text += SushiLog::format_text(" (+ pre-activation spilled to value {})",
                                                subtree.spilled);
            text += "\n";
        }

        for (const sa::Graph::DeclinedFusion& decline : report.declined())
            text += SushiLog::format_text("           declined {}: {}\n",
                                            decline.form != sa::Graph::FusedForm::None
                                                ? sa::Graph::fused_form_name(decline.form)
                                                : sa::Ops::name_of(decline.op),
                                            sa::Graph::fusion_decline_reason(decline.reason));

        return text;
    }

    /**
     * @brief Renders the cast pass's line of the configuration report.
     * @return One newline-terminated line.
     * @see applications/examples/README.md, "The configuration report".
     */
    std::string describe_precision(const sa::Graph::CastReport& report)
    {
        if (report.count() == 0)
            return SushiLog::format_text(
                "Precision: {}; every operation computes in the graph's own type\n",
                report.policy());

        return SushiLog::format_text(
            "Precision: {}; {} cast(s) - {} narrowing, {} widening back to the master type\n",
            report.policy(), report.count(),
            report.count_into(sa::DType::Half),
            report.count_into(sa::DType::Float32));
    }

    /**
     * @brief Renders the memory planner's line of the configuration report.
     * @return One line, or two when reuse was declined; always newline-terminated.
     * @see applications/examples/README.md, "The configuration report".
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

        std::string text = SushiLog::format_text(
            "Arenas   : {} ({}), {} buffers in {} instead of {} - {}% saved\n",
            arenas.size(), classes, buffers, human_bytes(packed), human_bytes(loose),
            SushiLog::fixed(saved, 0));

        if (!plan.reuse_enabled())
            text += SushiLog::format_text("           live ranges were not overlapped: {}\n",
                                            plan.reuse_note());

        return text;
    }

    /**
     * @brief Trains @p model end to end and prints the demo's whole contract.
     * @tparam M The model type; must satisfy NN::Module.
     * @param model The model to train; its parameters are declared by the trace.
     * @return Success, or the first failure of a call, or InvalidDescription when the step
     *         was not compiled exactly once.
     * @see applications/examples/README.md, "One template, two models".
     */
    template<typename M>
    [[nodiscard]] sa::Status train_and_report(sb::Engine& engine, M& model, const DataChoice& data,
                                              const sa::Data::Dataset& dataset,
                                              const Settings& settings, int64_t features,
                                              int64_t classes)
    {
        // Tracing, differentiation, casts and fusion are IR-to-IR; nothing is compiled yet.
        sa::Graph::FusionOptions fusion;
        fusion.enabled = settings.fusion;
        fusion.disabled = SushiRuntime::span<const sa::Graph::FusedForm>(settings.without.data(),
                                                                         settings.without.size());

        // Both policies outlive build_classifier and Trainer::make below.
        const sa::Graph::UniformPrecisionPolicy single_precision;
        const sa::Graph::MixedPrecisionPolicy mixed_precision;

        const sa::Graph::PrecisionPolicy* precision =
            settings.mixed_precision
                ? static_cast<const sa::Graph::PrecisionPolicy*>(&mixed_precision)
                : static_cast<const sa::Graph::PrecisionPolicy*>(&single_precision);

        const bool scaling = settings.loss_scale > 0.0;

        SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                      sa::Train::build_classifier(model, settings.batch_size, features, classes,
                                                  sa::DType::Float32, fusion, precision, scaling));

        SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
        sa::Graph::TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                      sa::Train::bind_parameters(built, state, pool));

        sa::NN::InitPolicy init;
        init.seed = settings.seed;
        SA_TRY(sa::NN::initialize(engine, state, init));

        const std::size_t parameter_count = parameters.size();

        double learning_rate = 0.0;
        SA_TRY_ASSIGN(const std::unique_ptr<sa::Optim::Optimizer> optimizer,
                      make_optimizer(engine, std::move(parameters), settings, learning_rate));

        // The scaler outlives the trainer, whose compiled plan records its screens.
        std::unique_ptr<sa::Optim::LossScaler> scaler;

        if (scaling)
        {
            sa::Optim::LossScaleOptions scale_options;
            scale_options.initial = settings.loss_scale;

            SA_TRY_ASSIGN(scaler, sa::Optim::LossScaler::make(
                engine, pool.tensor(built.loss_scale), parameter_count, scale_options));

            SA_TRY(optimizer->set_loss_scaler(scaler.get()));
        }

        // Trainer::make lowers and compiles the step, the one compile of the run.
        sa::Train::TrainerOptions options;
        options.epochs = 1;

        // The trainer checks the graph against the policy it was built with.
        options.precision = precision;

        // The objective build_classifier traced; it feeds and measures every step.
        const sa::Train::CrossEntropyObjective objective(settings.batch_size, classes,
                                                         sa::DType::Float32);
        SA_TRY_ASSIGN(const std::unique_ptr<sa::Train::Trainer> trainer,
                      sa::Train::Trainer::make(engine, built, pool, objective, *optimizer,
                                               options));

        // The step count comes from the dataset, which the MNIST path may cap below --batches.
        const int64_t total_steps =
            settings.epochs * static_cast<int64_t>(dataset.batch_count());

        std::cout
            << "SushiAI MLP demo - configuration\n"
            << "----------------------------------------------------------------\n"
            << "SushiAI  : " << SushiAI::version_string()
            << " on SushiBLAS " << SUSHIBLAS_VERSION_STRING << "\n"
            << "Data     : " << data.provenance << "\n"
            << SushiLog::format_text("Model    : {} -> {} -> {}  ({} learnable scalars)\n",
                                       features, settings.hidden, classes,
                                       sa::NN::parameter_elements(model))
            << SushiLog::format_text("Optimizer: {} (lr {}{})\n",
                                       settings.optimizer == OptimizerKind::SGD ? "sgd" : "adamw",
                                       SushiLog::general(learning_rate),
                                       settings.optimizer == OptimizerKind::SGD
                                           ? SushiLog::format_text(
                                                 ", momentum {}", SushiLog::general(K_SGD_MOMENTUM))
                                           : std::string{})
            << SushiLog::format_text("Batches  : {} x {} from '{}'\n",
                                       dataset.batch_count(), settings.batch_size, dataset.name())
            << SushiLog::format_text("Graph    : {} IR nodes over {} values\n",
                                       built.graph.node_count(), built.graph.value_count())
            << describe_precision(built.casts)
            << (scaling
                    ? SushiLog::format_text("Scaling  : gradients scaled by {}, "
                                              "halved on overflow, doubled every {} clean "
                                              "steps\n",
                                              SushiLog::fixed(scaler->scale(), 0),
                                              scaler->state().options().growth_interval)
                    : std::string{"Scaling  : off; no node is emitted and the graph is "
                                  "unchanged\n"})
            << describe_fusion(built.fusion)
            << SushiLog::format_text("Buffers  : {} tensors, {} (one allocation each: {})\n",
                                       pool.tensor_count(), human_bytes(pool.allocated_bytes()),
                                       human_bytes(pool.unpacked_bytes()))
            << describe_arenas(pool)
            << SushiLog::format_text("Schedule : {} epochs x {} batches = {} steps, seed {}\n\n",
                                       settings.epochs, dataset.batch_count(), total_steps,
                                       settings.seed);

        // The trajectory is collected and printed as a block; see applications/examples/README.md.
        std::string trajectory;
        sa::Train::RunReport run;

        for (int64_t epoch = 1; epoch <= settings.epochs; ++epoch)
        {
            SA_TRY_ASSIGN(const sa::Train::RunReport epoch_report, trainer->fit(dataset));

            if (epoch == 1) run.first_loss = epoch_report.first_loss;
            run.last_loss = epoch_report.last_loss;
            run.last_metrics = epoch_report.last_metrics;
            run.operations_per_step = epoch_report.operations_per_step;
            run.steps += epoch_report.steps;
            run.duration_ms += epoch_report.duration_ms;

            trajectory += SushiLog::format_text(
                "  epoch {}/{} loss {} -> {}   accuracy {}%\n",
                SushiLog::right(epoch, 3), SushiLog::left(settings.epochs, 3),
                SushiLog::fixed(epoch_report.first_loss, 6),
                SushiLog::fixed(epoch_report.last_loss, 6),
                SushiLog::right(SushiLog::fixed(epoch_report.last_metrics.value_or("accuracy",
                                                                                   0.0) * 100.0,
                                                                                   2), 6));
        }
        run.compile_count = trainer->compile_count();

        // The compile count is the last line printed, in one write.
        std::cout
            << "\n================================================================\n"
            << " SushiAI MLP demo - result\n"
            << " " << (data.is_mnist ? "MNIST" : "synthetic fallback")
            << ", " << (settings.optimizer == OptimizerKind::SGD ? "sgd" : "adamw")
            << SushiLog::format_text(", {} -> {} -> {}\n", features, settings.hidden, classes)
            << "================================================================\n"
            << trajectory
            << "\n" << run.to_string()
            << (scaling ? scaler->to_string() : std::string{})
            << "----------------------------------------------------------------\n"
            << SushiLog::format_text(
                   "  compiles: {}  <- one lowering, one compile, {} replays.\n",
                   run.compile_count, run.steps)
            << "================================================================\n"
            << std::flush;

        SA_FAIL_IF(run.compile_count != 1, sa::StatusCode::InvalidDescription,
                   "The step was compiled {} times; it must be compiled exactly once.",
                   run.compile_count);

        return sa::Status();
    }

    /**
     * @brief Parses the command line, loads the data, and trains the chosen model.
     * @return Success, also after printing the usage text, or the first failure of a step.
     */
    [[nodiscard]] sa::Status run(int argc, char** argv)
    {
        bool wants_help = false;
        SA_TRY_ASSIGN(const Settings settings, parse_command_line(argc, argv, wants_help));
        if (wants_help)
        {
            std::cout << K_USAGE;
            return sa::Status();
        }

        SA_TRY_ASSIGN(sr::Runtime runtime, sa::Detail::create_runtime());
        sb::Engine engine(runtime);

        SA_TRY_ASSIGN(const DataChoice data, load_dataset(engine, settings));
        const sa::Data::Dataset& dataset = *data.dataset;

        const sa::Data::TargetDescription* const labels =
            sa::Data::find_target(dataset.targets(), sa::Data::K_ROLE_LABELS);
        SA_FAIL_IF(labels == nullptr, sa::StatusCode::InvalidDescription,
                   "the '{}' data publishes no '{}' role", dataset.name(),
                   sa::Data::K_ROLE_LABELS);

        const int64_t features = dataset.sample_shape().num_elements();
        const int64_t classes = labels->classes;

        // The activation is a type, so the choice is made here and nowhere after it.
        if (settings.activation == ActivationKind::GELU)
        {
            sa::NN::Sequential model{
                sa::NN::Linear{features, settings.hidden},
                sa::NN::GELU{},
                sa::NN::Linear{settings.hidden, classes},
            };
            return train_and_report(engine, model, data, dataset, settings, features, classes);
        }

        sa::NN::Sequential model{
            sa::NN::Linear{features, settings.hidden},
            sa::NN::ReLU{},
            sa::NN::Linear{settings.hidden, classes},
        };
        return train_and_report(engine, model, data, dataset, settings, features, classes);
    }

} // namespace

/** @brief Runs the demo and returns its exit code. */
int main(int argc, char** argv)
{
    try
    {
        const sa::Status status = run(argc, argv);
        if (!status.is_ok())
        {
            std::cerr << "sa demo mlp failed: " << status.message() << "\n";
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
