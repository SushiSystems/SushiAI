/****************************************************************/
/* test_config.cpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_config.cpp
 * @brief Tests how the config layer parses its JSON documents and which ones it refuses.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/module.hpp>
#include "../../common/test_throws.hpp"

using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    /** @brief A span over a literal allowed-key list, for reject_unknown_keys. */
    template<std::size_t N>
    SushiRuntime::span<const std::string_view> allowed(const std::string_view (&keys)[N])
    {
        return SushiRuntime::span<const std::string_view>(keys, N);
    }
}

TEST(Unit_Config, ParsesEveryScalarKind)
{
    const Value root = parse_json(
        R"({"a": 1, "b": 2.5, "c": "text", "d": true, "e": null, "f": [1, 2]})", "<test>");

    ASSERT_EQ(root.kind(), Kind::OBJECT);
    EXPECT_EQ(root.at("a", "<test>").as_int("a"), 1);
    EXPECT_DOUBLE_EQ(root.at("b", "<test>").as_real("b"), 2.5);
    EXPECT_EQ(root.at("c", "<test>").as_string("c"), "text");
    EXPECT_TRUE(root.at("d", "<test>").as_bool("d"));
    EXPECT_TRUE(root.at("e", "<test>").is_null());
    EXPECT_EQ(root.at("f", "<test>").as_array("f").size(), 2u);
}

TEST(Unit_Config, KeepsTheKeyOrderTheFileSpelled)
{
    // Key order follows the file; see tests/functional/unit/README.md.
    const Value root = parse_json(R"({"zebra": 1, "apple": 2, "mango": 3})", "<test>");
    EXPECT_EQ(root.keys(), (std::vector<std::string>{"zebra", "apple", "mango"}));
}

TEST(Unit_Config, AnIntegerReadsAsARealButARealIsNotAWholeNumber)
{
    // An integer widens to a real, never the reverse; see tests/functional/unit/README.md.
    const Value root = parse_json(R"({"lr": 3, "units": 2.5})", "<test>");

    EXPECT_DOUBLE_EQ(root.at("lr", "<test>").as_real("lr"), 3.0);
    EXPECT_EQ(root.at("lr", "<test>").as_int("lr"), 3);
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("'units'"),
                    "'units'", "whole number");
}

TEST(Unit_Config, AWrongKindNamesBothTheContextAndTheKindFound)
{
    const Value root = parse_json(R"({"units": "many"})", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "<test>").as_int("layer 0 'units'"),
                    "layer 0 'units'", "a string");
}

TEST(Unit_Config, AMissingKeyNamesTheKeyAndListsWhatIsPresent)
{
    const Value root = parse_json(R"({"alpha": 1, "beta": 2})", "model.json");
    SA_EXPECT_ERROR((void)root.at("gamma", "model.json"),
                    "model.json", "'gamma'", "'alpha'", "'beta'");
}

TEST(Unit_Config, ReadingAKeyOffANonObjectSaysWhatItActuallyIs)
{
    const Value root = parse_json(R"([1, 2, 3])", "<test>");
    SA_EXPECT_ERROR((void)root.at("units", "the model"), "the model", "an array");
}

TEST(Unit_Config, AnUnknownKeyIsAnErrorThatListsWhatIsAllowed)
{
    const Value root = parse_json(R"({"type": "linear", "unitss": 256})", "model.json");
    const std::string_view keys[] = {"type", "units"};

    SA_EXPECT_ERROR(root.reject_unknown_keys("layer 0 'linear'", allowed(keys)),
                    "layer 0 'linear'", "'unitss'", "'units'", "'type'");
}

TEST(Unit_Config, AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything)
{
    const Value with_keys = parse_json(R"({"type": "linear", "units": 256})", "<test>");
    const std::string_view keys[] = {"type", "units"};
    EXPECT_NO_THROW(with_keys.reject_unknown_keys("layer 0", allowed(keys)));

    // An empty allowed set refuses every key; see tests/functional/unit/README.md.
    const std::string_view none[] = {"type"};
    const Value stray = parse_json(R"({"type": "gelu", "units": 8})", "<test>");
    SA_EXPECT_ERROR(stray.reject_unknown_keys("layer 1 'gelu'", allowed(none)),
                    "layer 1 'gelu'", "'units'");
}

TEST(Unit_Config, WithoutRemovesTheDiscriminatorAndLeavesTheRest)
{
    const Value layer = parse_json(R"({"type": "linear", "units": 256, "use_bias": false})",
                                   "<test>");
    const Value attributes = layer.without("type");

    EXPECT_FALSE(attributes.has("type"));
    EXPECT_TRUE(attributes.has("units"));
    EXPECT_EQ(attributes.keys(), (std::vector<std::string>{"units", "use_bias"}));

    // Removing an absent key is not an error; see tests/functional/unit/README.md.
    EXPECT_EQ(attributes.without("type").keys().size(), 2u);
}

TEST(Unit_Config, AtOrFallsBackWithoutThrowing)
{
    const Value root = parse_json(R"({"units": 256})", "<test>");
    const Value fallback = Value::boolean(true);

    EXPECT_EQ(root.at_or("units", fallback).as_int("units"), 256);
    EXPECT_TRUE(root.at_or("use_bias", fallback).as_bool("use_bias"));

    // A non-object falls back instead of throwing; see tests/functional/unit/README.md.
    const Value not_an_object = Value::integer(1);
    EXPECT_TRUE(not_an_object.at_or("use_bias", fallback).as_bool("use_bias"));
}

TEST(Unit_Config, MalformedJsonNamesTheOriginAndTheComplaint)
{
    SA_EXPECT_ERROR((void)parse_json("{ not json", "train.json"),
                    "train.json", "not valid JSON");
}

TEST(Unit_Config, AnObjectBuiltFromMismatchedVectorsIsRefused)
{
    std::vector<std::string> keys{"a", "b"};
    std::vector<Value> values{Value::integer(1)};
    SA_EXPECT_ERROR((void)Value::object(std::move(keys), std::move(values)), "2 keys", "1 value");
}

namespace
{
    /** @brief Parses @p text as a model document called "model.json". */
    SushiAI::Config::ModelSpec model_from(std::string_view text)
    {
        return SushiAI::Config::parse_model(parse_json(text, "model.json"), "model.json");
    }

    constexpr std::string_view GOOD_MODEL = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";
}

TEST(Unit_Config, ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes)
{
    const SushiAI::Config::ModelSpec model = model_from(GOOD_MODEL);

    EXPECT_EQ(model.input_features, 6);
    ASSERT_EQ(model.layers.size(), 3u);
    EXPECT_EQ(model.layers[0].type, "linear");
    EXPECT_EQ(model.layers[1].type, "gelu");
    EXPECT_EQ(model.layers[2].type, "linear");

    // The attributes no longer carry "type"; see tests/functional/unit/README.md.
    EXPECT_FALSE(model.layers[0].attributes.has("type"));
    EXPECT_TRUE(model.layers[0].attributes.has("units"));
    EXPECT_EQ(model.layers[1].attributes.keys().size(), 0u);
}

TEST(Unit_Config, ChainsLayerWidthsFromTheInputFeatures)
{
    SushiAI::NN::LayerStack stack = SushiAI::NN::build_layer_stack(model_from(GOOD_MODEL));

    EXPECT_EQ(stack.size(), 3u);
    EXPECT_EQ(stack.output_features(), 4);

    // An activation keeps the width it was handed; see tests/functional/unit/README.md.
    EXPECT_EQ(stack.layer(0).output_features(), 8);
    EXPECT_EQ(stack.layer(1).output_features(), 8);
}

TEST(Unit_Config, AConfiguredStackTracesToTheDeclaredOutputWidth)
{
    SushiAI::NN::LayerStack stack = SushiAI::NN::build_layer_stack(model_from(GOOD_MODEL));

    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);
    const SushiAI::Graph::ValueId x = builder.input(SushiAI::Shape{4, 6}, "inputs");

    EXPECT_EQ(graph.value(stack.forward(builder, x)).shape, (SushiAI::Shape{4, 4}));
}

TEST(Unit_Config, AnUnknownLayerTypeListsEveryTypeThisBuildHas)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "conv2d", "units": 8}]
    })");

    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(model),
                    "conv2d", "layer type", "'linear'", "'gelu'");
}

TEST(Unit_Config, AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory)
{
    const SushiAI::Config::ModelSpec typo = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "unitss": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(typo), "layer 0 'linear'", "'unitss'");

    // An activation accepts no attribute; see tests/functional/unit/README.md.
    const SushiAI::Config::ModelSpec stray = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "gelu", "units": 8}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(stray), "layer 0 'gelu'", "'units'");
}

TEST(Unit_Config, ALayerFactoryRefusesANonPositiveWidth)
{
    const SushiAI::Config::ModelSpec model = model_from(R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 0}]
    })");
    SA_EXPECT_ERROR((void)SushiAI::NN::build_layer_stack(model), "layer 0 'linear'", "'units'");
}

TEST(Unit_Config, UseBiasDefaultsToTrueAndCanBeTurnedOff)
{
    SushiAI::NN::LayerStack with_bias = SushiAI::NN::build_layer_stack(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4}]
    })"));
    SushiAI::NN::LayerStack without_bias = SushiAI::NN::build_layer_stack(model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "linear", "units": 4, "use_bias": false}]
    })"));

    SushiAI::Graph::Graph a;
    SushiAI::Graph::Graph b;
    const SushiAI::Graph::GraphBuilder builder_a(a);
    const SushiAI::Graph::GraphBuilder builder_b(b);
    (void)with_bias.forward(builder_a, builder_a.input(SushiAI::Shape{2, 6}, "x"));
    (void)without_bias.forward(builder_b, builder_b.input(SushiAI::Shape{2, 6}, "x"));

    EXPECT_EQ(SushiAI::NN::parameter_count(with_bias), 2u);
    EXPECT_EQ(SushiAI::NN::parameter_count(without_bias), 1u);
}

TEST(Unit_Config, AMissingModelVersionIsAnErrorThatSaysWhy)
{
    SA_EXPECT_ERROR((void)model_from(R"({"input": {"features": 6}, "layers": []})"),
                    "sushiai_model", "misread");
}

TEST(Unit_Config, AFutureModelVersionIsAnError)
{
    SA_EXPECT_ERROR(
        (void)model_from(R"({"sushiai_model": 2, "input": {"features": 6}, "layers": []})"),
        "version 2", "version 1");
}

TEST(Unit_Config, AnUnknownTopLevelModelKeyIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6},
        "layers": [{"type": "gelu"}], "optimizer": {"type": "adamw"}
    })"), "'optimizer'", "'layers'");
}

TEST(Unit_Config, AModelWithNoLayersIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": []
    })"), "no layers");
}

TEST(Unit_Config, ANonPositiveInputWidthIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 0}, "layers": [{"type": "gelu"}]
    })"), "'features'", "positive");
}

TEST(Unit_Config, ALayerThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_ERROR((void)model_from(R"({
        "sushiai_model": 1, "input": {"features": 6}, "layers": ["linear"]
    })"), "layer 0", "a string");
}

namespace
{
    /** @brief Parses @p text as a train document, resolving paths against ".". */
    SushiAI::Config::TrainSpec train_from(std::string_view text)
    {
        return SushiAI::Config::parse_train(parse_json(text, "train.json"), "train.json", ".");
    }

    constexpr std::string_view GOOD_TRAIN = R"({
        "sushiai_train": 1,
        "model": "model.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings", "batch_size": 8, "batches": 2, "classes": 4},
        "optimizer": {"type": "adamw", "learning_rate": 0.003},
        "epochs": 3,
        "checkpoint": "out/model.ckpt"
    })";

    /**
     * @class TemporaryDirectory
     * @brief A directory that removes itself, so a failing test leaves nothing behind.
     */
    class TemporaryDirectory
    {
        public:
            explicit TemporaryDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path_);
                std::filesystem::create_directories(path_);
            }

            ~TemporaryDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            TemporaryDirectory(const TemporaryDirectory&) = delete;
            TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

            const std::filesystem::path& path() const noexcept { return path_; }

            /** @brief Writes @p text into this directory as @p name. */
            std::filesystem::path write(const char* name, std::string_view text) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary);
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                return file;
            }

        private:
            std::filesystem::path path_;
    };
}

TEST(Unit_Config, ParsesATrainDocumentIntoItsFourTypedBlocks)
{
    const SushiAI::Config::TrainSpec spec = train_from(GOOD_TRAIN);

    EXPECT_EQ(spec.objective_type, "cross_entropy");
    EXPECT_EQ(spec.data.type, "rings");
    EXPECT_EQ(spec.optimizer_type, "adamw");
    EXPECT_EQ(spec.epochs, 3);

    // Each block's "type" is separated out, so no factory has to allow it.
    EXPECT_FALSE(spec.optimizer_attributes.has("type"));
    EXPECT_DOUBLE_EQ(spec.optimizer_attributes.at("learning_rate", "optimizer")
                                              .as_real("learning_rate"), 0.003);
    EXPECT_EQ(spec.data.attributes.at("batch_size", "data").as_int("batch_size"), 8);
}

TEST(Unit_Config, ResolvesRelativePathsAgainstTheConfigsOwnDirectory)
{
    const TemporaryDirectory directory("sushiai_config_paths");
    const std::filesystem::path config = directory.write("train.json", GOOD_TRAIN);

    const SushiAI::Config::TrainSpec spec = SushiAI::Config::load_train(config);

    EXPECT_EQ(spec.model_path, directory.path() / "model.json");
    EXPECT_EQ(spec.checkpoint, directory.path() / "out" / "model.ckpt");
}

TEST(Unit_Config, LeavesAnAbsolutePathAlone)
{
    const std::filesystem::path absolute =
        std::filesystem::temp_directory_path() / "elsewhere" / "model.json";

    // generic_string() writes '/', which JSON does not escape; see tests/functional/unit/README.md.
    const std::string text =
        R"({"sushiai_train": 1, "model": ")" + absolute.generic_string() +
        R"(", "objective": {"type": "cross_entropy"},
            "data": {"type": "rings"}, "optimizer": {"type": "adamw"}})";

    const SushiAI::Config::TrainSpec spec =
        SushiAI::Config::parse_train(parse_json(text, "train.json"), "train.json",
                                     "/some/other/place");
    EXPECT_EQ(spec.model_path, absolute);
}

TEST(Unit_Config, EpochsDefaultsToOneAndMustBePositive)
{
    const SushiAI::Config::TrainSpec defaulted = train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })");
    EXPECT_EQ(defaulted.epochs, 1);
    EXPECT_TRUE(defaulted.checkpoint.empty());

    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochs": 0
    })"), "'epochs'", "positive");
}

TEST(Unit_Config, AnUnknownTopLevelTrainKeyIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochz": 3
    })"), "'epochz'", "'epochs'");
}

TEST(Unit_Config, AMissingTrainBlockNamesIt)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "optimizer": {"type": "adamw"}
    })"), "'data'");
}

TEST(Unit_Config, ATypedBlockThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": "cross_entropy",
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), "'objective'", "a string");
}

TEST(Unit_Config, AFutureTrainVersionIsAnError)
{
    SA_EXPECT_ERROR((void)train_from(R"({
        "sushiai_train": 2, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), "version 2", "version 1");
}

TEST(Unit_Config, ParsesAnEvalDocument)
{
    const SushiAI::Config::DataSpec data = SushiAI::Config::parse_data(
        parse_json(R"({
            "sushiai_data": 1,
            "data": {"type": "rings", "batch_size": 512, "batches": 1, "seed": 3735928575}
        })", "eval.json"), "eval.json");

    EXPECT_EQ(data.type, "rings");
    EXPECT_EQ(data.attributes.at("batch_size", "data").as_int("batch_size"), 512);
    EXPECT_EQ(data.attributes.at("seed", "data").as_int("seed"), 3735928575);
}

TEST(Unit_Config, AnEvalDocumentCarriesNoModelBlock)
{
    // An eval document naming a model is an error; see tests/functional/unit/README.md.
    SA_EXPECT_ERROR((void)SushiAI::Config::parse_data(
        parse_json(R"({"sushiai_data": 1, "data": {"type": "rings"}, "model": "m.json"})",
                   "eval.json"), "eval.json"), "'model'", "'data'");
}

TEST(Unit_Config, LoadingAnAbsentFileNamesThePath)
{
    SA_EXPECT_ERROR((void)SushiAI::Config::load_train("no/such/train.json"),
                    "no/such/train.json");
}
