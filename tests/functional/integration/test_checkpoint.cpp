/**************************************************************************/
/* test_checkpoint.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// The device half of checkpointing: binding records to the pool's tensors.
// The byte format's own malformed-file surface is device-free and lives in
// tests/functional/unit/test_checkpoint_format.cpp.
//
// The point of this file is the last test in it. A Sequential and a LayerStack
// built from the equivalent model.json are two spellings of one model, and the
// claim that they ARE one model is proved by a checkpoint written by either
// loading into the other - not by a separate mechanism whose own correctness
// would then be the open question.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"
#include "../../common/test_throws.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Checkpoint : public SaTest::EngineTest {};

    constexpr std::string_view MODEL_JSON = R"({
        "sushiai_model": 1,
        "input": {"features": 6},
        "layers": [{"type": "linear", "units": 8},
                   {"type": "gelu"},
                   {"type": "linear", "units": 4}]
    })";

    /** @brief The same architecture, spelled at compile time. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp make_sequential()
    {
        return Mlp(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    NN::LayerStack make_stack()
    {
        return NN::build_layer_stack(
            Config::parse_model(Config::parse_json(MODEL_JSON, "<test>"), "<test>"));
    }

    /**
     * @class Materialised
     * @brief A traced, bound and initialised model, and the pool behind it.
     *
     * Owns the pool, because a TensorPool hands out references into its own
     * map and every tensor a test holds has to outlive the comparison.
     */
    template<typename M>
    class Materialised
    {
        public:
            Materialised(SushiBLAS::Engine& engine, M model, std::uint64_t seed)
                : model_(std::move(model)), pool_(engine)
            {
                const Graph::GraphBuilder builder(graph_);
                const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
                (void)model_.forward(builder, x);

                // Parameters are leaves, so one call allocates every one of
                // them - and the input leaf besides, which costs a [4, 6]
                // buffer and keeps this fixture from having to know which
                // leaves are which.
                pool_.materialise_leaves(graph_);

                NN::InitPolicy policy;
                policy.seed = seed;
                policy.zero_final_layer = false;   // so every parameter carries real numbers
                NN::initialize(engine, model_, pool_, policy);
            }

            M& model() noexcept { return model_; }
            Graph::TensorPool& pool() noexcept { return pool_; }

            /** @brief Every parameter's raw bit patterns, in walk order. */
            std::vector<std::uint32_t> bits()
            {
                std::vector<std::uint32_t> out;
                NN::for_each_parameter(model_, [&](const std::string&, NN::Parameter& parameter)
                {
                    const SushiBLAS::Tensor& tensor = pool_.tensor(parameter.id());
                    const float* values = tensor.data_as<float>();
                    for (int64_t i = 0; i < tensor.num_elements; ++i)
                    {
                        std::uint32_t pattern = 0;
                        std::memcpy(&pattern, &values[i], sizeof(pattern));
                        out.push_back(pattern);
                    }
                });
                return out;
            }

            /** @brief Every parameter's dotted name, in walk order. */
            std::vector<std::string> names()
            {
                std::vector<std::string> out;
                NN::for_each_parameter(model_, [&out](const std::string& name, NN::Parameter&)
                {
                    out.push_back(name);
                });
                return out;
            }

        private:
            M model_;
            Graph::Graph graph_{};
            Graph::TensorPool pool_;
    };

    /**
     * @class ScratchFile
     * @brief A checkpoint path that removes itself.
     */
    class ScratchFile
    {
        public:
            explicit ScratchFile(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ~ScratchFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            ScratchFile(const ScratchFile&) = delete;
            ScratchFile& operator=(const ScratchFile&) = delete;

            const std::filesystem::path& path() const noexcept { return path_; }

        private:
            std::filesystem::path path_;
    };
}

TEST_F(Integration_Checkpoint, SavesAndLoadsEveryParameterBitForBit)
{
    const ScratchFile file("sushiai_roundtrip.ckpt");

    Materialised<Mlp> trained(*engine, make_sequential(), 11);
    const std::vector<std::uint32_t> expected = trained.bits();

    IO::save_checkpoint(file.path(), trained.model(), trained.pool(), *engine);

    // A different seed, so a load that did nothing at all would be visible.
    Materialised<Mlp> fresh(*engine, make_sequential(), 22);
    ASSERT_NE(fresh.bits(), expected);

    IO::load_checkpoint(file.path(), fresh.model(), fresh.pool());
    EXPECT_EQ(fresh.bits(), expected);
}

TEST_F(Integration_Checkpoint, ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack)
{
    // THE equivalence test. Two spellings of one model prove they describe the
    // same thing by exchanging a checkpoint - so the compile-time and runtime
    // paths need no separate mechanism to be kept in step, and cannot drift
    // without this failing.
    const ScratchFile from_sequential("sushiai_equivalence_a.ckpt");
    const ScratchFile from_stack("sushiai_equivalence_b.ckpt");

    Materialised<Mlp> compile_time(*engine, make_sequential(), 7);
    Materialised<NN::LayerStack> run_time(*engine, make_stack(), 99);

    // The names must match first; the checkpoint is matched by name, so a
    // naming difference would surface as "does not supply" rather than as the
    // clearer statement that the two disagree about what they hold.
    ASSERT_EQ(compile_time.names(), run_time.names());
    ASSERT_EQ(compile_time.names(),
              (std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"}));

    const std::vector<std::uint32_t> sequential_bits = compile_time.bits();
    ASSERT_NE(run_time.bits(), sequential_bits);

    // Sequential -> LayerStack
    IO::save_checkpoint(from_sequential.path(), compile_time.model(),
                        compile_time.pool(), *engine);
    IO::load_checkpoint(from_sequential.path(), run_time.model(), run_time.pool());
    EXPECT_EQ(run_time.bits(), sequential_bits);

    // ... and LayerStack -> Sequential, over a third model so the return trip
    // is a real load rather than a comparison against what was already there.
    Materialised<Mlp> third(*engine, make_sequential(), 123);
    ASSERT_NE(third.bits(), sequential_bits);

    IO::save_checkpoint(from_stack.path(), run_time.model(), run_time.pool(), *engine);
    IO::load_checkpoint(from_stack.path(), third.model(), third.pool());
    EXPECT_EQ(third.bits(), sequential_bits);
}

TEST_F(Integration_Checkpoint, AParameterTheFileDoesNotSupplyThrowsAndNamesIt)
{
    const ScratchFile file("sushiai_missing.ckpt");

    // A checkpoint of a DIFFERENT architecture, so "2.weight" and "2.bias" are
    // absent from it entirely.
    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 5);
    IO::save_checkpoint(file.path(), shallow.model(), shallow.pool(), *engine);

    Materialised<Mlp> deep(*engine, make_sequential(), 6);
    SA_EXPECT_ERROR(IO::load_checkpoint(file.path(), deep.model(), deep.pool()),
                    "does not supply", "2.weight");
}

TEST_F(Integration_Checkpoint, AParameterTheModelDoesNotHaveThrowsAndNamesIt)
{
    const ScratchFile file("sushiai_extra.ckpt");

    Materialised<Mlp> deep(*engine, make_sequential(), 5);
    IO::save_checkpoint(file.path(), deep.model(), deep.pool(), *engine);

    using Shallow = NN::Sequential<NN::Linear>;
    Materialised<Shallow> shallow(*engine, Shallow(NN::Linear{6, 8}), 6);
    SA_EXPECT_ERROR(IO::load_checkpoint(file.path(), shallow.model(), shallow.pool()),
                    "2.weight", "which this model does not have");
}

TEST_F(Integration_Checkpoint, AShapeMismatchThrowsWithBothShapes)
{
    const ScratchFile file("sushiai_shape.ckpt");

    // Same names, different widths - which an element-count comparison would
    // catch here but not for a transpose. The shape is what is compared.
    using OneLayer = NN::Sequential<NN::Linear>;
    Materialised<OneLayer> wide(*engine, OneLayer(NN::Linear{6, 8}), 5);
    IO::save_checkpoint(file.path(), wide.model(), wide.pool(), *engine);

    Materialised<OneLayer> narrow(*engine, OneLayer(NN::Linear{6, 4}), 6);
    SA_EXPECT_ERROR(IO::load_checkpoint(file.path(), narrow.model(), narrow.pool()),
                    "0.weight", "[8, 6]", "[4, 6]");
}

TEST_F(Integration_Checkpoint, ADuplicatedNameThrowsAndNamesTheDuplicate)
{
    const ScratchFile file("sushiai_duplicate.ckpt");

    // Hand-built, because save_checkpoint cannot produce one: the walk yields
    // each name once. This is exactly the file a counter-based loader would
    // accept while leaving a parameter reading uninitialised device memory.
    Materialised<Mlp> model(*engine, make_sequential(), 5);

    {
        const std::vector<float> weight(8 * 6, 1.0f);
        const std::vector<float> bias(8, 2.0f);
        std::ofstream out(file.path(), std::ios::binary);
        IO::CheckpointWriter writer(out, 4);
        writer.write(IO::TensorRecord{"0.weight", DType::FLOAT32, Shape{8, 6}}, weight.data());
        writer.write(IO::TensorRecord{"0.bias", DType::FLOAT32, Shape{8}}, bias.data());
        writer.write(IO::TensorRecord{"0.bias", DType::FLOAT32, Shape{8}}, bias.data());
        writer.write(IO::TensorRecord{"2.bias", DType::FLOAT32, Shape{4}}, bias.data());
        writer.finish();
    }

    SA_EXPECT_ERROR(IO::load_checkpoint(file.path(), model.model(), model.pool()),
                    "supplies", "0.bias", "twice");
}
