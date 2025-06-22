#pragma once
#include <cmath>
#include <random>
#include <algorithm>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#include "tensor.h"

namespace SushiAI 
{
    class Initializer 
    {
        public:
            virtual ~Initializer() = default;

            virtual void initialize(const std::shared_ptr<Tensor>& t) const = 0;
    };

    // 1) Uniform (a,b)
    class UniformInitializer : public Initializer
    {
        float lower, upper;

        public:
            UniformInitializer(float a, float b) : lower(a), upper(b) {}

            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<float> dist(lower, upper);

                auto& ptr = t -> getData();
                size_t N = t -> getTotalSize();

                for (size_t i = 0; i < N; ++i)
                    ptr[i] = dist(gen);
            }
        };

    // 2) Normal (μ,σ)
    class NormalInitializer : public Initializer
    {
        float mean, stddev;

        public:
            NormalInitializer(float m, float s) : mean(m), stddev(s) {}

            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                std::mt19937 gen(std::random_device{}());
                std::normal_distribution<float> dist(mean, stddev);

                auto& ptr = t -> getData();
                size_t N = t -> getTotalSize();

                for (size_t i = 0; i < N; ++i)
                    ptr[i] = dist(gen);
            }
    };

    // Yardımcı: fan-in ve fan-out hesaplama
    inline void computeFans(const std::shared_ptr<Tensor>& t, int& fan_in, int& fan_out)
    {
        auto shape = t -> getShape();

        if (shape.size() == 2)
        {
            fan_in = shape[0];
            fan_out = shape[1];
        }
        else if (shape.size() > 2)
        {
            // Conv: [C_out, C_in, kH, kW]
            fan_in = shape[1] * shape[2] * shape[3];
            fan_out = shape[0] * shape[2] * shape[3];
        }
        else
            fan_in = fan_out = 1;
    }

    // 3) Xavier / Glorot Uniform
    class XavierUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(6.0f / (fan_in + fan_out));
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    // 4) Xavier / Glorot Normal
    class XavierNormal : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float stddev = std::sqrt(2.0f / (fan_in + fan_out));
                NormalInitializer n(0.0f, stddev);
                n.initialize(t);
            }
    };

    // 5) He / Kaiming Uniform
    class HeUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(6.0f / fan_in);
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    // 6) He / Kaiming Normal
    class HeNormal : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float stddev = std::sqrt(2.0f / fan_in);
                NormalInitializer n(0.0f, stddev);
                n.initialize(t);
            }
    };

    // 7) LeCun Uniform
    class LeCunUniform : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                int fan_in, fan_out;
                computeFans(t, fan_in, fan_out);

                float bound = std::sqrt(3.0f / fan_in);
                UniformInitializer u(-bound, bound);
                u.initialize(t);
            }
    };

    // 8) Orthogonal Initialization (2D matrisler için)
    class OrthogonalInitializer : public Initializer
    {
        public:
            void initialize(const std::shared_ptr<Tensor>& t) const override
            {
                auto shape = t->getShape();

                if (shape.size() != 2)
                    throw std::runtime_error("Orthogonal only supports 2D tensors");

                int rows = shape[0], cols = shape[1];

                std::vector<float> mat(rows * cols);
                std::mt19937 gen(std::random_device{}());
                std::normal_distribution<float> dist(0.0f, 1.0f);

                for (auto& x : mat)
                    x = dist(gen);

                // Eigen kullanarak QR ayrıştırması (alternatif: Gram-Schmidt)
                #ifdef USE_EIGEN
                    Eigen::Map<Eigen::MatrixXf> M(mat.data(), rows, cols);
                    Eigen::HouseholderQR<Eigen::MatrixXf> qr(M);
                    Eigen::MatrixXf Q = qr.householderQ();

                    std::copy(Q.data(), Q.data() + rows * cols, t.data());
                #else
                    throw std::runtime_error("Orthogonal requires Eigen support");
                #endif
        }
    };
}
