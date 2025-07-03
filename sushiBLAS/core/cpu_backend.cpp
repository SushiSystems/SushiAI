/**************************************************************************/
/*  cpu_backend.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2025-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include <omp.h>   
#include <algorithm>

#include "tensor.h"
#include "cpu_backend.h"

namespace SushiAI::SushiBLAS
{

    #pragma region Check Contiguous Macro

    #define CHECK_CONTIGUOUS(t)                                  \
                                                                 \
    do                                                           \
    {                                                            \
        const auto& s = (t).getStrides();                        \
        bool ok = true;                                          \
        std::size_t expect = 1;                                  \
                                                                 \
        for (int i = int(s.size()) - 1; i >= 0; --i)             \
        {                                                        \
            if (s[i] != expect)                                  \
                ok = false; break;                               \
                                                                 \
            expect *= (t).getShape()[i];                         \
        }                                                        \
                                                                 \
        assert(ok && "Tensor must be contiguous (row-major)");   \
    }                                                            \
    while (0)

    #pragma endregion

    #pragma region Constructor

    CPUBackend::CPUBackend(std::size_t block) : blockSize(block) {  }

    #pragma endregion
     
    #pragma region Operations

    void CPUBackend::gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta)
    {
        CHECK_CONTIGUOUS(A); 
        CHECK_CONTIGUOUS(B); 
        CHECK_CONTIGUOUS(C);

        const auto& Ash = A.getShape();
        const auto& Bsh = B.getShape();
        const auto& Csh = C.getShape();

        assert(Ash.size() == 2 && Bsh.size() == 2 && Csh.size() == 2);

        std::size_t M = Ash[0], K = Ash[1], N = Bsh[1];

        assert(Bsh[0] == K);
        assert(Csh[0] == M && Csh[1] == N);

        const float* Adata = A.getData().data();
        const float* Bdata = B.getData().data();
        float* Cdata = C.getData().data();

        if (beta != 1.0f) 
        {
            #pragma omp parallel for
            for (std::size_t i = 0; i < M * N; ++i)
                Cdata[i] *= beta;
        }

        std::size_t blockS = blockSize;
        for (std::size_t ii = 0; ii < M; ii += blockS) 
        {
            std::size_t iMax = std::min(ii + blockS, M);

            for (std::size_t kk = 0; kk < K; kk += blockS) 
            {
                std::size_t kMax = std::min(kk + blockS, K);
                for (std::size_t jj = 0; jj < N; jj += blockS) 
                {
                    std::size_t jMax = std::min(jj + blockS, N);

                    #pragma omp parallel for collapse(2) schedule(static)
                    for (std::size_t i = ii; i < iMax; ++i) 
                    {
                        for (std::size_t j = jj; j < jMax; ++j) 
                        {
                            float sum = 0.0f;

                            for (std::size_t k = kk; k < kMax; ++k)
                                sum += Adata[i * K + k] * Bdata[k * N + j];

                            Cdata[i * N + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }

    void CPUBackend::gemv(const Tensor& A, const Tensor& x, Tensor& y, float alpha, float beta) 
    {
        CHECK_CONTIGUOUS(A); 
        CHECK_CONTIGUOUS(x); 
        CHECK_CONTIGUOUS(y);
        const auto& Ash = A.getShape();
        assert(Ash.size() == 2 && x.getShape().size() == 1 && y.getShape().size() == 1);

        std::size_t M = Ash[0], N = Ash[1];
        assert(x.getTotalSize() == N && y.getTotalSize() == M);

        const float* Ad = A.getData().data();
        const float* xd = x.getData().data();
        float* yd = y.getData().data();

        #pragma omp parallel for
        for (std::size_t i = 0; i < M; ++i) 
        {
            float sum = 0.0f;

            for (std::size_t j = 0; j < N; ++j)
                sum += Ad[i * N + j] * xd[j];

            yd[i] = alpha * sum + beta * yd[i];
        }
    }

    void CPUBackend::axpy(float alpha, const Tensor& x, Tensor& y) 
    {
        CHECK_CONTIGUOUS(x); 
        CHECK_CONTIGUOUS(y);
        assert(x.getTotalSize() == y.getTotalSize());

        const float* xd = x.getData().data();
        float* yd = y.getData().data();
        std::size_t  n = x.getTotalSize();

        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i)
            yd[i] += alpha * xd[i];
    }

    float CPUBackend::dot(const Tensor& x, const Tensor& y) 
    {
        CHECK_CONTIGUOUS(x); 
        CHECK_CONTIGUOUS(y);
        assert(x.getTotalSize() == y.getTotalSize());

        const float* xd = x.getData().data();
        const float* yd = y.getData().data();
        std::size_t  n = x.getTotalSize();

        float acc = 0.0f;

        #pragma omp parallel for reduction(+:acc)
        for (std::size_t i = 0; i < n; ++i)
            acc += xd[i] * yd[i];

        return acc;
    }

    void CPUBackend::relu(Tensor& x) 
    {
        CHECK_CONTIGUOUS(x);
        float* d = x.getData().data();
        std::size_t n = x.getTotalSize();

        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i)
            if (d[i] < 0.0f) d[i] = 0.0f;
    }

    #pragma endregion
}
