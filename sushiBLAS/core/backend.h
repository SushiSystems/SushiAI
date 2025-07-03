/**************************************************************************/
/*  backend.h                                                             */
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

#pragma once

#include <memory>

#include "tensor.h"

namespace SushiAI::SushiBLAS
{
    class Backend 
    {
        public:
            virtual ~Backend() = default;

            #pragma region Operations

            virtual void axpy(float alpha, const Tensor& x, Tensor& y) = 0;
            virtual float dot(const Tensor& x, const Tensor& y) = 0;

            virtual void gemv(const Tensor& A, const Tensor& x, Tensor& y, float alpha = 1.0f, float beta = 0.0f) = 0;

            virtual void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f, float beta = 0.0f) = 0;

            #pragma endregion

            #pragma region Element-Wise Activation Functions

            virtual void relu(Tensor& x) = 0;

            #pragma endregion
    };

    #pragma region Get & Set Backend

    Backend& getBackend();
    void setBackend(std::unique_ptr<Backend> b);

    #pragma endregion

    #pragma region Operations

    inline float dot(const Tensor& x, const Tensor& y)
    {
        return getBackend().dot(x, y);
    }
    inline float dot(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& y)
    {
        return dot(*x, *y);
    }

    inline void axpy(float alpha, const Tensor& x, Tensor& y)
    {
        getBackend().axpy(alpha, x, y);
    }
    inline void axpy(float alpha, const std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y)
    {
        axpy(alpha, *x, *y);
    }

    inline void gemv(const Tensor& A, const Tensor& x, Tensor& y, float alpha = 1.0f, float beta = 0.0f) 
    {
        getBackend().gemv(A, x, y, alpha, beta);
    }
    inline void gemv(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y, float alpha = 1.0f, float beta = 0.0f)
    {
        gemv(*A, *x, *y, alpha, beta);
    }


    inline void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha = 1.0f, float beta = 0.0f)
    {
        getBackend().gemm(A, B, C, alpha, beta);
    }
    inline void gemm(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B, const std::shared_ptr<Tensor>& C, float alpha = 1.0f, float beta = 0.0f)
    {
        gemm(*A, *B, *C, alpha, beta);
    } 

    #pragma endregion  

    #pragma region Element-Wise Activation Functions

    inline void relu(Tensor& x)
    {
        getBackend().relu(x);
    }
    inline void relu(std::shared_ptr<Tensor>& x)
    {
        relu(*x);
    }

    #pragma endregion
}
