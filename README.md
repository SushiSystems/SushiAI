# SushiAI

**Lightweight Deep Learning Framework;** **designed to be fast, minimal, and extensible, starting with MLP support on CPU.**

---

## About Project

**SushiAI** is an artificial intelligence framework developed at [Sushi Systems](https://www.sushisystems.io). Its primary purpose is to serve as the AI backend for **Sushiverse**, a virtual world project, by training the agents that operate within it.

The framework was built from the ground up to address the need for a lightweight, CPU-friendly AI library suitable for game development and real-time simulations. This approach was also taken to deepen the core understanding of neural networks and agent-based systems.

---

## Features

- Lightweight and fast tensor engine
- Autograd (automatic differentiation)
- Built-in optimizers: SGD, Adam
- Modular activation functions: ReLU, Tanh, Sigmoid, Softmax
- Model serialization (save & load trained models)
- Core operations powered by SushiBLAS (embedded BLAS backend) (WIP)

---

## Installation

### Build from Source

```bash
git clone https://github.com/YOUR_USERNAME/SushiAI.git
cd SushiAI
mkdir build && cd build
cmake ..
make -j4
```

> **Requirements:**
> - CMake ≥ 3.14
> - C++14-compliant compiler (GCC or Clang)

---

## Getting Started

Below is a minimal example for training a simple Multi-Layer Perceptron (MLP) model to predict constant values:

```cpp
#include "tensor.h"
#include "ops.h"
#include "loss.h"
#include "layer.h"
#include "optimizer.h"
#include "sequential.h"
#include "initializer.h"

using namespace SushiAI;

int main() 
{
    auto model = std::make_shared<Sequential>();
    model -> add(std::make_shared<Linear>(2, 16, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model -> add(std::make_shared<ReLU>());
    model -> add(std::make_shared<Linear>(16, 1, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));

    auto input = Tensor::Ones({100, 2});
    auto target = Tensor::Zeros({100, 1});

    auto lossFunction = std::make_shared<MSELoss>();
    auto optimizer = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        auto prediction = model -> forward(input);
        auto loss = lossFunction -> forward(prediction, target);

        loss -> backward();
        optimizer -> step(model -> parameters());
        optimizer -> zeroGradient(model -> parameters());

        std::cout << "Epoch " << epoch << " Loss: " << loss -> getData()[0] << std::endl;
    }

    model -> saveModel("mlp.sushi");

    return 0;
}
```

## Project Structure

```bash
SushiAI/
└── core/ 
    ├── loss/           # MSE, Cross Entropy, etc.
    ├── neuralnetwork/  # Linear (Dense), Sequential, Activations, etc.
    ├── operations/     # add, matmul, element-wise ReLU, Tanh, etc.
    ├── optimizer/      # Adam, SGD, etc.
    └── tensor/         # Tensor engine
└── docs/               # Documents
    └── versions/       # Version log
├── models/             # SushiAI Model save & load directory (.sushi)
└── sushiBLAS/          # SushiBLAS Linear Algebra library
    └── core/           # BLAS Backend
```

---

## Examples

You can find more detailed examples in the `/examples` directory:
- [Training an MLP on the XOR problem](examples/mlp_xor.cpp)
- [Simple Linear Regression](examples/simple_regression.cpp)

---

## Roadmap

| Feature                   | Status     |
| ------------------------- | ---------- |
| Tensor Engine             | ✅ Done    |
| Autograd Engine           | ✅ Done    |
| ANN Structure (MLP)       | ✅ Done    |
| Optimizers (Adam, SGD)    | ✅ Done    |
| Losses (MSE, CrossE)      | ✅ Done    |
| BLAS Backend (SushiBLAS)  | 🔜 In Dev  |
| Dynamic Computation Graph | 🔜 Planned |
| CUDA Acceleration         | 🔜 Planned |
| RL Adapter                | 🔜 Planned |
| Unity Runtime Integration | 🔜 Planned |

> Please read our [Roadmap Document](ROADMAP.md) for more information.

---

## Contributing

We welcome all contributions to SushiAI.

- Fork the repo and create your feature branch.
- Follow C++ style used in the core.
- Write tests if applicable.
- Submit a pull request.

> Please read our [Contributing Guidelines](CONTRIBUTING.md) before contributing.

---

## License

SushiAI is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author & Contact

* **Author:** Mustafa Garip
* **E-Mail:** [mustafagarip@sushisystems.io](mailto:mustafagarip@sushisystems.io)
* **GitHub:** [https://github.com/sushimg/SushiAI](https://github.com/sushimg)
* **LinkedIn:** [https://www.linkedin.com/in/mustafgarip](https://www.linkedin.com/in/mustafgarip)

>*This project is built to form the AI infrastructure of Sushi Systems and to support ongoing research and development efforts. If you’ve got any questions or just want to share your thoughts, feel free to reach out through the links in the License and Contact section. I’ll get back to you as soon as I can. Thanks for stopping by and take care! ∼ Mustafa Garip*
