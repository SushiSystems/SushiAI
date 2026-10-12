# core

## Owns

The foundation the other modules build on. It owns the failure contract: `Status` carries a
`StatusCode` and a message, and `Result<T>` holds either a value or the `Status` that explains
its absence. It also owns the `Shape` value type, the `DType` and `Layout` aliases of the
SushiBLAS element types and layouts, and the `Error` exception with its fatal checks. Tensor and
arena allocation on a `SushiBLAS::Engine` lives here, as do the compile-time operation
identities in `SushiAI::Ops` and the guards that turn an exception from SushiBLAS or an `Error`
SushiRuntime returned into a `Status`.

## Depends on

The module declares no module dependency. It links two external targets as public:
`SushiRuntime::SushiRuntime` and `SushiBLAS::sushiblas`, through `EXTERNAL_PUBLIC`. It declares
no private dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/core/allocation.hpp` | `allocate_tensor`, `allocate_like`, `allocate_arena`, `view_tensor` |
| `SushiAI/core/check.hpp` | `FatalHandler`, `set_fatal_handler`, `SA_FATAL_IF`, `SA_FATAL`, `SA_ASSERT` |
| `SushiAI/core/common.hpp` | `DType`, `Layout`, `K_MAX_RANK`, `dtype_size`, `dtype_name`, `is_real_float`, `Error`, `SA_THROW_IF`, the alias `sa` |
| `SushiAI/core/ops.hpp` | `OpName`, `op`, `differentiable_op`, `gradient_barrier_op`, the identities such as `K_LEAF`, `K_MATMUL` and `K_BATCHNORM2D`, `K_ALL`, `GradientSource`, `Activation`, `K_ACTIVATIONS`, `activation_of` |
| `SushiAI/core/result.hpp` | `Result<T>`, `Result<T&>`, `Detail::status_of`, `SA_TRY_ASSIGN` |
| `SushiAI/core/seam.hpp` | `Detail::SeamPhase`, `Detail::guard`, `Detail::guard_value`, `Detail::record_and_run`, `Detail::status_of_error`, `Detail::create_runtime`, `Detail::run_recorded`, `Detail::run_step`, the alias `sr` |
| `SushiAI/core/shape.hpp` | `Shape`, `std::hash<SushiAI::Shape>` |
| `SushiAI/core/status.hpp` | `StatusCode`, `Status`, `SA_FAIL_IF`, `SA_ANNOTATE`, `SA_TRY` |
| `SushiAI/core/tensor_shape.hpp` | `shape_of` |

## Runtime errors

`Detail::status_of_error` gives each `SushiRuntime::ErrorCode` one `StatusCode` and keeps the
runtime's message.

| `ErrorCode` | `StatusCode` | What the caller met |
| --- | --- | --- |
| `AllocationFailed` | `OutOfMemory` | A device or host allocation the runtime could not satisfy, a plan build among them |
| `KernelFailed`, `DeviceLost` | `DeviceFailed` | A kernel that faulted, or a device that went away |
| `Unsupported` | `Unsupported` | A capability the runtime does not offer |
| `InvalidAccess` | `Unsupported` | A placement on a device the machine lacks, or an external handle the runtime refused |
| `InvalidState` | `Unsupported` | A run started while another graph's run is in flight on the same runtime |
| `InvalidGraph` | `InvalidDescription` | A graph the runtime cannot run as recorded: a count above capacity, buffers across device contexts, a launch past its handle, a cycle |
| `ProtocolMismatch` | `Incompatible` | A peer that speaks another version of the wire protocol |
| `CompileFailed` | `OutOfMemory` in the `Allocate` phase, `DeviceFailed` otherwise | A plan that could not be built, with no cause named |
| `DanglingResource` | none; `SA_FATAL` | A tensor freed before the step that names it, which breaks the contract of the pool that owns it |
