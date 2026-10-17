# seam

## Owns

The mapping of a failure to a `Status`: the guards that turn an exception from SushiBLAS, or an
`Error` SushiRuntime returned, into one. The calls that create a runtime and run what was
recorded are in `execution`, which maps their failures here.

## Depends on

`failure`, public. It links `SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/seam/seam.hpp` | `Detail::SeamPhase`, `Detail::guard`, `Detail::guard_value`, `Detail::status_of_error`, `Detail::status_of_current_exception` |

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
