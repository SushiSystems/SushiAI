# tests/core/distribution

The test sources of `modules/core/distribution` kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `exchange_test_support.hpp` | Declares the master and the worker on one InProcessFabric that the exchange tests use. |
| `replica_test_support.hpp` | Declares the model and the single-machine step the distribution tests share. |
| `test_cluster_section.cpp` | Tests that ClusterSection reads the cluster section and turns it into placement rules. |
| `test_exact_sum.cpp` | Tests which types the replicas' results add as exact counts. |
| `test_exchange_targets.cpp` | Tests the checks of the tensors a gradient exchange sums into, on shapes alone. |
| `test_gradient_exchange.cpp` | Tests GradientExchange: two replicas against one machine, two pinned runs, every type. |
| `test_host_access.cpp` | Tests which tensors the host copies of a replica may address. |
| `test_in_memory_replica_model.cpp` | Tests that InMemoryReplicaModel holds the counts it was given and the rows it was fed. |
| `test_in_memory_replica_model_factory.cpp` | Tests that InMemoryReplicaModelFactory keeps its calls and returns what its inner factory returns. |
| `test_in_memory_replica_serve_observer.cpp` | Tests that InMemoryReplicaServeObserver keeps the promise of ReplicaServeObserver. |
| `test_parameter_publish.cpp` | Tests ParameterPublish: the first copy, the gather of each replay and its refusals. |
| `test_replica_draws.cpp` | Tests that a replica's seed takes the run's seed, the replica and the run index. |
| `test_replica_kernel.cpp` | Tests that one replica served on InProcessFabric returns a single-machine step's results. |
| `test_replica_model.cpp` | Tests what a ReplicaModelFactory is called with, and what its model must hold. |
| `test_replica_plan.cpp` | Tests that ReplicaPlan splits a batch into rows, micro-batches and weights. |
| `test_replica_position.cpp` | Tests ReplicaPosition: the position of a step and its read from a pack's bytes. |
| `test_replica_serve_observer.cpp` | Tests that a replica kernel tells its observer of each task, with its position. |
| `test_replica_step.cpp` | Tests that ReplicaStep writes the results of its micro-batches, and their mean. |
| `test_replica_task.cpp` | Tests the parameters a replica node carries, and what decode() refuses. |
