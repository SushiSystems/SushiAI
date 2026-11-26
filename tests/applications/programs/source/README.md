# tests/applications/programs/source

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `cluster_fabric.hpp` | Declares Fabric, a master and the workers a test starts over one in-process fabric. |
| `cluster_test_files.hpp` | Declares ClusterTestFiles, the training file of two replicas the cluster tests share. |
| `compile_programs_log_category.cpp` | Checks at compile time the project and the module of the programs' log category. |
| `prepared_inference_harness.hpp` | Declares the builder of a PreparedInference over the tiny_yolo fixture, through prepare_inference(). |
| `prepared_run_harness.hpp` | Declares the builder of a PreparedRun over the tiny_yolo fixture, through prepare_run(). |
| `role_hidden_dataset.hpp` | Declares a dataset that serves another dataset's batches without one of its roles. |
| `test_cluster_run.cpp` | Tests one step of `sushiai train` over a master and a worker in one process. |
| `test_cluster_training.cpp` | Tests what `sushiai train` builds for a training file with and without a cluster. |
| `test_cluster_worker_watch.cpp` | Tests that ClusterWorkerWatch keeps the WorkerWatch contract over a real cluster. |
| `test_detection_report.cpp` | Tests the batch loop, the printout and the checks every detection report shares. |
| `test_detection_reports.cpp` | Tests the anchor, anchor-free and YOLOX detection reports over a prepared inference. |
| `test_device_split_run.cpp` | Tests a training step split across a GPU and a CPU, and the run without a split. |
| `test_domains.cpp` | Tests that each linked domain adds its rows beside the library's without a clash. |
| `test_epoch_loop.cpp` | Tests run_epoch_loop, the loop the trainer run and the cluster run share. |
| `test_host_readback.cpp` | Tests that a HostReadback refreshed over a device-resident tensor returns the values written into it. |
| `test_image_input.cpp` | Tests the data block that reads one image file for `sushiai inference`. |
| `test_in_memory_worker_watch.cpp` | Tests that InMemoryWorkerWatch keeps the WorkerWatch contract and moves its own clock. |
| `test_inference_preparation.cpp` | Tests that prepare_inference() honours its two hooks and the weight file it names. |
| `test_inference_program.cpp` | Tests run_inference_file over the tiny_yolo fixture: the anchor report. |
| `test_inference_run.cpp` | Tests the report table of inference_run and the task it refuses. |
| `test_master_cluster.cpp` | Tests the master side of a cluster run: its runtime's name, its attach and its wait. |
| `test_model_export.cpp` | Tests `sushiai export` through the registries: each format's fixture comes back. |
| `test_model_state_entries.cpp` | Tests that model_state_entries lists the state records of a model file without an engine. |
| `test_profile_arguments.cpp` | Tests that read_profile_arguments takes the --profile pair out of a command line. |
| `test_profile_session.cpp` | Tests that ProfileSession refuses a bad configuration naming its file and writes the record of a run that fails. |
| `test_registries.cpp` | Tests that compose_registries fills the registries of an app with the library's rows. |
| `test_replica_fallback_warning.cpp` | Tests ReplicaFallbackWarning: one WARN per replica and cause, none for the master. |
| `test_replica_tensors.cpp` | Tests ReplicaTensors::list(), the order of the tensors of a replica's packs. |
| `test_row_feed.cpp` | Tests that a row feed writes the rows of a batch into a setup's inputs. |
| `test_run_preparation.cpp` | Tests that prepare_run() starts the run where the training file says. |
| `test_run_start.cpp` | Tests that a run starts from what its file states. |
| `test_served_kernel.cpp` | Tests make_served_kernel(), which builds the replica kernel one machine serves. |
| `test_share_runtime.cpp` | Tests how a training file's device split narrows the runtime the run opens. |
| `test_share_step.cpp` | Tests one share of a split step on a CPU runtime of its own. |
| `test_trainer_run.cpp` | Tests that register_runs sets the fallback run once and adds no row. |
| `test_training_cluster_section.cpp` | Tests read_training_cluster: a fallback to the master is refused under "deterministic". |
| `test_training_program.cpp` | Tests run_training_file over the tiny_yolo fixture: the anchor row and its checkpoint. |
| `test_training_replica_model.cpp` | Tests TrainingReplicaModel and the files every replica is built from. |
| `test_training_run.cpp` | Tests the run table of training_run: the row a task picks, the fallback, and a classification run. |
| `test_training_setup.cpp` | Tests TrainingSetup, which builds what a training file describes on one engine. |
| `test_vision_training_runs.cpp` | Tests the anchor and anchor-free run rows over a prepared run of the tiny_yolo fixture. |
| `test_weights_export.cpp` | Tests the formats a weights export writes and that a refused export writes no file. |
| `test_weights_import.cpp` | Tests the containers a weights import reads and that a failed import writes no file. |
| `test_worker_arguments.cpp` | Tests read_worker_arguments, which sorts the command line of sushiai_worker. |
| `tiny_run_files.hpp` | Declares the training file the split-step tests share over the tiny run's model. |
| `tiny_yolo_files.hpp` | Declares the tiny_yolo files and runs the program tests share. |
| `worker_watch_contract.hpp` | Declares WorkerWatchContract, the promises every WorkerWatch keeps. |
