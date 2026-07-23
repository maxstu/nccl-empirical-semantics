This repository serves as the code artefact for the paper *Reducing Confusion and Broadcasting Understanding: An Empirical Study of the Semantics of NCCL*.

The litmus tests from the paper can be found in `experiments/`.
You can replicate the results discussed in the paper if you have access to GPU cluster which accepts jobs via SLURM and has at least three supported NVIDIA GPUs.
To exactly replicate the environment used in the paper, you should use three A100s running on a single node.

To run the tests, first ensure `make`, `conda` and CUDA 12.9.0 are available.
Depending on the configuration of your cluster,
you may need to tell the test runner how to initialise CUDA and conda:
if so, please edit `experiment_job.sh`,
and set the variables `CONDA_PATH` and `CUDA_SOURCE_PATH` appropriately.

You can then run all of the litmus tests from the paper by invoking `make run` - this will run each test once.
 Alternatively, call `make run-many` to run each test 100,000 times to confirm the outcomes are reliable.
In either case, but especially the latter, it is recommended to use a tool that makes sure the shell stays alive in the background while the tests are running,
as this may take a long time. `tmux` works well for this.
