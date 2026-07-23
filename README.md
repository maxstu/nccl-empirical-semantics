This repository serves as the code artefact for the paper *Reducing Confusion and Broadcasting Understanding: An Empirical Study of the Semantics of NCCL*.

The litmus tests from the paper can be found in `litmus_tests/`.
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

## Litmus test format

Each litmus test is a YAML file in `litmus_tests/`.
The `program` key holds the test itself, written in a small DSL (described below) that `compiler/` translates into the CUDA/NCCL host code inserted into `template.cu.j2`.
The remaining keys configure the system the program runs on and how its results are judged:

| Key | Meaning |
| --- | --- |
| `name`, `description` | Free-text identification (the litmus tests reference the paper figure they correspond to). |
| `required_gpus` | Number of ranks; ranks are numbered `0 .. required_gpus-1` and rank `i` runs on CUDA device `i`. |
| `buffers_per_gpu` | Number of device buffers allocated per rank, indexed `0 .. buffers_per_gpu-1`. |
| `buffer_size` | Number of elements in every buffer. Every buffer on every rank has this size. |
| `streams_per_gpu` | Number of CUDA streams created per rank, indexed `0 .. streams_per_gpu-1`. |
| `communicators` | List of cliques. The `i`-th list is the set of ranks participating in communicator `i`. |
| `initial_values` | `initial_values[r][b]` is the value every element of buffer `b` on rank `r` is filled with before each iteration. |
| `iterations` | Default number of times to run the program (overridable on the command line or via a batch config such as `run_once.yaml`). |
| `acceptable_results` | List of acceptable final states. Each state is `[rank][buffer]`, where an entry is a single value, a list of allowed values, or `"*"` to ignore that buffer (or `"*"` in place of a rank to ignore the whole rank). |
| `expected_deadlock` | If `true`, the test conforms when the program makes no progress. |
| `expected_undefined_behaviour` | If `true`, the test is recorded as conforming regardless of outcome. |

## The litmus test DSL

Whitespace is insignificant. `//` line comments and `/* ... */` block comments are permitted anywhere.
All identifiers (ranks, buffers, communicators, streams) are non-negative integers.

### Structure

```
program   ::= thread+ | body                 // a bare body is a single thread
thread    ::= 'thread' '{' body '}'
body      ::= [ (command | group) (';' (command | group))* [';'] ]
group     ::= 'group' '{' [ command (';' command)* [';'] ] '}'
command   ::= collective | p2p
```

Each `thread` compiles to a separate host thread (`std::jthread`); all threads wait on a barrier and then issue their commands in program order.
A `group` wraps its commands in `ncclGroupStart()` / `ncclGroupEnd()`. Groups cannot be nested and cannot contain `thread` blocks.

### Commands

Every command has the shape

```
name < rank, ... > [ buffers ] ( type, size, ... )
```

The angle brackets identify *where* the command is issued: the first element is always the issuing rank, and the last two are always the communicator and stream on that rank.
The square brackets name buffers local to the issuing rank.
The parentheses give the element type, the element count, and any operation-specific arguments.

Collectives take the issuing rank, a communicator and a stream, and a source and destination buffer.
Rooted collectives additionally take the root rank; reductions additionally take a reduction operator.

```
allReduce     < rank, comm, stream > [ src, dst ] ( type, size, op )
reduceScatter < rank, comm, stream > [ src, dst ] ( type, size, op )
allGather     < rank, comm, stream > [ src, dst ] ( type, size )
allToAll      < rank, comm, stream > [ src, dst ] ( type, size )
broadcast     < rank, comm, stream > [ src, dst ] ( type, size, root )
scatter       < rank, comm, stream > [ src, dst ] ( type, size, root )
gather        < rank, comm, stream > [ src, dst ] ( type, size, root )
reduce        < rank, comm, stream > [ src, dst ] ( type, size, root, op )
```

Point-to-point commands take the issuing rank and the peer rank, followed by the communicator and stream, and a single buffer.

```
send < rank, peer, comm, stream > [ buf ] ( type, size )
recv < rank, peer, comm, stream > [ buf ] ( type, size )
```

For `send`, `peer` is the receiving rank; for `recv`, `peer` is the sending rank.

Argument domains:

- `type` is `int` or `float` (passed to NCCL as `ncclInt` / `ncclFloat`).
- `size` is either an integer element count or the keyword `buffer_size`, which expands to the test's `buffer_size`.
- `op` is `sum` or `product`.
- `src` and `dst` may be the same buffer for an in-place collective.

### Compilation

A command `name<rank, ...>[...](...)` is compiled to a call to the corresponding NCCL function (`ncclAllReduce`, `ncclBroadcast`, `ncclSend`, ...) preceded by `cudaSetDevice(rank)` when the device differs from the previous command in the same thread.
Buffer `b` on the issuing rank becomes `devBuffers[rank][b]`, and `comm` and `stream` become the issuing rank's handle for communicator `comm` and its stream `stream`.

### Example

The two-rank broadcast of Figure 4 in the paper, issued from two threads:

```yaml
required_gpus: 2
buffers_per_gpu: 1
streams_per_gpu: 1
buffer_size: 128
communicators:
  - [0, 1]
program: |
    thread {
        broadcast<0,0,0>[0,0](int, buffer_size, 0);
    }
    thread {
        broadcast<1,0,0>[0,0](int, buffer_size, 0);
    }
initial_values:
    - [ 42 ]
    - [ 0 ]
acceptable_results:
    - [
        [ [ 42 ] ],
        [ [ 42 ] ],
      ]
```

Rank 0 and rank 1 each issue an in-place broadcast on buffer 0 over communicator 0, stream 0, with rank 0 as root, so both buffers are expected to hold `42` afterwards.
