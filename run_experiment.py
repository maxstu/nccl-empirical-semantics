from abc import ABC, abstractmethod
from collections import Counter
from compiler import compile_dsl
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from protobuf import experiment_pb2
from tqdm import tqdm
from typing import Union
import argparse
import itertools
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import yaml
import zmq

def on_exit(finalizer_func):
    """
    A decorator that ensures a finalizer function is called 
    when the decorated method exits (via return, break, or exception).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            finally:
                finalizer_func(self) 
        return wrapper
    return decorator

class JobStatus(Enum):
    READY = "R"
    PENDING = "PD"
    CONFIGURING = "CF"
    COMPLETING = "CG"
    CANCELLED = "CA"
    MISSING = "?"

@dataclass(frozen=True)
class ExitStatus(ABC):
    @abstractmethod
    def message(self) -> str:
        return "Not implemented for base class."

@dataclass(frozen=True)
class ExitSuccess(ExitStatus):
    def message(self) -> str:
        return "Experiment finished successfully."

@dataclass(frozen=True)
class ExitTimeout(ExitStatus):
    seconds: int
    def message(self) -> str:
        return f"Job was timed out after {self.seconds} seconds."

@dataclass(frozen=True)
class ExitMissing(ExitStatus):
    def message(self) -> str:
        return "Job disappeared from queue (cancelled by user or slurm issue)."

@dataclass(frozen=True)
class ExitError(ExitStatus):
    reason: str
    def message(self) -> str:
        return f"Got an unexpected error (likely programmer error): {reason}"

@dataclass
class JobResult():
    status: ExitStatus
    results: dict
    
class BadExperimentConfig(Exception):
    pass

class UnexpectedMessageException(Exception):
    pass 

@dataclass
class Results():
    progress: int
    observations: dict
    errors: list

class SlurmJob:
    def __init__(self, dispatch_script, output_dir, progress, monitor_address, attempt):
        self.dispatch_script = dispatch_script
        self.output_dir = output_dir
        self.progress = progress
        self.monitor_address = monitor_address
        self.attempt = attempt
        self.job_id = None
        self.successful = False
        self.log = None

    def __enter__(self):
        self.log = Path(self.output_dir) / f"attempt_{self.attempt}.log"

        command = subprocess.run(
            ["sbatch", 
             f"--output={self.log}", 
             self.dispatch_script, str(self.output_dir), 
             str(self.progress), self.monitor_address], 
            capture_output=True, text=True
        )

        if command.returncode != 0:
            raise RuntimeError(f"Error submitting job: {command.stderr}")
        
        job_id_match = re.search(r'\d+', command.stdout)
        if not job_id_match:
            raise RuntimeError(f"Could not parse Job ID from: {command.stdout}")

        self.job_id = job_id_match.group()
        print(f"Job {self.job_id} submitted successfully. Logging to {self.log}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.job_id and not self.successful:
            print(f"Cancelling SLURM job {self.job_id}")
            subprocess.run(["scancel", self.job_id])

    def success(self):
        self.successful = True

class ExperimentMonitor():
    context = None
    socket = None
    poller = None
    results = None
    progress_bar = None

    @staticmethod
    def _state_to_hashable(state_pb) -> tuple:
        ranks = []
        for rank in state_pb.ranks:
            buffers = []
            for buf in rank.buffers:
                elements = []
                for elem in buf.elements:
                    count_val = elem.count if elem.HasField('count') else None
                    elements.append((elem.value, count_val))
                buffers.append(tuple(elements))
            ranks.append(tuple(buffers))
        return tuple(ranks)
    
    def __init__(self) -> None:
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:0")
        self.poller.register(self.socket, zmq.POLLIN)
        self.results = Results(0, Counter(), [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.close()
        self.socket.close()
        self.context.term()

    def get_socket_address(self) -> str:
        endpoint = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')
        bound_port = endpoint.rsplit(':', 1)[-1]
        host_ip = socket.gethostbyname(socket.gethostname())
        return f"tcp://{host_ip}:{bound_port}"

    def job_status(self, job_id) -> JobStatus:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%t'], 
            capture_output=True, text=True
        )
        status_str = result.stdout.strip()
        
        if not status_str:
            return JobStatus.MISSING
        
        try:
            return JobStatus(status_str)
        except ValueError:
            return JobStatus.MISSING

    def render_bar(current, total, length=40):
        # Calculate mechanics
        fraction = current / total
        arrow = int(fraction * length) * '█'
        padding = (length - len(arrow)) * '-'
        percent = f"{100 * fraction:.1f}%"
        
        # Render to terminal
        sys.stdout.write(f"\r|{arrow}{padding}| {current}/{total} ({percent})")
        sys.stdout.flush()

    def handshake(self, job_id: int) -> bool:
        print("Waiting for CUDA job handshake... ")

        check_freq = 1 # second
        missing_limit = 60 # seconds
        missing_count = 0

        # Wait for job to appear in the queue, which may normally be slightly delayed
        while self.job_status(job_id) == JobStatus.MISSING and \
              missing_count * check_freq < missing_limit:
            missing_count += 1
            time.sleep(check_freq)

        # Wait for slurm to start running the job
        while self.job_status(job_id) in (JobStatus.PENDING, JobStatus.CONFIGURING):
            time.sleep(check_freq)

        socks = dict(self.poller.poll(60000)) 
        if self.socket in socks:
            msg = self.socket.recv_string()
            if msg == "READY":
                self.socket.send_string("OK")
                print("Handshake complete.")
                return True
        
        # If we drop down here, it either timed out or got a bad message
        if self.job_status(job_id) == JobStatus.MISSING:
            print("Failed (Job aborted or went missing).")
        else:
            print("Failed (Timeout waiting for READY).")
            
        return False

    def watch(self, job_id: str, timeout: int, iters: int, quiet: bool = False) -> JobResult:
        if not self.handshake(job_id):
            return None # Handshake failed, assume job is dead
        
        print(f"Monitoring Job {job_id}...")
        if not quiet:
            self.progress_bar = tqdm(total=iters, desc=f"Experiment Progress", unit="iter")
            self.progress_bar.update(self.results.progress)
        else:
            print(f"Starting at iteration {self.results.progress}/{iters}")

        last_update_time = time.monotonic_ns()
        running = False
        warmup_iters = 3
        pre_time = time.monotonic_ns()

        while True:
            socks = dict(self.poller.poll(1000 * timeout))
            
            if self.socket in socks:
                raw_msg = self.socket.recv()
                message = experiment_pb2.Message()
                message.ParseFromString(raw_msg)

                if message.HasField("result"):
                    for obs in message.result.observations:
                        state_tuple = self._state_to_hashable(obs.state)
                        self.results.observations[state_tuple] += obs.count
                    self.results.errors.extend(message.result.errors)

                if message.HasField("update"):
                    new_progress = message.update.iteration
                    delta = new_progress - self.results.progress
                    if delta > 0:
                        if self.progress_bar:
                            self.progress_bar.update(delta)
                        elif quiet and new_progress % 100 == 0:
                            print(f"Progress: {new_progress}/{iters}")

                    self.results.progress = new_progress
                    if not running and message.update.status == experiment_pb2.Update.Status.RUNNING:
                        running = True
                    if message.update.status == experiment_pb2.Update.Status.COMPLETE:
                        return JobResult(ExitSuccess(), self.results)
                    continue # check for next update
                else:
                    return JobResult(ExitError(f"Unexpected message from CUDA job: {message}"), self.results)
                
                # No message received from CUDA within timeout window
                if self.job_status(job_id) == JobStatus.MISSING:
                    # External issue / job killed by user
                    return JobResult(ExitMissing(), self.results)

                
            if not running and warmup_iters > 0:
                # We didn't receive a message within the timout,
                # but it might just be because the pipline isn't warm enough yet.
                # Allowed to retry a few times
                warmup_iters -= 1
                continue

            post_time = time.monotonic_ns()
            delta_sec = (post_time - pre_time) // (10**9)
            return JobResult(ExitTimeout(delta_sec), self.results)

class Experiment():
    name = None
    config = None
    source_code = None
    output_dir = None
    template = Environment(loader=FileSystemLoader('.')).get_template("template.cu.j2")
    dispatch_script = "experiment_job.sh"
    iters = None
    results = None

    def source_path(self) -> str:
        return Path(self.output_dir)/"experiment.cu"

    def __init__(self, config_file: str, iters: int) -> None:
        # Load the config from specified file
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Could not find experiment file: {config_file}")

        with open(config_file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise BadExperimentConfig(f"Not a valid yaml file: {config_file}")

        self.name = Path(config_file).stem
        self.results = Results(0, Counter(), [])

        # Create CUDA source code
        program = self.config['program']
        required_gpus = self.config['required_gpus']
        buffers_per_gpu = self.config['buffers_per_gpu']
        buffer_size = self.config['buffer_size']
        
        if 'streams_per_gpu' not in self.config:
             raise ValueError(f"No stream count specified for {self.name} (missing 'streams_per_gpu' in YAML).")
        streams_per_gpu = self.config['streams_per_gpu']

        initial_values_raw = self.config['initial_values']
        initial_values = str(initial_values_raw).replace("[", "{").replace("]", "}")
        
        # Safely attempt to get config_iters
        config_iters = self.config.get('iterations')

        # Prioritize CLI `iters`, then YAML `config_iters`
        if iters is not None:
            self.iters = iters
        elif config_iters is not None:
            self.iters = config_iters
        else:
            raise ValueError(f"No iteration count specified for {self.name} (neither in CLI nor YAML).")
        
        # Compile the litmus and fill the template
        litmus_test = compile_dsl(program, streams_per_gpu)
        
        communicators = self.config.get('communicators')
        if not communicators:
             raise ValueError(f"No communicator layout specified for {self.name} (missing 'communicators' in YAML).")
        
        num_cliques = len(communicators)

        self.source_code = self.template.render(
            program=litmus_test, 
            iterations=self.iters, 
            required_gpus=required_gpus, 
            buffer_size=buffer_size, 
            buffers_per_gpu=buffers_per_gpu, 
            streams_per_gpu=streams_per_gpu, 
            initial_values=initial_values,
            communicators=communicators,
            num_cliques=num_cliques
        ) 

    @on_exit(lambda self: self.save_final_results())
    def run(self, output_dir: str, max_retry: int, quiet: bool = False, resume: bool = False) -> None:
        self.output_dir = Path(output_dir)/self.name

        # Resume logic
        if resume and (self.output_dir / "final_results.yaml").exists():
            print(f"Skipping {self.name} (already completed).")
            return

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for item in self.output_dir.iterdir():
            if item.name == "final_results.yaml":
                continue # Keep results if we are somehow here, though resume should have caught it
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Copy protobuf files to the work directory
        shutil.copy("protobuf/experiment.pb.cc", self.output_dir)
        shutil.copy("protobuf/experiment.pb.h", self.output_dir)

        with self.source_path().open(mode='w') as f:
            f.write(self.source_code)

        print(f"Running {self.name}")

        for attempt in range(max_retry):
            print(f"Attempt {attempt+1} / {max_retry}")
            results = self.run_attempt(attempt, quiet)

            if not results:
                continue

            self.results.observations.update(results.observations)
            self.results.errors.extend(results.errors)
            self.results.progress = results.progress

            print(f"Cumulative progress: {self.results.progress}/{self.iters}")

            if self.results.progress >= self.iters:
                print(f"Experiment {self.name} completed successfully.")
                return

    def run_attempt(self, attempt: int, quiet: bool = False) -> dict | None:
        try:
            with ExperimentMonitor() as monitor, SlurmJob(
                    self.dispatch_script,
                    self.output_dir,
                    self.results.progress,
                    monitor.get_socket_address(),
                    attempt) as job:
                
                job_result = monitor.watch(job.job_id, 10, self.iters, quiet)
                
                if job_result:
                    print(job_result.status.message())
                    if isinstance(job_result.status, ExitSuccess):
                        job.success()
                    return job_result.results
                    
        except Exception as e:
            print(f"Attempt failed with exception: {e}")
        
        return None

    class _Conformance(Enum):
        ACCEPTED = "yes"
        REJECTED = "no"
        SPEC_MALFORMED = "malformed conformance spec in yaml"
        SPEC_MISSING = "ignored (no spec given)"
        WILDCARD = "ignored (wildcard)"

    def _check_conformance(self, state_tuple: tuple, acceptable_results: list | None) -> _Conformance:
        if not acceptable_results:
            return self._Conformance.SPEC_MISSING
        
        for spec in acceptable_results:
            if spec == "*":
                return self._Conformance.WILDCARD
            
            if len(state_tuple) != len(spec):
                return self._Conformance.SPEC_MALFORMED
                
            for r in range(len(state_tuple)):
                if spec[r] == "*":
                    continue # spec says the data in this rank is irrelevant
                
                if isinstance(spec[r], list) and len(state_tuple[r]) != len(spec[r]):
                    return self._Conformance.SPEC_MALFORMED
                elif not isinstance(spec[r], list) and spec[r] != "*":
                     return self._Conformance.SPEC_MALFORMED
                    
                for b in range(len(state_tuple[r])):
                    if spec[r][b] == "*":
                        continue # spec says the data in this buffer is irrelevant

                    match spec[r][b]:
                        case str() if spec[r][b] == "*":
                            continue # wildcard
                        case float(val) | int(val):
                            allowed_vals = {float(val)}
                        case str(val):
                            try:
                                allowed_vals = {float(val)}
                            except ValueError:
                                return self._Conformance.SPEC_MALFORMED
                        case list(buf):
                            try:
                                allowed_vals = {float(x) for x in buf}
                            except ValueError:
                                return self._Conformance.SPEC_MALFORMED
                        case _:
                            return self._Conformance.SPEC_MALFORMED

                    # state_tuple[r][b] is a tuple of (value, count) 
                    buffer_vals = set(elem[0] for elem in state_tuple[r][b])
                    
                    if not buffer_vals.issubset(allowed_vals):
                        return self._Conformance.REJECTED
                    
        return self._Conformance.ACCEPTED

    def _format_state(self, state_tuple: tuple) -> str:
        lines = ["{"]
        for r_idx, r in enumerate(state_tuple):
            buf_strs = []
            for b in r:
                el_strs = []
                for val, count in b:
                    # If count is None, it was a Unique element. Otherwise, Run/Counter.
                    if count is None:
                        el_strs.append(f"{val}")
                    else:
                        el_strs.append(f"({val}: {count})")
                buf_strs.append("[" + ", ".join(el_strs) + "]")
            
            rank_str = f"    Rank {r_idx}: [" + ", ".join(buf_strs) + "]"
            lines.append(rank_str)
        lines.append("}")
        return "\n".join(lines)

    def save_final_results(self):
        def multiline_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        yaml.add_representer(str, multiline_presenter)

        acceptable_results = self.config.get('acceptable_results')
        expected_deadlock = self.config.get('expected_deadlock', False)
        expected_ub = self.config.get('expected_undefined_behaviour', False)

        formatted_observations = []
        for state_tuple, count in self.results.observations.most_common():
            obs_dict = {
                "count": count, 
                "state": self._format_state(state_tuple)
            } 
            
            # Check if the result was expected
            obs_dict["conforms"] = self._check_conformance(state_tuple, acceptable_results).value
                
            formatted_observations.append(obs_dict)

        # Overall experiment conformance
        if expected_deadlock:
            if self.results.progress == 0:
                experiment_conforms = "yes (deadlocked as expected)"
            else:
                experiment_conforms = "no (completed but deadlock was expected)"
        elif expected_ub:
            experiment_conforms = "yes (undefined behaviour as expected)"
        else:
            if self.results.progress < self.iters:
                experiment_conforms = "no (deadlocked or failed prematurely)"
            elif not acceptable_results:
                experiment_conforms = "ignored (no spec given)"
            else:
                # Check if all observations conform
                all_obs_conform = all(obs["conforms"] == "yes" for obs in formatted_observations)
                if all_obs_conform:
                    experiment_conforms = "yes"
                else:
                    experiment_conforms = "no (some observations do not conform)"

        final_data = {
            "name": self.name,
            "total_iterations": self.results.progress,
            "expected_deadlock": expected_deadlock,
            "expected_undefined_behaviour": expected_ub,
            "experiment_conforms": experiment_conforms,
            "observations": formatted_observations,
            "errors": self.results.errors
        }
        
        result_file = Path(self.output_dir) / "final_results.yaml"
        
        with open(result_file, 'w') as f:
            yaml.dump(final_data, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NCCL experiments and litmus tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run a single experiment:
    python3 run_experiment.py experiments/simple.yaml

  Run all experiments in the experiments/ directory:
    python3 run_experiment.py

  Run a batch of experiments using a configuration file:
    python3 run_experiment.py --config batch_run.yaml

  Run in the background with quiet mode and resume capability:
    nohup python3 run_experiment.py --config batch_run.yaml -q -r > session.log 2>&1 &

Configuration File Format (YAML):
  output: "./results"   # Optional: Output directory
  iters: 1000           # Optional: Number of iterations per test
  quiet: true           # Optional: Suppress interactive progress bars
  resume: true          # Optional: Skip already finished tests
  tests:                # Optional: List of specific tests to run
    - experiments/simple.yaml
    - experiments/p2p_causality.yaml
"""
    )

    parser.add_argument("path", nargs="?", default=None, help="Path to a single experiment YAML file. If omitted, all tests in experiments/ are run.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory to store results (default: ./output)")
    parser.add_argument("-i", "--iters", type=int, default=None, help="Number of iterations for each experiment, overriding values in experiment YAMLs.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a batch configuration YAML file.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress interactive progress bars (recommended for background runs).")
    parser.add_argument("-r", "--resume", action="store_true", help="Skip experiments that already have a 'final_results.yaml' in the output directory.")

    args = parser.parse_args()

    # Default values
    output_dir = "./output"
    iterations = None
    tests_to_run = []
    quiet = args.quiet
    resume = args.resume

    if args.config:
        if not os.path.isfile(args.config):
            print(f"Error: Configuration file {args.config} not found.")
            sys.exit(1)
        with open(args.config, 'r') as f:
            try:
                run_config = yaml.safe_load(f)
                if not isinstance(run_config, dict):
                    raise ValueError("Config file must be a dictionary")
                
                # Check for conflicts
                conflicts = []
                if args.path and 'tests' in run_config:
                    conflicts.append("test path(s)")
                if args.output and 'output' in run_config:
                    conflicts.append("output directory")
                if args.iters and 'iters' in run_config:
                    conflicts.append("iteration count")
                if args.quiet and 'quiet' in run_config:
                    conflicts.append("quiet mode")
                if args.resume and 'resume' in run_config:
                    conflicts.append("resume mode")
                
                if conflicts:
                    print(f"Error: Conflict detected! The following options were specified both on the command line and in the config file: {', '.join(conflicts)}.")
                    sys.exit(1)

                output_dir = run_config.get('output', output_dir)
                iterations = run_config.get('iters', iterations)
                tests_to_run = run_config.get('tests', [])
                quiet = run_config.get('quiet', quiet)
                resume = run_config.get('resume', resume)
            except (yaml.YAMLError, ValueError) as exc:
                print(f"Error parsing configuration file: {exc}")
                sys.exit(1)

    # CLI values override/set if not already set by config (and no conflict)
    if args.output:
        output_dir = args.output
    if args.iters:
        iterations = args.iters
    if args.path:
        tests_to_run = [args.path]

    # If no tests specified, run all in experiments/
    if not tests_to_run:
        for yaml_file in sorted(os.listdir("experiments")):
            if yaml_file.endswith(".yaml"):
                tests_to_run.append(os.path.join("experiments", yaml_file))

    # Run the experiments
    for test_path in tests_to_run:
        print(f"--- Processing {test_path} ---")
        try:
            experiment = Experiment(test_path, iterations)
            experiment.run(output_dir, max_retry=3, quiet=quiet, resume=resume)
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {test_path}: {e}")

    print("All done!")
