import unittest
from unittest.mock import MagicMock, patch
from run_experiment import Experiment, Results
from collections import Counter
from pathlib import Path

class TestHarness(unittest.TestCase):
    def setUp(self):
        # Mocking yaml load to avoid file I/O
        self.mock_config = {
            'name': 'test_experiment',
            'program': 'thread { allReduce<0,0,0>[0,1](float, 10, sum); }',
            'required_gpus': 1,
            'buffers_per_gpu': 2,
            'buffer_size': 10,
            'streams_per_gpu': 1,
            'iterations': 10,
            'initial_values': [[1.0]],
            'communicators': [[0]],
            'acceptable_results': [
                [['*']] # Placeholder
            ]
        }

    @patch('run_experiment.os.path.isfile')
    @patch('run_experiment.yaml.safe_load')
    @patch('run_experiment.open', create=True)
    @patch('run_experiment.compile_dsl')
    def test_conformance_check_basic(self, mock_compile, mock_open, mock_yaml, mock_isfile):
        mock_isfile.return_value = True
        mock_yaml.return_value = self.mock_config
        mock_compile.return_value = "void main() {}"
        
        exp = Experiment('fake_path.yaml', iters=10)
        
        # State: Rank 0, Buffer 0: [(1.0, None)] (Unique type)
        state = (
            (( (1.0, None), ),),
        )
        
        # Exact match
        spec = [[ [1.0] ]]
        self.assertEqual(exp._check_conformance(state, spec).value, "yes")
        
        # Mismatch
        spec = [[ [2.0] ]]
        self.assertEqual(exp._check_conformance(state, spec).value, "no")

    @patch('run_experiment.os.path.isfile')
    @patch('run_experiment.yaml.safe_load')
    @patch('run_experiment.open', create=True)
    @patch('run_experiment.compile_dsl')
    def test_conformance_check_wildcards(self, mock_compile, mock_open, mock_yaml, mock_isfile):
        mock_isfile.return_value = True
        mock_yaml.return_value = self.mock_config
        mock_compile.return_value = ""
        exp = Experiment('fake_path.yaml', iters=10)
        
        # State: Rank 0, Buffer 0: [(1.0, 5), (2.0, 5)], Buffer 1: [(3.0, 10)]
        state = (
            ( ((1.0, 5), (2.0, 5)), ((3.0, 10),) ),
        )
        
        # Rank wildcard
        spec = ['*']
        self.assertEqual(exp._check_conformance(state, spec).value, "ignored (wildcard)")

        # Rank wildcard (nested)
        spec = [['*']]
        self.assertEqual(exp._check_conformance(state, spec).value, "yes")
        
        # Buffer wildcard
        # One outcome, containing one rank, which has two buffer specs.
        spec = [[ ["*", [3.0]] ]]
        self.assertEqual(exp._check_conformance(state, spec).value, "yes")
        
        # Multi-value spec
        # One outcome, containing one rank, which has two buffer specs.
        spec = [[ [ [1.0, 2.0], "*" ] ]]
        self.assertEqual(exp._check_conformance(state, spec).value, "yes")

    @patch('run_experiment.os.path.isfile')
    @patch('run_experiment.yaml.safe_load')
    @patch('run_experiment.open', create=True)
    @patch('run_experiment.compile_dsl')
    @patch('run_experiment.Path.mkdir')
    def test_ub_expectation(self, mock_mkdir, mock_compile, mock_open, mock_yaml, mock_isfile):
        mock_isfile.return_value = True
        config = self.mock_config.copy()
        config['expected_undefined_behaviour'] = True
        mock_yaml.return_value = config
        mock_compile.return_value = ""
        
        exp = Experiment('fake_path.yaml', iters=10)
        exp.output_dir = Path("./fake_output")
        
        # Simulate a crash (zero progress)
        exp.results = Results(progress=0, observations=Counter(), errors=["OOM"])
        
        # Mocking save_final_results dependencies
        with patch('run_experiment.yaml.dump') as mock_dump, \
             patch('run_experiment.open', create=True):
            exp.save_final_results()
            
            # Get the data passed to yaml.dump
            final_data = mock_dump.call_args[0][0]
            self.assertEqual(final_data['experiment_conforms'], "yes (undefined behaviour as expected)")
            self.assertTrue(final_data['expected_undefined_behaviour'])

    @patch('run_experiment.os.path.isfile')
    @patch('run_experiment.yaml.safe_load')
    @patch('run_experiment.open', create=True)
    @patch('run_experiment.compile_dsl')
    @patch('run_experiment.Path.mkdir')
    def test_deadlock_expectation(self, mock_mkdir, mock_compile, mock_open, mock_yaml, mock_isfile):
        mock_isfile.return_value = True
        config = self.mock_config.copy()
        config['expected_deadlock'] = True
        mock_yaml.return_value = config
        mock_compile.return_value = ""
        
        exp = Experiment('fake_path.yaml', iters=10)
        exp.output_dir = Path("./fake_output")
        
        # Progress is 0 (deadlocked)
        exp.results = Results(progress=0, observations=Counter(), errors=[])
        
        with patch('run_experiment.yaml.dump') as mock_dump, \
             patch('run_experiment.open', create=True):
            exp.save_final_results()
            final_data = mock_dump.call_args[0][0]
            self.assertEqual(final_data['experiment_conforms'], "yes (deadlocked as expected)")

        # Progress is 10 (didn't deadlock as expected)
        exp.results.progress = 10
        with patch('run_experiment.yaml.dump') as mock_dump, \
             patch('run_experiment.open', create=True):
            exp.save_final_results()
            final_data = mock_dump.call_args[0][0]
            self.assertEqual(final_data['experiment_conforms'], "no (completed but deadlock was expected)")

if __name__ == '__main__':
    unittest.main()
