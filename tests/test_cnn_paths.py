"""CNN default outputs must stay in their checkout when launched elsewhere."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CNNPathTests(unittest.TestCase):
    def test_defaults_and_explicit_overrides_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        code = '''
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'cnn'))
sys.path.insert(0, str(root / 'cnn/external'))
from inference import InferenceConfig
from create_sequences_whistles import SequenceConfig
from utils.config import TrainConfig
from inference_conf import DEFAULT_OUTPUT_ROOT
config = TrainConfig()
assert Path(config.models_dir) == root / 'cnn/runs/models'
assert Path(config.figs_dir) == root / 'cnn/runs/figures'
assert Path(config.reports_dir) == root / 'cnn/runs/reports'
assert config.spectrogram_cache_dir == root / 'cnn/runs/spectrogram_cache'
assert InferenceConfig().output_dir == root / 'cnn/runs/inference'
assert DEFAULT_OUTPUT_ROOT == root / 'cnn/runs/external_inference'
assert SequenceConfig(input_source='org/repo', source='hf').resolved_output_csv == root / 'cnn/runs/sequences/org__repo/whistle_sequences.csv'
assert TrainConfig(models_dir='relative/models').models_dir == 'relative/models'
assert InferenceConfig(output_dir=Path('relative/predictions')).output_dir == Path('relative/predictions')
assert InferenceConfig(checkpoint_path=Path('~/model.pt')).checkpoint_path == Path.home() / 'model.pt'
assert Path(TrainConfig(models_dir='~/models').models_dir) == Path.home() / 'models'
'''
        with tempfile.TemporaryDirectory(prefix='cnn-paths-') as folder:
            env = dict(os.environ, MPLCONFIGDIR=folder)
            result = subprocess.run([sys.executable, '-c', code, str(root)], cwd=folder,
                                    env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
