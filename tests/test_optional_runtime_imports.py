"""Exercise import boundaries in fresh processes, even on simulator-equipped hosts."""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLOCK_RUNTIME = """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'glfw', 'OpenGL', 'rclpy', 'lerobot'}:
            raise AssertionError(f'Optional runtime imported by a contract: {fullname}')

sys.meta_path.insert(0, BlockRuntime())
"""


@pytest.mark.parametrize('code', [
    """
    from experiments.iros2027.contact.play import Preview
    preview = Preview()
    preview.key(32)
    assert preview.paused.is_set()
    preview.key(32)
    assert not preview.paused.is_set()
    for key, command in [('R', 'restart'), ('N', 'next'), ('Q', 'quit')]:
        preview.key(ord(key))
        assert preview.command == command
    preview.close()
    assert 'experiments.iros2027.contact.run' not in sys.modules
    """,
    """
    import runpy
    sys.argv = ['contact-play', '--help']
    runpy.run_module('experiments.iros2027.contact.play', run_name='__main__')
    """,
    """
    import numpy as np
    import yaml
    from experiments.iros2027.i001_v2.physics import ROOT, resolution, finite_observation, probe_width
    from experiments.iros2027.i001_v2.data import current_matches
    from experiments.iros2027.i001_v2.analysis import metrics
    from pathlib import Path
    assert ROOT == Path.cwd()
    config = yaml.safe_load(Path('experiments/iros2027/i001_v2/config.yaml').read_text())
    assert resolution(config).shape == (58,)
    x = finite_observation(np.zeros(58), config, np.random.default_rng(0))
    assert len(current_matches(np.stack([x, x]), config)) == 1
    assert probe_width(2600, 2400, 70, .001, -1, .032) == .032
    assert metrics([0, 1], [.5, .5])['auroc'] == .5
    assert 'experiments.iros2027.contact.run' not in sys.modules
    """,
], ids=['preview-controls', 'preview-help', 'i001-observation-contracts'])
def test_contracts_do_not_import_optional_runtimes(code):
    result = subprocess.run(
        [sys.executable, '-c', BLOCK_RUNTIME + textwrap.dedent(code)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
