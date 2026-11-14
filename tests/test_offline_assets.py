"""Only generated fixtures. Never inspect actual target assets or their directories."""
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from games import GAME, asset_paths
from solution.offline_assets import (preflight, strict_csv, reference_labels,
                                     isolation_guard, run_reference, file_sha256)
from solution.submission import write_submission
from starter.baseline import nearest_player_labels, smooth, team_of_each_track


class OfflineAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mtah-synthetic-offline-')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, n=180):
        root = self.root / 'synthetic-assets'
        paths = {k: Path(v) for k, v in asset_paths(data_dir=str(root)).items()}
        paths['atd_meta'].parent.mkdir(parents=True)
        paths['atd_meta'].write_text('<Root><Team id="h"><Name>synthetic-home</Name></Team>'
                                    '<Team id="a"><Name>synthetic-away</Name></Team>'
                                    '<Player teamId="h"/><Player teamId="a"/></Root>')
        rows = []
        for frame in range(n):
            ball = '0.1,0.5,0' if frame < 60 else ('0.9,0.5,0' if frame < 120 else 'NaN,NaN,NaN')
            rows.append(f'{frame+1}:0.1,0.5,0;0.9,0.5,0:{ball}\n')
        paths['atd_raw'].write_text(''.join(rows))
        return root, paths

    def test_two_videos_are_insufficient_without_opening_files(self):
        root = self.root / 'synthetic-assets'
        for source in ('tactical', 'broadcast'):
            path = Path(asset_paths(data_dir=str(root), source=source)['video'])
            path.parent.mkdir(parents=True)
            path.write_bytes(b'not a video - synthetic existence fixture')
        with patch('pathlib.Path.open', side_effect=AssertionError('must not open contents')):
            result = preflight(root)
        self.assertFalse(result['reference_ready'])
        self.assertFalse(result['contents_read'])
        self.assertEqual(len(result['assets']), 9)
        self.assertEqual(len(result['missing_required']), 2)
        self.assertIn('缺少 Tactical ATD XML/TXT', result['message'])

    def test_symlink_escape_is_rejected(self):
        root, paths = self.fixture()
        paths['atd_raw'].unlink()
        outside = self.root / 'synthetic-other.txt'
        outside.write_text('never read')
        paths['atd_raw'].symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink/path escape'):
            preflight(root)

    def test_reference_preserves_official_smooth50_inside_halves(self):
        _, paths = self.fixture()
        blocks = [(10, 80), (100, 169)]
        actual = reference_labels(paths['atd_meta'], paths['atd_raw'], 180, (105.3, 68), blocks)
        codes = nearest_player_labels(str(paths['atd_raw']), team_of_each_track(str(paths['atd_meta'])),
                                      180, (105.3, 68))
        expected = smooth(codes, 50)
        mask = np.zeros(180, dtype=bool)
        for start, stop in blocks:
            mask[start:stop+1] = True
        np.testing.assert_array_equal(actual[mask], expected[mask])
        self.assertTrue(np.all(actual[~mask] == 'X'))

    def test_malformed_timeline_fails_instead_of_silent_baseline_skip(self):
        _, paths = self.fixture(4)
        paths['atd_raw'].write_text(paths['atd_raw'].read_text().replace('2:', '1:'))
        with self.assertRaisesRegex(ValueError, 'invalid ATD timeline'):
            reference_labels(paths['atd_meta'], paths['atd_raw'], 4, (105.3, 68), [(0, 3)])

    def test_strict_csv_rejects_rows_labels_and_half_leakage(self):
        file = self.root / 'synthetic.csv'
        write_submission(file, list('XHADX'))
        self.assertTrue(strict_csv(file, 5, [(1, 3)])['format_valid'])
        for content in ('frame,label\n0,H\n1,A\n', 'frame,label\n0,X\n2,H\n',
                        'frame,label\n0,X\n1,Z\n', 'frame,label\n0,X\n'):
            file.write_text(content)
            with self.assertRaises(ValueError):
                strict_csv(file, 2, [(1, 1)])

    def test_cli_isolation_refuses_direct_execution_and_workspace(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'wrapper'):
                isolation_guard(self.root)
        env = {'MTAH_OFFLINE_SANDBOX': 'macos-deny-network', 'CODEX_TEST': '1'}
        with patch.dict(os.environ, env, clear=True), patch('platform.system', return_value='Darwin'):
            with self.assertRaisesRegex(ValueError, 'Codex environment'):
                isolation_guard(self.root)
        env.pop('CODEX_TEST')
        env['MTAH_FORBIDDEN_WORKSPACE'] = str(Path(__file__).resolve().parents[1])
        with patch.dict(os.environ, env, clear=True), patch('platform.system', return_value='Darwin'):
            with self.assertRaisesRegex(ValueError, 'outside Codex workspaces'):
                isolation_guard(self.root)

    def test_synthetic_run_receipt_and_no_overwrite(self):
        root, _ = self.fixture(180)
        out = self.root / 'synthetic-output'
        game = {'n_video_frames': 180, 'pitch': (105.3, 68)}
        with patch('solution.offline_assets.GAMES', {GAME: game}), patch(
                'solution.offline_assets.half_video_blocks', return_value=[(1, 10, 80), (2, 100, 169)]):
            target = run_reference(root, out)
            receipt = json.loads((out / 'LOCAL_ONLY_receipt.json').read_text())
            self.assertEqual(receipt['output_sha256'], file_sha256(target))
            self.assertFalse(receipt['gt_read'])
            self.assertFalse(receipt['validation']['accuracy_evaluated'])
            self.assertEqual(receipt['validation']['rows'], 180)
            with self.assertRaises(FileExistsError):
                run_reference(root, out)

    def test_shell_wrappers_reject_multiple_codex_environment_names(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        for name in ('official_assets_preflight.command', 'run_offline_reference.command', 'run_offline_baseline.command'):
            with self.subTest(script=name):
                # No assets or arguments: the inherited-environment check must run first.
                result = subprocess.run(['/bin/bash', str(scripts / name)],
                    env={'PATH': '/usr/bin:/bin', 'HOME': str(self.root),
                         'CODEX_SYNTHETIC_A': '1', 'CODEX_SYNTHETIC_B': '1'},
                    input='', capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn('inherited Codex environment', result.stderr)
                self.assertNotIn('too many arguments', result.stderr)


if __name__ == '__main__':
    unittest.main()
