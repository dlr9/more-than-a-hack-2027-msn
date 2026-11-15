"""Verify trusted local research artifacts; optionally replay every IDSSE fold."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_hashes(registry):
    for item in registry['artifacts']:
        path = Path(item['path'])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'missing or linked artifact: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'artifact changed: {path}')
    print(f"Verified {len(registry['artifacts'])} saved model/calibration hashes", flush=True)


def replay():
    import joblib
    import numpy as np
    from solution.modeling import load_prepared

    root = Path('local_artifacts/r6')
    for match in ('J03WPY', 'J03WOH', 'J03WQQ'):
        for variant in ('M1', 'M2_dropout', 'M3'):
            folder = root / ('missing-stability-v1' if variant == 'M3' else 'idsse-m1-v1')
            stem = match if variant == 'M3' else f'{match}-{variant}'
            model = joblib.load(folder / f'{stem}.joblib')
            for domain in ('natural', 'dropout'):
                batch = load_prepared(root / 'prepared' / match / f'{domain}.npz')
                with np.load(folder / f'{stem}-{domain}.npz', allow_pickle=False) as saved:
                    if not np.array_equal(model.predict(batch), saved['decoded']):
                        raise ValueError(f'decoded replay differs: {stem} {domain}')
                    if variant != 'M3':
                        alive, home = model.probabilities(batch)
                        if not (np.array_equal(alive, saved['p_alive']) and
                                np.array_equal(home, saved['p_home'])):
                            raise ValueError(f'probability replay differs: {stem} {domain}')
            print(f'Exact natural/dropout replay: {stem}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Also requires prepared caches and saved predictions')
    args = parser.parse_args()
    registry = json.loads(Path('docs/R6_MODEL_REGISTRY.json').read_text())
    verify_hashes(registry)
    if args.replay:
        replay()


if __name__ == '__main__':
    main()
