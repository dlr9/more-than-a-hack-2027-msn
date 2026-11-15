"""Reproducible possession inference and official evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=r'F:\non_video_dataset')
    p.add_argument('--skip-extract', action='store_true')
    p.add_argument('--model', choices=['auto','v1','v2'], default='auto')
    p.add_argument('--config', type=Path, default=root/'models/fusion_v2_config.json')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    os.environ['MTAH_DATA_DIR'] = a.data_dir
    sys.path[:0] = [str(root/'models'), str(root/'starter'), str(root/'evaluation')]
    model = a.model
    if model == 'auto':
        manifest = root/'models/default_model.json'
        model = json.loads(manifest.read_text())['model'] if manifest.exists() else 'v1'
    from baseline import write_submission
    import possession_metrics as pm
    if model == 'v2':
        from fusion_v2 import features, predict, DEFAULT
        config = dict(DEFAULT, **json.loads(a.config.read_text()))
        labels = predict(features(a.data_dir, root/'features_v2.npz'), config)
    else:
        from extract_features import extract_features
        from possession_detector import build_advanced_model
        import games
        import numpy as np
        cache = root/'extracted_features.npz'
        # Legacy cache has no provenance: regenerate to respect changed data paths.
        raw, meta = games.require(['atd_raw','atd_meta'], data_dir=a.data_dir)
        np.savez_compressed(cache, **extract_features(raw, meta))
        labels = build_advanced_model(cache, data_dir=a.data_dir)
    output = a.output or root/f'evaluation/results/model_fusion_{model}.csv'
    write_submission(str(output), labels)
    error = pm.validate_submission(pm.read_label_csv(str(output)),199319)
    if error: raise ValueError(error)
    subprocess.run([sys.executable,str(root/'evaluation/run_evaluation.py'),
                    '--results',str(output.resolve().parent),'--gt','public','--bootstrap','1000'],check=True)

if __name__ == '__main__': main()
