"""Post-inference short-spell diagnostics; never used in model features."""
import argparse
import json
import sys
import numpy as np
from fusion_v4 import ROOT
sys.path.insert(0, str(ROOT/'evaluation'))
import possession_metrics as pm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='v4_refine_reports')
    args = parser.parse_args()
    if args.study not in ('v4_reports', 'v4_refine_reports'):
        parser.error('Unsupported study')
    out = ROOT/'evaluation'/args.study
    table = pm.read_label_csv(str(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    gt = pm.label_codes(table)
    regions = pm.valid_runs(gt)
    truth = pm.possession_spells(gt, regions)
    short = np.array([b-a < 125 for a,b,k in truth])
    with np.load(ROOT/'features_v2.npz') as f:
        visible = np.isfinite(f['ball']).all(axis=1)
    dead = (table['label'].to_numpy() == 'D') & visible
    result = {}
    baseline_flags = None
    paths = [('v2', ROOT/'evaluation/results/model_fusion_v2.csv'),
             ('v3', ROOT/'evaluation/results/model_fusion_v3.csv'),
             ('v4', out/'candidate.csv')]
    for name, path in paths:
        if not path.exists():
            result[name] = {'unavailable': True}
            continue
        prediction = pm.read_label_csv(str(path))
        spells = pm.possession_spells(pm.label_codes(prediction), regions)
        flags = pm.matched_gt_flags(pm.spell_overlaps(truth, spells), len(truth), .5)
        if name == 'v2':
            baseline_flags = flags
        result[name] = dict(gt_short_spells=int(short.sum()),
            matched_short_spells=int((short & flags).sum()),
            v2_matched_short_lost=int((short & baseline_flags & ~flags).sum()) if baseline_flags is not None else None,
            visible_dead_recall=float(np.mean(prediction['label'].to_numpy()[dead] == 'D')),
            predicted_spells=len(spells), matched_gt_spells=int(flags.sum()))
    (out/'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
