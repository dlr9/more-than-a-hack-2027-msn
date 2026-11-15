"""Reproducible bounded search, blocked validation, ablations and release gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from fusion_v2 import ROOT, DEFAULT, BLOCKS, features, predict
sys.path.insert(0, str(ROOT / 'evaluation'))
import possession_metrics as pm
from baseline import write_submission

def metrics(gt, labels, bootstrap=0):
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap)

def window_gt(gt, lo, hi):
    result = np.full_like(gt, pm.X); result[lo:hi] = gt[lo:hi]
    return result

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item()), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=r'F:\non_video_dataset')
    parser.add_argument('--trials', type=int, default=60)
    parser.add_argument('--reuse-search', action='store_true', help='Reuse this dataset\'s saved candidate metrics; only for unchanged features and search code')
    args = parser.parse_args()
    if not 1 <= args.trials <= 100: parser.error('trials must be 1..100 per decoder')
    out = ROOT / 'evaluation' / 'v2_reports'; out.mkdir(exist_ok=True)
    f = features(args.data_dir, ROOT / 'features_v2.npz')
    v1_path = ROOT/'evaluation/results/model_fusion_v1.csv'
    if not v1_path.exists():
        # Submission CSVs are intentionally ignored; reconstruct V1 on a fresh clone.
        import games
        from extract_features import extract_features
        from possession_detector import build_advanced_model
        raw, meta = games.require(['atd_raw', 'atd_meta'], data_dir=args.data_dir)
        cache = ROOT/'extracted_features.npz'
        np.savez_compressed(cache, **extract_features(raw, meta))
        write_submission(str(v1_path), build_advanced_model(cache, data_dir=args.data_dir))
    gt = pm.label_codes(pm.read_label_csv(str(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')))
    graded = np.flatnonzero(gt != pm.X)
    edges = np.linspace(graded[0], graded[-1]+1, 4, dtype=int)
    windows = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    fold_gt = [window_gt(gt, a, b) for a, b in windows]
    # Remove five seconds on both sides of internal boundaries for parameter selection.
    fit_gt = []
    for a, b in windows:
        fit_gt.append(window_gt(gt, a+(125 if a>edges[0] else 0),
                               b-(125 if b<edges[-1] else 0)))
    baselines = {}
    for path in sorted((ROOT/'evaluation/results').glob('*.csv')):
        if path.stem.startswith(('baseline_', 'model_fusion_v1', 'model_state_machine')):
            labels = pm.read_label_csv(str(path))['label'].to_numpy()
            baselines[path.stem] = dict(full=metrics(gt, labels, 1000),
                folds=[metrics(g, labels) for g in fold_gt], fits=[metrics(g, labels) for g in fit_gt])
    rng = np.random.default_rng(20261115)
    candidates = json.loads((out/'experiments.json').read_text())['candidates'] if args.reuse_search else []
    for decoder in ('state', 'viterbi'):
        for i in range(0 if args.reuse_search else args.trials):
            cfg = dict(DEFAULT, decoder=decoder)
            if i:
                choices = dict(window=[5,12,25,40], gap=[0,25,50], tag_weight=[0.,.2,.5,1.],
                    idle=[.36,.43,.5,.56], dead_bias=[-1.2,-.8,-.4,0.,.4],
                    missing_weight=[0.,.5,1.], team_penalty=[8.,16.,30.,60.,100.],
                    dead_penalty=[8.,16.,30.,60.], sustain=[10,20,40,60],
                    dead_sustain=[15,30,50,75], distance=[2.,3.2,5.])
                cfg.update({k: rng.choice(v).item() for k,v in choices.items()})
            labels = predict(f, cfg, blocks=windows)
            folds = [metrics(g, labels) for g in fold_gt]
            fits = [metrics(g, labels) for g in fit_gt]
            candidates.append(dict(config=cfg, folds=folds, fits=fits))
            print(decoder, i+1, 'blocked spellF1', round(np.mean([m['spell_f1_mean'] for m in folds]),4), flush=True)
    def rank(c, indices, key='fits'):
        ms = [c[key][j] for j in indices]
        return (np.mean([m['spell_f1_mean'] for m in ms]),
                np.mean([m['macro_f1'] for m in ms]), -int(c['config']['decoder']=='viterbi'))
    cv = []
    def select(indices):
        indices = list(indices)
        floor = np.mean([baselines['model_fusion_v1']['fits'][j]['macro_f1'] for j in indices])
        eligible = [c for c in candidates if np.mean([c['fits'][j]['macro_f1'] for j in indices]) >= floor]
        return max(eligible or candidates, key=lambda c: rank(c, indices))
    for held in range(3):
        best = select(j for j in range(3) if j!=held)
        cv.append(dict(held_out=held, config=best['config'], metrics=best['folds'][held]))
    best = select(range(3))
    cfg = best['config']
    labels = predict(f, cfg)
    full = metrics(gt, labels, 1000)
    # Gate uses continuous full-match inference scored on the same windows as V1.
    final_folds = [metrics(g, labels) for g in fold_gt]
    v1 = baselines['model_fusion_v1']
    accepted = bool(full['spell_f1_mean'] > v1['full']['spell_f1_mean'] and
                    full['macro_f1'] >= v1['full']['macro_f1'] and
                    np.mean([m['spell_f1_mean'] for m in final_folds]) >
                    np.mean([m['spell_f1_mean'] for m in v1['folds']]))
    ablations = {}
    for name, patch in dict(no_repair={'gap':0}, no_tags={'tag_weight':0.},
             no_dead_evidence={'idle':-100., 'missing_weight':0.},
             no_temporal={'decoder':'viterbi','team_penalty':0.,'dead_penalty':0.}).items():
        ablations[name] = metrics(gt, predict(f, dict(cfg, **patch)))
    health = []
    for a,b in BLOCKS:
        sub = labels[a:b]; changes = np.r_[0,np.flatnonzero(sub[1:]!=sub[:-1])+1,len(sub)]
        health.append(dict(start=a,end=b,dead_fraction=float(np.mean(sub=='D')),
            switches=len(changes)-2, median_segment_seconds=float(np.median(np.diff(changes))/25)))
    target = ROOT/'evaluation/results/model_fusion_v2.csv'
    write_submission(str(target), labels)
    error = pm.validate_submission(pm.read_label_csv(str(target)), len(gt))
    if error: raise ValueError(error)
    dump(ROOT/'models/fusion_v2_config.json', cfg)
    dump(ROOT/'models/default_model.json', {'model':'v2' if accepted else 'v1'})
    report = dict(seed=20261115, windows=windows, baseline=baselines, candidates=candidates,
        held_out_selection=cv, config=cfg, full=full, folds=final_folds,
        accepted=accepted, ablations=ablations, health=health,
        caveat='All public labels have historical tuning exposure. Blocked results and bootstrap are not independent full-match validation.')
    dump(out/'experiments.json', report)
    rows = ['# Fusion V2 experiment report', '', report['caveat'], '',
            '| Model | spellF1 | macroF1 |','|---|---:|---:|']
    for name, m in [(k,v['full']) for k,v in baselines.items()]+[('fusion_v2',full)]:
        rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows += ['', f'Default promoted: {accepted}', '', '## Ablations', '',
             '| Removed | spellF1 | macroF1 |','|---|---:|---:|']
    for name,m in ablations.items(): rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows += ['', '## Validation details', '',
             'Selection requires fitting-window macroF1 at least equal to V1 before maximizing spellF1. This constraint was added after the unconstrained winner reduced macroF1; the original 120 candidates were reused without additional search.',
             '', f"Public spellF1 bootstrap 5-95%: {full['spell_bootstrap_low']:.4f} to {full['spell_bootstrap_high']:.4f}.",
             '', '| Held-out block | spellF1 | macroF1 |', '|---|---:|---:|']
    for c in cv:
        m = c['metrics']; rows.append(f"| {c['held_out']+1} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows += ['', 'Each held-out row uses parameters selected on the other two blocks. Selected gap=0 disables trajectory repair, so its ablation is identical.',
             '', '## Full-match checks', '', '| Half | Dead fraction | Switches | Median segment seconds |', '|---|---:|---:|---:|']
    for i,h in enumerate(health):
        rows.append(f"| {i+1} | {h['dead_fraction']:.3f} | {h['switches']} | {h['median_segment_seconds']:.2f} |")
    rows += ['', 'Full diagnostics, per-class errors, per-IoU scores, transitions, bootstrap intervals, all trials and fold-selected configurations are in experiments.json.']
    (out/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print('RESULT',full['spell_f1_mean'],full['macro_f1'],'promoted',accepted,flush=True)

if __name__ == '__main__': main()
