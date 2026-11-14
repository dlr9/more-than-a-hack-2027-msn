"""Publish aggregate OTHER-match evidence only, never frame arrays or target reports."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared
from solution.quality import phase_coverage
from solution.heuristic import majority_smooth
from scripts.experiment_v2 import ROOT,OUT,IDS

def main():
    result=json.loads((OUT/'experiment_results.json').read_text())
    assert len(result['rows'])==72, 'experiment has not completed all fixed folds'
    summary=json.loads((OUT/'summary.json').read_text())
    audits=[]
    for m in IDS:
        b=load_prepared(OUT/f'{m}-observed.npz')
        audits.append(phase_coverage(b,ROOT/'skillcorner'/m/f'{m}_dynamic_events.csv'))
    baselines={}
    for r in result['rows']:
        view='all' if r['variant']=='all' else 'observed'
        key=(r['heldout'],r['domain'],view)
        if key not in baselines:
            path=OUT/('metrica.npz' if r['domain']=='metrica_partial' else f"{r['heldout']}-{'dropout' if r['domain']=='dropout_stress' else view}.npz")
            b=load_prepared(path);ya=b.get('raw_y_alive',b['y_alive']);yh=b['y_home']
            labels=majority_smooth(b['nearest_codes'],round(2*float(b['fps'])))
            truth=np.full(len(ya),'X');truth[ya==0]='D';truth[(ya==1)&(yh==1)]='H';truth[(ya==1)&(yh==0)]='A'
            known=(truth!='X')&(b['period']>0)
            baselines[key]=dict(partial_had_macro_f1=float(f1_score(truth[known],labels[known],labels=list('HAD'),average='macro',zero_division=0)),per_class_f1=dict(zip('HAD',f1_score(truth[known],labels[known],labels=list('HAD'),average=None,zero_division=0).tolist())))
        r['reference_smooth_2s']=baselines[key]
    # Paired match bootstrap: useful for change estimates, still only four development matches.
    rng=np.random.default_rng(2027);paired=[]
    for domain in ['skillcorner','dropout_stress']:
        for variant in ['observed','dropout']:
            before=[next(r for r in result['rows'] if r['domain']==domain and r['heldout']==m and r['variant']=='all' and not r['calibrated'])['ha_macro_f1'] for m in IDS]
            after=[next(r for r in result['rows'] if r['domain']==domain and r['heldout']==m and r['variant']==variant and not r['calibrated'])['ha_macro_f1'] for m in IDS]
            delta=np.array(after)-np.array(before)
            paired.append(dict(domain=domain,variant=variant,comparison='uncalibrated_all',mean_delta=float(delta.mean()),paired_match_bootstrap_95_ci=np.quantile(rng.choice(delta,(2000,4)).mean(axis=1),[.025,.975]).tolist(),n_matches=4))
    result.update(summary=summary,paired_deltas=paired,phase_audit=audits,limitations=['Weak SkillCorner alive/dead labels; no complete independent HAD GT','Four historical development matches, not a fresh final holdout','Single Metrica game repeated across training folds; not four independent games','Event control is an observed proximity proxy; not validated tackle/interception events','Calibration diagnostics for alive head inherit weak labels','No candidate promoted; no target inputs or GT used'])
    Path('docs/external_validation_v2.json').write_text(json.dumps(result,separators=(",",":"))+'\n')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    Path('docs/external_sources_v2.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# External development evidence — 2026-11-14','', 'These are other-match development results. **No candidate is promoted.** SkillCorner HAD labels are weak phase proxies; Metrica labels cover only event interiors. Full spell F1 is not reported. See [protocol](experiment_v2_protocol.md) and [all aggregate metrics](external_validation_v2.json).','','## HA cross-fit macro F1 (four matches)','','| Domain | Variant | Calibrated | Mean | Match bootstrap 95% interval |','|---|---|---|---:|---|']
    for r in summary:
        lo,hi=r['match_bootstrap_95_ci'];lines.append(f"| {r['domain']} | {r['variant']} | {r['calibrated']} | {r['ha_macro_f1_mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','Intervals resample four development matches; they are not strong generalization evidence. Paired deltas and intervals are in the JSON.','','## Metrica partial HAD diagnostic','','Each row averages predictions from four separately trained fold models on the **same** Metrica game. This is descriptive model variability, not four-match external validation.','','| Variant | Calibrated | HA macro F1 | Fixed HAD macro F1 | Control-proxy HAD macro F1 | Reference HAD macro F1 |','|---|---|---:|---:|---:|---:|']
    for v in ['all','observed','dropout']:
        for c in [False,True]:
            group=[r for r in result['rows'] if r['domain']=='metrica_partial' and r['variant']==v and r['calibrated']==c]
            metrics=[np.mean([r['ha_macro_f1'] for r in group])]+[np.mean([r[d]['partial_had_macro_f1'] for r in group]) for d in ['fixed','control_proxy','reference_smooth_2s']]
            lines.append(f'| {v} | {c} | '+' | '.join(f'{x:.4f}'for x in metrics)+' |')
    lines+=['','## Label coverage audit','','| Match | In-period frames | Phase-gap frames | HA supervised frames | HA missing-ball frames | Gap activity overlap, >1s from boundaries |','|---|---:|---:|---:|---:|---:|']
    for a in audits: lines.append('| '+' | '.join(str(a[k]) for k in ['match_id','in_period_frames','phase_gap_frames','ha_frames','ha_missing_ball_frames','gap_interior_activity_overlap'])+' |')
    lines+=['','Dynamic-event overlap is same-provider evidence and does not independently establish ball state. Gap duration, visibility and overlap details are in the JSON. No trustworthy D labels are inferred from absence.','','## Calibration and missing-ball strata','','JSON rows contain both heads’ Brier score, log loss, ten-bin ECE/reliability and HA missing-ball counts/metrics. Synthetic dropout is a stress test with preserved source labels, not evidence that naturally missing-ball labels are representative.','','## Reproduction and disposition','','Fixed protocol, source hashes and failures are documented in [engineering review](engineering_review_v2.md). No hyperparameter/model was selected using these results. The final output gate remains closed.']
    Path('docs/external_validation_v2.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))

if __name__=='__main__':main()
