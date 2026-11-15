"""Bounded V3 search with continuous inference, provenance and explicit promotion."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from fusion_v3 import ROOT,DEFAULT,BLOCKS,features,prepare,predict
from fusion_v2 import predict as v2_predict
sys.path.insert(0,str(ROOT/'evaluation'))
import possession_metrics as pm
from baseline import write_submission

SEED=20261116

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item()),encoding='utf-8')

def masked(gt,a,b):
    g=np.full_like(gt,pm.X); g[a:b]=gt[a:b]; return g

def score(gt,labels,boot=0): return pm.evaluate(gt,pm.label_codes(labels),bootstrap_n=boot)

def candidates():
    rng=np.random.default_rng(SEED); configs=[dict(DEFAULT)]
    choices=dict(window=[5,12,25],distance=[2.,3.2],confidence=[0.,.5,1.],
        attack_weight=[0.,.2,.5],setpiece_weight=[0.,.2,.6],missing_weight=[.5,1.],
        idle=[.43,.5,.56],dead_bias=[-1.2,-.8,-.4],visible_weight=[0.,.4,.8,1.2],
        team_penalty=[30.,60.,100.],enter_penalty=[10.,30.,60.],exit_penalty=[10.,30.,60.],
        dynamic=[0.,.4,.7])
    for i in range(119):
        c=dict(DEFAULT,**{k:rng.choice(v).item() for k,v in choices.items()})
        # Three prespecified experiment families, including fixed-transition controls.
        if i<39: c.update(visible_weight=0.,dynamic=0.)
        elif i<79: c.update(dynamic=0.)
        c['gap']=25 if i%10==0 else 0
        configs.append(c)
    return configs

def verify_reuse(old,new):
    if old!=new: raise ValueError('Experiment fingerprint mismatch; rerun search')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default=r'F:\non_video_dataset')
    ap.add_argument('--reuse-search',action='store_true')
    ap.add_argument('--promote',action='store_true')
    args=ap.parse_args(); out=ROOT/'evaluation/v3_reports'; out.mkdir(exist_ok=True)
    f=features(args.data_dir)
    gt_path=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt=pm.label_codes(pm.read_label_csv(str(gt_path)))
    ids=np.flatnonzero(gt!=pm.X); edges=np.linspace(ids[0],ids[-1]+1,4,dtype=int)
    windows=[(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]
    folds=[masked(gt,a,b) for a,b in windows]
    fits=[masked(gt,a+(125 if i else 0),b-(125 if i<2 else 0)) for i,(a,b) in enumerate(windows)]
    cfgs=candidates()
    import games
    paths=games.require(['atd_raw','atd_meta','smart_tagging'],data_dir=args.data_dir)
    provenance=dict(seed=SEED,windows=windows,configs=cfgs,files={str(Path(p).name):digest(p) for p in
        paths+[gt_path,ROOT/'models/fusion_v2.py',ROOT/'models/fusion_v3.py',Path(__file__),ROOT/'models/fusion_v2_config.json',ROOT/'evaluation/possession_metrics.py']})
    # JSON canonicalization makes tuple/list distinctions irrelevant.
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    v2cfg=json.loads((ROOT/'models/fusion_v2_config.json').read_text())
    base_labels=v2_predict(f,v2cfg)
    baseline=dict(full=score(gt,base_labels,1000),folds=[score(g,base_labels) for g in folds],fits=[score(g,base_labels) for g in fits])
    prepared={gap:prepare(f,gap) for gap in sorted(set(c['gap'] for c in cfgs))}
    if args.reuse_search:
        old=json.loads((out/'search.json').read_text()); verify_reuse(old['fingerprint'],fingerprint); trials=old['trials']
    else:
        trials=[]
        for i,c in enumerate(cfgs):
            labels=predict(f,c,prepared[c['gap']])
            ms=[score(g,labels) for g in fits]
            fs=[score(g,labels) for g in folds]
            trials.append(dict(config=c,fits=ms,folds=fs))
            print(f"trial {i+1}/120 spell={np.mean([m['spell_f1_mean'] for m in fs]):.4f} macro={np.mean([m['macro_f1'] for m in fs]):.4f}",flush=True)
        save(out/'search.json',dict(fingerprint=fingerprint,provenance=provenance,trials=trials))
    def select(indices):
        ix=list(indices); floor=np.mean([baseline['fits'][j]['macro_f1'] for j in ix])
        eligible=[t for t in trials if np.mean([t['fits'][j]['macro_f1'] for j in ix])>=floor]
        if not eligible: return None
        return max(eligible,key=lambda t:(np.mean([t['fits'][j]['spell_f1_mean'] for j in ix]),np.mean([t['fits'][j]['macro_f1'] for j in ix]),-t['config']['dynamic']))
    held=[]
    for i in range(3):
        t=select(j for j in range(3) if j!=i)
        held.append(dict(block=i+1,config=t['config'] if t else v2cfg,metrics=t['folds'][i] if t else baseline['folds'][i]))
    best=select(range(3))
    cfg=best['config'] if best else DEFAULT
    labels=predict(f,cfg,prepared[cfg['gap']]); full=score(gt,labels,1000)
    final_folds=[score(g,labels) for g in folds]
    accepted=bool(best and full['spell_f1_mean']>baseline['full']['spell_f1_mean'] and full['macro_f1']>=baseline['full']['macro_f1'] and
        np.mean([m['spell_f1_mean'] for m in final_folds])>np.mean([m['spell_f1_mean'] for m in baseline['folds']]))
    ablations={}
    for name,patch in dict(no_confidence={'confidence':0.},no_visible_dead={'visible_weight':0.},no_dynamic={'dynamic':0.},no_tags={'attack_weight':0.,'setpiece_weight':0.},no_repair={'gap':0}).items():
        c=dict(cfg,**patch); ablations[name]=score(gt,predict(f,c,prepared[c['gap']]))
    target=ROOT/'evaluation/results/model_fusion_v3.csv'; write_submission(str(target),labels)
    error=pm.validate_submission(pm.read_label_csv(str(target)),len(gt))
    if error: raise ValueError(error)
    save(ROOT/'models/fusion_v3_config.json',cfg)
    if args.promote and accepted: save(ROOT/'models/default_model.json',{'model':'v3'})
    health=[]
    for a,b in BLOCKS:
        x=labels[a:b]; change=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1,len(x)]
        health.append(dict(dead_fraction=float(np.mean(x=='D')),switches=len(change)-2,median_segment_seconds=float(np.median(np.diff(change))/25)))
    # Diagnostics use GT only after inference; never enter features or predictions.
    errors=[]
    from fusion_v2 import runs
    truth=pm.read_label_csv(str(gt_path))['label'].to_numpy()
    for a,b in runs((truth!='X')&(labels!=truth)):
        errors.append(dict(start=int(a),end=int(b),seconds=(b-a)/25,truth=''.join(sorted(set(truth[a:b]))),prediction=''.join(sorted(set(labels[a:b])))))
    save(out/'errors.json',errors)
    report=dict(fingerprint=fingerprint,baseline=baseline,full=full,folds=final_folds,held_out=held,config=cfg,accepted=accepted,ablations=ablations,health=health)
    save(out/'summary.json',report)
    rows=['# V3 optimization report','','Public scores are development results, not independent full-match validation. All 120 candidates were fixed before search; every candidate uses continuous half-match decoding.','',
        '| Model | spellF1 | macroF1 | Dead recall |','|---|---:|---:|---:|']
    for name,m in [('V2',baseline['full']),('V3',full)]: rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['D']['recall']:.4f} |")
    rows+=['',f'Promotion eligible: {accepted}. Search never changes the default unless --promote is explicitly supplied and the gate passes.',
        '',f"V3 spellF1 bootstrap 5–95%: {full['spell_bootstrap_low']:.4f}–{full['spell_bootstrap_high']:.4f}.",'',
        '| Held-out block | spellF1 | macroF1 |','|---|---:|---:|']
    for h in held:
        m=h['metrics']; rows.append(f"| {h['block']} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows+=['','Each block selects parameters on the other two blocks; fitting excludes 125 boundary frames. Historical label exposure and model design choices prevent interpreting these as independent test results.','','| Ablation | spellF1 | macroF1 |','|---|---:|---:|']
    for name,m in ablations.items(): rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows+=['','Full-match diagnostics: '+json.dumps(health),'','No video, private labels, or manual prediction corrections were used. See summary.json for diagnostics and errors.json for error intervals. Search records and fingerprints remain local generated artifacts.']
    (out/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    print('FINAL',full['spell_f1_mean'],full['macro_f1'],'eligible',accepted,flush=True)

if __name__=='__main__':main()
