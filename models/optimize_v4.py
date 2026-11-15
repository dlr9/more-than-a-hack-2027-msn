"""V4 bounded search with pooled official selection and leave-block audit."""
import argparse, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from fusion_v4 import ROOT,BASE,DEFAULT,BLOCKS,features,prepare,predict
from fusion_v2 import predict as v2_predict
from optimize_v3 import digest, save, masked, score, verify_reuse
sys.path.insert(0,str(ROOT/'evaluation'))
import possession_metrics as pm
from baseline import write_submission


def candidates():
    # Fixed before evaluation. Deduplicate zero-boost support variants.
    return [dict(boost=b,team_penalty=t,dead_penalty=d,support=s)
        for b,t,d,s in itertools.product([0.,.4,.8,1.2],[30.,60.],[15.,30.,60.],['movement','stationary'])
        if b or s=='movement']


def refinement_candidates():
    """Stage two: 21 additions, declared after the base study plateaued."""
    return candidates()+[dict(boost=b,team_penalty=t,dead_penalty=30.,support=s)
        for b,t,s in itertools.product([0.,.4,.8,1.2],[45.,50.,55.],['movement','stationary'])
        if b or s=='movement']


def select(trials, baseline, key):
    eligible=[t for t in trials if t[key]['macro_f1']>=baseline[key]['macro_f1']]
    if not eligible: return None
    return max(eligible,key=lambda t:(t[key]['spell_f1_mean'],t[key]['macro_f1'],
        -t['config']['boost'],-abs(t['config']['team_penalty']-60),-abs(t['config']['dead_penalty']-30)))


def publish_candidate(root, out, cfg, labels, accepted, promote):
    """Rejected/ordinary experiments must never overwrite a deployed model."""
    save(out/'candidate_config.json',cfg)
    write_submission(str(out/'candidate.csv'),labels)
    if accepted and promote:
        save(root/'models/fusion_v4_config.json',cfg)
        write_submission(str(root/'evaluation/results/model_fusion_v4.csv'),labels)
        save(root/'models/default_model.json',{'model':'v4'})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default=r'F:\non_video_dataset')
    ap.add_argument('--reuse-search',action='store_true')
    ap.add_argument('--promote',action='store_true')
    ap.add_argument('--refine',action='store_true',help='Second-stage 21 additions; retain base study separately')
    args=ap.parse_args(); out=ROOT/'evaluation'/('v4_refine_reports' if args.refine else 'v4_reports'); out.mkdir(exist_ok=True)
    f=features(args.data_dir); p=prepare(f)
    gt_path=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt=pm.label_codes(pm.read_label_csv(str(gt_path)))
    ids=np.flatnonzero(gt!=pm.X); edges=np.linspace(ids[0],ids[-1]+1,4,dtype=int)
    windows=[(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]
    folds=[masked(gt,a,b) for a,b in windows]
    # Pool retained frames under the official scorer; do not average clipped-spell F1s.
    train=[]
    for a,b in windows:
        g=gt.copy(); g[max(0,a-125):min(len(g),b+125)]=pm.X; train.append(g)
    cfgs=refinement_candidates() if args.refine else candidates()
    import games
    paths=games.require(['atd_raw','atd_meta','smart_tagging'],data_dir=args.data_dir)
    sources=[gt_path,Path(__file__),ROOT/'models/fusion_v2.py',ROOT/'models/fusion_v3.py',ROOT/'models/fusion_v4.py',ROOT/'models/optimize_v3.py',ROOT/'models/fusion_v2_config.json',ROOT/'evaluation/possession_metrics.py',ROOT/'games.py',ROOT/'starter/baseline.py',ROOT/'starter/load_smart_tagging.py',ROOT/'features_v2.npz',ROOT/'features_v3_speed.npz']
    provenance=dict(windows=windows,purge=125,candidates=cfgs,selection='pooled official F1, full-public refit',files={str(Path(x).name):digest(x) for x in paths+sources})
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    baseline_labels=v2_predict(f,BASE)
    # Identity check prevents accidentally replacing the V2 reference with a different model.
    np.testing.assert_array_equal(predict(f,DEFAULT,p),baseline_labels)
    def evaluate(labels):
        values=dict(full=score(gt,labels),folds=[score(g,labels) for g in folds])
        for i,g in enumerate(train): values['train'+str(i)]=score(g,labels)
        return values
    baseline=evaluate(baseline_labels)
    if args.reuse_search:
        old=json.loads((out/'search.json').read_text()); verify_reuse(old['fingerprint'],fingerprint); trials=old['trials']
    else:
        trials=[]
        for i,c in enumerate(cfgs):
            labels=predict(f,c,p); trial=dict(config=c,**evaluate(labels)); trials.append(trial)
            print(f"trial {i+1}/{len(cfgs)} full spell={trial['full']['spell_f1_mean']:.4f} macro={trial['full']['macro_f1']:.4f}",flush=True)
        save(out/'search.json',dict(fingerprint=fingerprint,provenance=provenance,trials=trials))
    held=[]
    for i in range(3):
        chosen=select(trials,baseline,'train'+str(i))
        held.append(dict(block=i+1,config=chosen['config'] if chosen else DEFAULT,metrics=chosen['folds'][i] if chosen else baseline['folds'][i]))
    selected=select(trials,baseline,'full')
    cfg=selected['config'] if selected else DEFAULT
    labels=predict(f,cfg,p); full=score(gt,labels,1000)
    final_folds=[score(g,labels) for g in folds]
    base_full=score(gt,baseline_labels,1000)
    held_score=float(np.mean([h['metrics']['spell_f1_mean'] for h in held]))
    base_fold=float(np.mean([m['spell_f1_mean'] for m in baseline['folds']]))
    accepted=bool(full['spell_f1_mean']>base_full['spell_f1_mean'] and full['macro_f1']>=base_full['macro_f1'] and
        np.mean([m['spell_f1_mean'] for m in final_folds])>base_fold and held_score>=base_fold)
    ablations={}
    for name,patch in dict(no_visible_boost={'boost':0.},v2_transition_penalties={'team_penalty':60.,'dead_penalty':30.}).items():
        ablations[name]=score(gt,predict(f,dict(cfg,**patch),p))
    target=out/'candidate.csv'; write_submission(str(target),labels)
    error=pm.validate_submission(pm.read_label_csv(str(target)),len(gt))
    if error: raise ValueError(error)
    publish_candidate(ROOT,out,cfg,labels,accepted,args.promote)
    health=[]
    for a,b in BLOCKS:
        sub=labels[a:b]; changes=np.r_[0,np.flatnonzero(sub[1:]!=sub[:-1])+1,len(sub)]
        health.append(dict(dead_fraction=float(np.mean(sub=='D')),switches=len(changes)-2,median_segment_seconds=float(np.median(np.diff(changes))/25)))
    truth=pm.read_label_csv(str(gt_path))['label'].to_numpy(); visible=np.isfinite(f['ball']).all(axis=1)
    strat={}
    for name,pred in [('v2',baseline_labels),('v4',labels)]:
        strat[name]={}
        for state,mask in [('visible',visible),('missing',~visible)]:
            g=gt.copy();g[~mask]=pm.X
            cm=pm.confusion(g,pm.label_codes(pred)); strat[name][state]=cm.tolist()
    report=dict(fingerprint=fingerprint,provenance=provenance,config=cfg,accepted=accepted,baseline=base_full,baseline_folds=baseline['folds'],full=full,folds=final_folds,held_out=held,held_out_mean_spell=held_score,baseline_mean_spell=base_fold,ablations=ablations,health=health,stratified_confusion=strat)
    save(out/'summary.json',report)
    from fusion_v2 import runs
    save(out/'errors.json',[dict(start=int(a),end=int(b),seconds=(b-a)/25) for a,b in runs((truth!='X')&(truth!=labels))])
    study_note=('Second-stage refinement: 21 additions after the initial 42-candidate study plateaued, 63 total configurations re-evaluated. This is adaptive development, not an untouched test.' if args.refine else 'Forty-two fixed candidates in the base study.')
    rows=['# V4 conservative refinement','','All public results are development scores. '+study_note+' V2 team emissions are unchanged. Zero boost with V2 penalties exactly reproduces V2.','',
        '| Model | spellF1 | macroF1 | Dead recall | Predicted spells |','|---|---:|---:|---:|---:|']
    for name,m in [('V2',base_full),('V4',full)]: rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['D']['recall']:.4f} | {m['n_pred_spells']} |")
    rows+=['',f'Promotion eligible: {accepted}. Held-out mean spellF1 {held_score:.4f}; V2 block mean {base_fold:.4f}.',
        '', 'Selection uses pooled official F1 on the retained frames, not an average of cropped-spell F1s. Held-out labels and their 125-frame buffer are masked for each fold selection. Final refit uses all public labels, explicitly as development data.',
        '', '| Held-out block | spellF1 | macroF1 |','|---|---:|---:|']
    for h in held:
        m=h['metrics'];rows.append(f"| {h['block']} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows+=['','| Ablation | spellF1 | macroF1 |','|---|---:|---:|']
    for name,m in ablations.items():rows.append(f"| {name} | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} |")
    rows+=['',f"V4 spellF1 bootstrap 5–95%: {full['spell_bootstrap_low']:.4f}–{full['spell_bootstrap_high']:.4f}.",
        '', 'Public labels have historical tuning and design exposure. Cross-validation and bootstrap do not establish independent full-match improvement.',
        '', 'Full-match diagnostics: '+json.dumps(health)]
    (out/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    print('FINAL',full['spell_f1_mean'],full['macro_f1'],'eligible',accepted,flush=True)

if __name__=='__main__':main()
