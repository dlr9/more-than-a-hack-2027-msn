"""Score frozen historical candidate edits on complete public external HAD.

No mechanism rerun, parameter fitting, edit selection by GT, or target access.
"""
import argparse,datetime,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r11.spell_roi import score,audit_edit
from solution.idsse import digest
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from evaluation import possession_metrics as pm
from scripts.research_r7 import IDS,load,checked_path,write_new
from scripts.diagnose_r9_spell_loss import cache_path,DOMAINS
from scripts.probe_r10_acquisition import hc_path
PROTOCOL=Path('docs/R11_ROI_PROTOCOL.json');ROOT=Path('local_artifacts/r11/roi')

def specs():
    for m in IDS:
        for d in DOMAINS:
            yield m,d,'E09E2',Path(f'local_artifacts/r9-v2/e09e2/{m}-{d}-M3.npz')
            if d=='natural':yield m,d,'HC_M3',hc_path(m,d)

def edits(base,candidate,blocks):
    out=[]
    for lo,hi in blocks:
        a=None
        for i in range(lo,hi+1):
            changed=i<hi and base[i]!=candidate[i]
            if a is not None and (not changed or candidate[i]!=candidate[a]):
                out.append((int(a),int(i),str(candidate[a])));a=None
            if changed and a is None:a=i
    return out

def freeze():
    inputs={}
    for m,d,v,p in specs():
        for f in [checked_path(m,d),cache_path(m,'M3',d),p]:inputs[str(f)]=digest(f)
    write_new(PROTOCOL,dict(id='R11-H4-FROZEN_ROI',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      hypothesis='A recovered event is useful only if the resulting complete spells match; evaluate existing frozen edits, never regenerate/tune R9/R10 mechanisms.',
      source='Public IDSSE DFL three previously used matches; complete source HAD evaluator only; not blind',
      variants='E09E2 vs M3 in12domains and HC_M3 vs M3 on3natural matches; existing saved predictions only',
      edits='maximal same-candidate-label changed intervals, selected from predictions alone within continuous half blocks',
      individual_budget='All E09E2 edits; for HC_M3 use at most32 intervals uniformly spread in chronological index, selected without GT; full joint delta always all edits',
      controls='No-edit exactidentity; independently assess each frozen edit vs same full baseline; effects nonadditive, no summing to claim improvement',
      outputs='Full five-IoU TP/FP/FN, macro, D, short GT spell matching; isolated positive/zero/negative ROI aggregate; label arrays never exported',
      inputs=inputs,code_hashes={p:digest(p) for p in ['r11/spell_roi.py','scripts/evaluate_r11_roi.py','evaluation/possession_metrics.py']},
      new_fits=0,selection_allowed=False,rollback='unchanged historical candidate caches',stop='No parameter/model change based on this retrospective diagnosis'))

def official(g,p,blocks):return pm.spell_scores(pm.label_codes(g),pm.fill_abstentions(pm.label_codes(p),blocks),blocks)

def check(s,o):
    assert s.n_gt==o['n_gt'] and s.n_pred==o['n_pred']
    assert abs(s.spell_f1_mean-o['mean_f1'])<=1e-14
    for j,t in enumerate(pm.SPELL_IOUS):
        assert s.matched_by_iou[j]==o['per_iou'][t]['n_matched']
        assert s.f1_by_iou[j]==o['per_iou'][t]['f1']

def diagnostics(g,p,blocks,fps):
    gc=pm.label_codes(g);pc=pm.label_codes(p);gs=pm.possession_spells(gc,blocks);ps=pm.possession_spells(pc,blocks);ov=pm.spell_overlaps(gs,ps)
    short={}
    for label,lo,hi in [('lt1',0,1),('1to5',1,5),('ge5',5,float('inf'))]:
        ix=[i for i,(a,b,c) in enumerate(gs) if lo<=(b-a)/fps<hi]
        short[label]=dict(n_gt=len(ix),matched_by_iou={str(t):int(pm.matched_gt_flags(ov,len(gs),t)[ix].sum()) for t in pm.SPELL_IOUS})
    ds=[s for s in pm.segments(gc,blocks) if s[2]==pm.D]
    return dict(short=short,missed_D_frames=int(((gc==pm.D)&(pc!=pm.D)).sum()),false_D_frames=int(((gc!=pm.D)&(pc==pm.D)).sum()),complete_D_misses=sum(not np.any(pc[a:b]==pm.D) for a,b,c in ds))

def run():
    p=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for f,h in {**p['inputs'],**p['code_hashes']}.items():
        if digest(f)!=h:raise ValueError('drift '+f)
    rows=[]
    for m,d,v,path in specs():
        out=ROOT/f'{m}-{d}-{v}.json'
        if out.exists():
            r=json.loads(out.read_text());assert r['protocol_sha']==digest(PROTOCOL);rows.append(r);continue
        start=time.monotonic();cpu=time.process_time();b=load(m,d);g=labels_from_batch(b).tolist();base=np.load(cache_path(m,'M3',d))['decoded'].tolist();cand=np.load(path)['decoded'].tolist();blocks=[(int(a),int(z)) for a,z in continuous_blocks(b['frame'],b['period'])]
        before=score(g,base,blocks=blocks);after=score(g,cand,blocks=blocks);ob=official(g,base,blocks);oa=official(g,cand,blocks);check(before,ob);check(after,oa)
        es=edits(base,cand,blocks);selected=es if v=='E09E2' or len(es)<=32 else [es[i] for i in np.linspace(0,len(es)-1,32,dtype=int)]
        changes=[]
        for a,z,k in selected:
            e=audit_edit(g,base,a,z,k,blocks=blocks)
            changed=base.copy();changed[a:z]=[k]*(z-a);actual=official(g,changed,blocks)
            assert abs(e['edited']['spell_f1_mean']-actual['mean_f1'])<=1e-14
            changes.append({x:e[x] for x in ['delta_spellF1','delta_matched_iou_05','delta_pred_spells']}|dict(duration_s=(z-a)/float(b['fps'])))
        delta=np.array([x['delta_spellF1'] for x in changes]);r=dict(id=p['id'],match=m,domain=d,variant=v,baseline=ob,candidate=oa,
          full_delta_spell=after.spell_f1_mean-before.spell_f1_mean,baseline_diagnostics=diagnostics(g,base,blocks,float(b['fps'])),candidate_diagnostics=diagnostics(g,cand,blocks,float(b['fps'])),
          edit_count=len(es),evaluated_isolated=len(selected),isolated=dict(positive=int((delta>1e-14).sum()),negative=int((delta< -1e-14).sum()),zero=int((abs(delta)<=1e-14).sum()),sum_isolated_delta=float(delta.sum()),nonadditivity_warning=True,per_edit_no_positions=changes),
          source_prediction_sha=digest(path),baseline_prediction_sha=digest(cache_path(m,'M3',d)),protocol_sha=digest(PROTOCOL),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes=p['code_hashes'],new_fits=0,wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_new(out,r);rows.append(r);print(m,d,v,'joint',round(r['full_delta_spell'],6),'edits',len(es),'isolated +/-',r['isolated']['positive'],r['isolated']['negative'],flush=True)
    write_new(ROOT/'results.json',dict(rows=rows,official_equivalent=True,selection_allowed=False,new_fits=0));write_new('docs/R11_ROI_RESULTS.json',dict(rows=rows,official_equivalent=True,selection_allowed=False,new_fits=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
