"""Post-score forensic accounting and exact replay; never select/tune a model."""
import json
from pathlib import Path
import numpy as np
from scripts.research_r12 import ROOT,P,matched_changes
from scripts.research_r7 import load,write_new
from scripts.evaluate_r8_decoder import prediction_path
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
from r12.control_global import decode
from r11.spell_roi import score
from scripts.evaluate_r11_roi import check
from evaluation import possession_metrics as pm

def extra(g,p,blocks,visible):
 gc=pm.label_codes(g);pc=pm.label_codes(p);gs=pm.possession_spells(gc,blocks)
 cuts=0;affected=0
 for a,b,_ in gs:
  dead=pc[a:b]==pm.D
  cuts+=int(np.sum(np.diff(np.r_[False,dead,False].astype(int))==1))
  affected+=int(dead.any())
 same=[];recovered=0
 for lo,hi in blocks:
  seg=pm.segments(gc,[(lo,hi)])
  for j,(a,b,t) in enumerate(seg):
   if t==pm.D and 0<j<len(seg)-1 and seg[j-1][2]==seg[j+1][2]!=pm.D:
    team=seg[j-1][2];same.append((a,b,team))
    # Both correctly typed D boundaries within 1s, plus correct surrounding owner.
    tr=pm.transitions(pc,[(lo,hi)])
    recovered+=int(any(abs(i-a)<=25 and x==team and y==pm.D for i,x,y in tr) and any(abs(i-b)<=25 and x==pm.D and y==team for i,x,y in tr))
 out=dict(false_D_runs_inside_true_possessions=cuts,true_possessions_cut_by_D=affected,same_team_restarts=len(same),same_team_restarts_both_boundaries_within1s=recovered,strata={})
 for name,mask in [('observed',visible),('missing',~visible)]:
  out['strata'][name]=dict(n=int(mask.sum()),wrong=int((mask&(g!=p)).sum()),false_D=int((mask&(g!='D')&(p=='D')).sum()),missed_D=int((mask&(g=='D')&(p!='D')).sum()))
 return out

def run():
 result=json.loads(Path('docs/R12_RESULTS.json').read_text());rows=[];replays=0;scorings=0
 for row in result['rows']:
  m,d=row['match'],row['domain'];b=load(m,d);g=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']))
  path=ROOT/f'{m}-{d}.npz';assert digest(path)==row['prediction_sha'];z=dict(np.load(path))
  events=json.loads((ROOT/f'{m}-{d}-anchors.json').read_text());signal=np.full(len(g),-1,np.int8)
  for e in events:signal[e['start']:e['end']]=int(e['team']=='A')
  signals={'global':signal,'no_anchors':np.full(len(g),-1,np.int8),'shift':signal.copy(),'shuffle_team':signal.copy()};rng=np.random.default_rng(1201)
  for lo,hi in blocks:signals['shift'][lo:hi]=np.roll(signal[lo:hi],750)
  for e in events:signals['shuffle_team'][e['start']:e['end']]=rng.integers(2)
  alive=np.load(prediction_path(m,'M3',d))['p_alive'];home=np.load(f'local_artifacts/r8/e08c/{m}-{d}.npz')['p_home']
  for name,s in signals.items():
   p,l,_=decode(alive,home,s,b['frame'],b['period'],25,z['HC_M3']);np.testing.assert_array_equal(p,z[name]);np.testing.assert_array_equal(l,z[name+'_latent']);replays+=1
  visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
  measures={}
  for name in row['scores']:
   official=pm.spell_scores(pm.label_codes(g),pm.label_codes(z[name]),blocks)
   check(score(g.tolist(),z[name].tolist(),blocks=blocks),official)
   assert abs(official['mean_f1']-row['scores'][name]['metrics']['spell']['mean_f1'])<1e-14
   measures[name]=extra(g,z[name],blocks,visible);scorings+=1
  gt=pm.possession_spells(pm.label_codes(g),blocks)
  coverage={}
  for name,low,high in [('lt1',0,25),('1to5',25,125),('ge5',125,float('inf'))]:
   selected=[(a,c,t) for a,c,t in gt if low<=c-a<high]
   coverage[name]=dict(n=len(selected),has_correct_anchor=sum(bool(np.any(signal[a:c]==t)) for a,c,t in selected))
  direct=[t for t in pm.transitions(pm.label_codes(g),blocks) if t[1]!=pm.D and t[2]!=pm.D]
  # Time support is diagnostic only, never used as inference candidates.
  supported=sum(any(abs(e['start']-i)<=25 and int(e['team']=='A')==team for e in events) for i,_,team in direct)
  rows.append(dict(match=m,domain=d,additional_metrics=measures,anchor_spell_coverage=coverage,true_direct_handover_count=len(direct),direct_handover_with_correct_anchor_start_within1s=supported))
 write_new('docs/R12_VERIFICATION.json',dict(rows=rows,exact_prediction_replays=replays,official_and_ROI_equivalent_scores=scorings,protocol_sha=digest(P),posthoc_diagnostics_not_selection=True))
 print('Verified',replays,'decoder replays and',scorings,'complete official/ROI scores')
if __name__=='__main__':run()
