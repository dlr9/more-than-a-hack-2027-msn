"""E09-D evaluator-only counterfactuals. Never serialize oracle label arrays."""
import json, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import possession_metrics as pm
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from solution.idsse import digest
from scripts.research_r7 import IDS, load, checked_path, write_new
from scripts.evaluate_r8_decoder import prediction_path

PROTOCOL=Path('docs/R9_V2_E09D_PROTOCOL.json')
ROOT=Path('local_artifacts/r9-v2/e09d')
DOMAINS=('natural','dropout','short','long')

def cache_path(m,v,d):
    if v=='M3': return prediction_path(m,v,d)
    if v=='H1': return Path('local_artifacts/r8/e08a')/f'{m}-H1-{d}.npz'
    return Path('local_artifacts/r8/e08c')/f'{m}-{d}.npz'

def snap_boundaries(gt, pred, blocks, tol):
    """Oracle timing only: keep state order, snap only inside original neighbors."""
    out=pred.copy(); moved=0
    for lo,hi in blocks:
        pts=pm.transitions(pred,[(lo,hi)])
        pairs,_,_=pm.match_transitions(pm.transitions(gt,[(lo,hi)]),pts,tol,typed='pair')
        targets={p[0]:g[0] for g,p in pairs}
        edges=[lo]+[p[0] for p in pts]+[hi]
        proposed=[lo]
        for j,p in enumerate(pts,1):
            t=targets.get(p[0],p[0])
            # Original neighboring edges remain strict bounds; no erased short spell.
            proposed.append(t if edges[j-1]<t<edges[j+1] else p[0])
        proposed.append(hi)
        # Simultaneous snaps may cross; reject both conflicting moves.
        bad={j for j in range(1,len(proposed)-1) if proposed[j]<=proposed[j-1] or proposed[j]>=proposed[j+1]}
        for j in bad: proposed[j]=edges[j]
        if np.any(np.diff(proposed)<=0): raise AssertionError('nonmonotone oracle')
        states=[int(pred[lo])]+[p[2] for p in pts]
        for a,b,s in zip(proposed[:-1],proposed[1:],states):out[a:b]=s
        moved+=sum(a!=b for a,b in zip(proposed,edges))
    return out,moved

def repairs(gt,pred,home,blocks,fps,reachable):
    """Evaluation-only hypothetical repairs; no algorithm is allowed to call this."""
    false=(gt!=pm.D)&(pred==pm.D);miss=(gt==pm.D)&(pred!=pm.D)
    fd=pred.copy();fd[false]=np.where(home[false]>=.5,pm.H,pm.A)
    md=pred.copy();md[miss]=pm.D
    dead=fd.copy();dead[miss]=pm.D
    team=pred.copy();mask=(gt!=pm.D)&(pred!=pm.D);team[mask]=gt[mask]
    both=dead.copy();both[gt!=pm.D]=gt[gt!=pm.D]
    local=pred.copy();local[reachable&mask]=gt[reachable&mask]
    timing,moved=snap_boundaries(gt,pred,blocks,round(fps))
    return dict(false_D=fd,missed_D=md,all_D=dead,team=team,both=both,timing_1s=timing,observable_team=local),moved

def budget(gt,pred,home,blocks,fps,visible,players,reachable):
    if not np.isin(gt,[0,1,2]).all() or not np.isin(pred,[0,1,2]).all():raise ValueError('complete HAD only')
    gs=pm.possession_spells(gt,blocks);ps=pm.possession_spells(pred,blocks)
    base=pm.spell_scores(gt,pred,blocks);ov=pm.spell_overlaps(gs,ps)
    bins={}
    for name,lo,hi in [('lt0p5',0,.5),('0p5to1',.5,1),('1to3',1,3),('ge3',3,float('inf'))]:
        ids=[i for i,(a,b,_) in enumerate(gs) if lo<=(b-a)/fps<hi]
        bins[name]=dict(n_gt=len(ids),no_same_team_overlap=sum(not np.any(pred[a:b]==s) for i,(a,b,s) in enumerate(gs) if i in ids),
            missed_by_iou={str(t):int(sum(not pm.matched_gt_flags(ov,len(gs),t)[i] for i in ids)) for t in pm.SPELL_IOUS})
    fragmented=cuts=wrong_switches=0;context=dict(ball_missing=0,zero_players=0,observable_control=0)
    for a,b,s in gs:
        seg=pm.segments(pred,[(a,b)])
        c=sum(t==pm.D and j>0 and j+1<len(seg) and seg[j-1][2]!=pm.D and seg[j+1][2]!=pm.D for j,(_,_,t) in enumerate(seg))
        cuts+=c;fragmented+=c>0
        wrong_switches+=sum(x!=pm.D and y!=pm.D for _,x,y in pm.transitions(pred,[(a,b)]))
        if not pm.matched_gt_flags(ov,len(gs),.5)[gs.index((a,b,s))]:
            context['ball_missing']+=bool(np.any(~visible[a:b]));context['zero_players']+=bool(np.any(players[a:b]==0));context['observable_control']+=bool(np.any(reachable[a:b]))
    dead=[s for s in pm.segments(gt,blocks) if s[2]==pm.D]
    joined=sum(a<lo and hi<b for lo,hi,_ in dead for a,b,_ in ps)
    oracles,moved=repairs(gt,pred,home,blocks,fps,reachable)
    scores={}
    for name,p in oracles.items():
        s=pm.spell_scores(gt,p,blocks)
        s.update(delta_spell=s['mean_f1']-base['mean_f1'],changed_frames=int(np.sum(p!=pred)),
            delta_matched={str(t):s['per_iou'][t]['n_matched']-base['per_iou'][t]['n_matched'] for t in pm.SPELL_IOUS})
        scores[name]=s
    boundary={}
    for name,select in [('HA',lambda x:x[1]!=pm.D and x[2]!=pm.D),('D',lambda x:x[1]==pm.D or x[2]==pm.D)]:
        offsets=[];ng=npred=missed=spurious=0
        for block in blocks:
            g=[x for x in pm.transitions(gt,[block]) if select(x)];p=[x for x in pm.transitions(pred,[block]) if select(x)]
            pairs,mi,sp=pm.match_transitions(g,p,round(fps),typed='pair');ng+=len(g);npred+=len(p);missed+=len(mi);spurious+=len(sp)
            offsets.extend((b[0]-a[0])/fps for a,b in pairs)
        boundary[name]=dict(n_gt=ng,n_pred=npred,n_missed=missed,n_spurious=spurious,n_matched=len(offsets),mean_abs_seconds=float(np.mean(np.abs(offsets))) if offsets else None)
    return dict(base=base,oracle=scores,short_spells=bins,false_D_cut_count=cuts,fragmented_gt_spells=fragmented,
        true_D_joined_by_one_pred_spell=joined,n_dead=len(dead),complete_missed_D=sum(not np.any(pred[a:b]==pm.D) for a,b,_ in dead),
        false_D_frames=int(np.sum((pred==pm.D)&(gt!=pm.D))),missed_D_frames=int(np.sum((pred!=pm.D)&(gt==pm.D))),
        HA_switches_inside_constant_GT_spell=wrong_switches,boundaries=boundary,unmatched_at_p5_context_overlapping=context,
        reachable_frames=int(reachable.sum()),reachable_gt_spells=sum(np.any(reachable[a:b]) for a,b,_ in gs),timing_moves=moved,
        D_HA_interaction=scores['both']['delta_spell']-scores['all_D']['delta_spell']-scores['team']['delta_spell'])

def run():
    protocol=json.loads(PROTOCOL.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    for p,h in protocol['inputs'].items():
        if digest(p)!=h:raise ValueError('input drift '+p)
    rows=[]
    for m in IDS:
      for d in DOMAINS:
        b=load(m,d);gt=pm.label_codes(labels_from_batch(b));fps=float(b['fps']);blocks=continuous_blocks(b['frame'],b['period'])
        names=b['feature_names'].tolist();col=lambda n:b['X'][:,names.index(n)]
        visible=col('ball_visible')==1;players=col('n_home_positions')+col('n_away_positions')
        reach=visible&(col('n_home_positions')>=8)&(col('n_away_positions')>=8)&(np.minimum(col('home_d1_m'),col('away_d1_m'))<=2)&(np.abs(col('home_d1_m')-col('away_d1_m'))>=1)&(col('ball_speed_mps')>=.5)&(col('ball_speed_mps')<=12)
        for v in ('M3','H1','C1'):
          out=ROOT/f'{m}-{d}-{v}.json'
          if out.exists():
            row=json.loads(out.read_text())
            if row['protocol_sha']!=digest(PROTOCOL) or row['script_sha']!=digest(__file__):raise ValueError('run identity drift')
            rows.append(row);continue
          start=time.monotonic();cpu=time.process_time();path=cache_path(m,v,d);c=dict(np.load(path,allow_pickle=False));pred=pm.label_codes(c['decoded'])
          full=budget(gt,pred,c['p_home'],blocks,fps,visible,players,reach)
          halves={str(i+1):budget(gt[a:z],pred[a:z],c['p_home'][a:z],[(0,z-a)],fps,visible[a:z],players[a:z],reach[a:z]) for i,(a,z) in enumerate(blocks)}
          row=dict(match=m,domain=d,variant=v,full=full,halves=halves,protocol_sha=digest(PROTOCOL),script_sha=digest(__file__),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),prediction_sha=digest(path),data_sha=digest(checked_path(m,d)),fits=0,wall_cpu_memory=dict(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
          write_new(out,row);rows.append(row)
          print(m,d,v,{k:round(s['delta_spell'],4) for k,s in full['oracle'].items()},flush=True)
    if not (ROOT/'results.json').exists():write_new(ROOT/'results.json',dict(rows=rows,oracle_arrays_saved=False))
    print('Complete: 36 full-match and 72 half budgets; zero fits; no oracle labels saved',flush=True)
if __name__=='__main__':run()
