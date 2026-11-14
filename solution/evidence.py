"""Versioned external evaluation: partial labels never certify full possession spells."""
from __future__ import annotations
import numpy as np
from evaluation import possession_metrics as pm
from .heuristic import majority_smooth
from .quality import period_blocks

PROTOCOL = 'external-evidence-v3.0'

def seconds_to_frames(seconds: float, fps: float, minimum: int = 0) -> int:
    if not np.isfinite([seconds, fps]).all() or seconds < 0 or fps <= 0:
        raise ValueError('invalid physical time or fps')
    return max(minimum, int(np.floor(seconds * fps + .5)))

def labels_from_batch(batch: dict, policy: str = 'strict') -> np.ndarray:
    if policy == 'strict':
        alive, home = batch['y_alive'], batch['y_home']
    elif policy == 'phase_proxy' and 'raw_y_alive' in batch:
        alive, home = batch['raw_y_alive'], batch['phase_y_home']
    else:
        raise ValueError('unsupported label policy')
    labels = np.full(len(alive), 'X')
    labels[alive == 0] = 'D'
    labels[(alive == 1) & (home == 1)] = 'H'
    labels[(alive == 1) & (home == 0)] = 'A'
    labels[batch['period'] <= 0] = 'X'
    return labels

def period_reference(batch: dict, half_window_s: float = 2.) -> np.ndarray:
    labels = np.full(len(batch['frame']), 'X')
    window = seconds_to_frames(half_window_s, float(batch['fps']))
    for a,b in period_blocks(batch['period']):
        labels[a:b+1] = majority_smooth(batch['nearest_codes'][a:b+1], window)
    return labels

def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def known_runs(truth, period):
    runs=[]
    for a,b in period_blocks(period):
        runs.extend((int(a+lo),int(a+hi)) for lo,hi in _runs(truth[a:b+1] != 'X'))
    return runs

def _transition_report(gt, pred, runs, fps, tolerance_s):
    tau = seconds_to_frames(tolerance_s, fps)
    # Both GT and predicted candidates require complete annotation across tolerance window.
    mask=np.zeros(len(gt), dtype=bool)
    for a,b in runs:
        if b-a > 2*tau:
            mask[a+tau:b-tau] = True
    g=[x for x in pm.transitions(gt,runs) if mask[x[0]]]
    p=[x for x in pm.transitions(pred,runs) if mask[x[0]]]
    scores=pm.transition_scores(g,p,tau,typed='from_to')
    out={k:v for k,v in scores.items() if k not in {'pairs','missed','spurious'}}
    for key in ['mean_latency','median_latency','mean_abs_latency','p90_abs_latency']:
        value=out.pop(key)
        out[key+'_s']=None if value is None else float(value/fps)
    if not g:
        for key in ['precision','recall','f1']:
            out[key]=None
    out.update(nominal_tolerance_s=tolerance_s,effective_tolerance_s=tau/fps,tolerance_frames=tau)
    # Entering/exiting D errors are meaningful only as a source-specific proxy here.
    out['dead_entries'] = _latency_subset(g,p,tau,fps,enter=True)
    out['dead_exits'] = _latency_subset(g,p,tau,fps,enter=False)
    return out

def _latency_subset(gt,pred,tau,fps,enter):
    is_kind=lambda t:t[2]==pm.D if enter else t[1]==pm.D
    g,p=[t for t in gt if is_kind(t)],[t for t in pred if is_kind(t)]
    pairs,missed,spurious=pm.match_transitions(g,p,tau,typed='from_to')
    offsets=[(b[0]-a[0])/fps for a,b in pairs]
    return dict(n_gt=len(g),n_matched=len(pairs),n_missed=len(missed),n_spurious=len(spurious),mean_latency_s=float(np.mean(offsets)) if offsets else None)

def score_sequence(truth, prediction, period, fps, *, semantics='partial_proxy', short_s=.48):
    truth,prediction,period=np.asarray(truth),np.asarray(prediction),np.asarray(period)
    if truth.ndim!=1 or truth.shape!=prediction.shape or period.shape!=truth.shape:
        raise ValueError('unaligned evaluation timelines')
    if not np.isin(truth,list('HADX')).all() or not np.isin(prediction,list('HADX')).all():
        raise ValueError('invalid label')
    seconds_to_frames(0,fps)
    truth=truth.copy();truth[period<=0]='X'
    gt,pr=pm.label_codes(truth),pm.label_codes(prediction)
    cm=pm.scored_confusion(pm.confusion(gt,pr))
    runs=known_runs(truth,period)
    blocks=[(a,b+1)for a,b in period_blocks(period)]
    segments=pm.segments(pr,blocks)
    spells=[(a,b,k)for a,b,k in segments if k in (pm.H,pm.A)]
    known=(truth!='X')&(period>0)
    inside=period>0
    has_labels=bool(known.any())
    is_complete=bool(inside.any() and known[inside].all())
    certified=semantics in {'independent_complete','synthetic_complete'}
    if certified and not is_complete:
        raise ValueError('complete annotation claim has unknown intervals')
    output=dict(protocol=PROTOCOL,semantics=semantics,n_known=int(known.sum()),coverage=float(known.sum()/max(1,inside.sum())),n_label_islands=len(runs),
                partial_macro_f1=float(pm.macro_f1(cm)) if has_labels else None,
                per_class={name:dict(zip(['precision','recall','f1'],map(float,pm.per_class_prf(cm,i))))|{'support':int(cm[i].sum())} for i,name in enumerate('HAD')},
                confusion=cm.tolist(),full_spell_f1=None,full_spell_status='not_established',
                prediction_geometry=dict(n_segments=len(segments),n_team_spells=len(spells),short_team_spells=sum((b-a)/fps<short_s for a,b,_ in spells),mean_team_spell_s=float(np.mean([(b-a)/fps for a,b,_ in spells])) if spells else None),
                live_to_dead_proxy_errors=int(cm[0,2]+cm[1,2]),dead_to_live_proxy_errors=int(cm[2,0]+cm[2,1]))
    # This deliberately exposes what old island scoring measured, without making it a headline.
    output['censored_island_spell_diagnostic']=float(pm.spell_scores(gt,pr,runs)['mean_f1']) if runs else None
    if certified:
        output['full_spell_f1']=float(pm.spell_scores(gt,pr,blocks)['mean_f1'])
        output['full_spell_status']='verified_complete' if semantics=='independent_complete' else 'synthetic_test_only'
    output['partial_proxy_transitions']=[_transition_report(gt,pr,runs,fps,t) for t in (.48,1.)]
    cuts=merges=stable_switches=0
    for a,b,k in pm.segments(gt,runs):
        local=pr[a:b]
        if k in (pm.H,pm.A):
            stable_switches+=int(np.sum((local[1:]!=local[:-1])&np.isin(local[1:],[pm.H,pm.A])&np.isin(local[:-1],[pm.H,pm.A])))
            cuts+=sum(lo>0 and hi<len(local) and local[lo-1] in (pm.H,pm.A) and local[hi] in (pm.H,pm.A) for lo,hi in _runs(local==pm.D))
        elif k==pm.D:
            merges+=int(len(local)>0 and np.isin(local,[pm.H,pm.A]).all())
    output['source_proxy_fragmentation']=dict(false_dead_cuts=int(cuts),completely_missed_dead_segments=int(merges),stable_team_switches=stable_switches)
    return output

def visibility_strata(batch):
    names=batch['feature_names'].astype(str).tolist();x=batch['X'];fps=float(batch['fps'])
    visible=x[:,names.index('ball_visible')]==1
    duration=np.zeros(len(visible))
    for a,b in period_blocks(batch['period']):
        for lo,hi in _runs(~visible[a:b+1]):duration[a+lo:a+hi]=(hi-lo)/fps
    home,away=x[:,names.index('home_d1_m')],x[:,names.index('away_d1_m')]
    changes=np.zeros(len(visible),dtype=int)
    nearest=batch['nearest_codes'];width=seconds_to_frames(1.,fps,1)
    for a,b in period_blocks(batch['period']):
        switches=np.r_[0,(nearest[a+1:b+1]!=nearest[a:b])&(nearest[a+1:b+1]>=0)&(nearest[a:b]>=0)]
        cumulative=np.r_[0,np.cumsum(switches)]
        idx=np.arange(b-a+1);changes[a:b+1]=cumulative[idx+1]-cumulative[np.maximum(0,idx-width+1)]
    return dict(visible=visible,missing_short=(duration>0)&(duration<=.5),missing_medium=(duration>.5)&(duration<=2.),missing_long=duration>2.,near_opponents_proxy=visible&(home<3)&(away<3)&(np.abs(home-away)<1),frequent_nearest_switch_proxy=changes>=2)

def error_attribution(truth, p_alive, p_home, decoded):
    truth=np.asarray(truth);known=np.isin(truth,['H','A'])
    home=np.where(p_home>=.5,'H','A')
    raw=np.array(['H','A','D'])[np.argmax(np.column_stack([p_alive*p_home,p_alive*(1-p_home),1-p_alive]),axis=1)]
    return dict(n_known_ha=int(known.sum()),home_head_wrong=int((known&(home!=truth)).sum()),alive_head_below_half=int((known&(p_alive<.5)).sum()),joint_argmax_dead=int((known&(raw=='D')).sum()),decoder_harmed_raw_correct=int((known&(raw==truth)&(decoded!=truth)).sum()),decoder_fixed_raw_wrong=int((known&(raw!=truth)&(decoded==truth)).sum()))
