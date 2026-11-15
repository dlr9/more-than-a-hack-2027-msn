"""Complete public-source metrics with denominators and correctly typed boundaries."""
import numpy as np
from evaluation import possession_metrics as pm
from .evidence import labels_from_batch, visibility_strata
from .quality import period_blocks
from scripts.evaluate_r6 import evaluate as evaluate_r6


def _lengths(spells, fps):
    values = [(b-a)/fps for a, b, _ in spells]
    return dict(n=len(values), mean_s=float(np.mean(values)) if values else None,
                median_s=float(np.median(values)) if values else None,
                duration_counts=dict(lt1=sum(x<1 for x in values), one_to3=sum(1<=x<3 for x in values), ge3=sum(x>=3 for x in values)))


def extended(batch, alive, home, decoded):
    truth = labels_from_batch(batch)
    if not np.isin(truth, list('HAD')).all():
        raise ValueError('R7 complete evaluation refuses partial labels')
    fps = float(batch['fps'])
    score = evaluate_r6(batch, alive, home, decoded, crosschecked=True)
    # Legacy R6 typed='from_to' falls through to destination-only in the scorer.
    # Keep it explicitly legacy; new boundaries use the supported typed='pair'.
    score['legacy_destination_only_transitions'] = score.pop('partial_proxy_transitions')
    gt, pred = pm.label_codes(truth), pm.label_codes(decoded)
    blocks = [(a, b+1) for a, b in period_blocks(batch['period'])]
    spells = pm.spell_scores(gt, pred, blocks)
    score['spell_by_iou'] = spells['per_iou']
    gs, ps = pm.possession_spells(gt, blocks), pm.possession_spells(pred, blocks)
    score['spell_lengths'] = dict(truth=_lengths(gs, fps), prediction=_lengths(ps, fps))
    overlaps = pm.spell_overlaps(gs, ps)
    score['matched_iou'] = {}
    score['short_spell_recall'] = {}
    for threshold in pm.SPELL_IOUS:
        used_g, used_p, ious = set(), set(), []
        for iou, g, p in overlaps:
            if iou < threshold: break
            if g not in used_g and p not in used_p:
                used_g.add(g);used_p.add(p);ious.append(iou)
        score['matched_iou'][str(threshold)] = dict(n=len(ious), mean=float(np.mean(ious)) if ious else None, median=float(np.median(ious)) if ious else None)
        flags = pm.matched_gt_flags(overlaps, len(gs), threshold)
        for name, lo, hi in [('lt1', 0, 1), ('one_to3', 1, 3), ('ge3', 3, float('inf'))]:
            which = np.array([lo <= (b-a)/fps < hi for a, b, _ in gs], dtype=bool)
            n = int(which.sum());matched = int(flags[which].sum())
            score['short_spell_recall'][name+'@'+str(threshold)] = dict(n_gt=n, n_matched=matched, recall=matched/n if n else None)
    gt_dead = [s for s in pm.segments(gt, blocks) if s[2] == pm.D]
    frag = score['source_proxy_fragmentation']
    score['dead_errors'] = dict(false_cuts=frag['false_dead_cuts'], gt_ha_spells=len(gs),
        false_cuts_per_gt_spell=frag['false_dead_cuts']/len(gs) if gs else None,
        missed_segments=frag['completely_missed_dead_segments'], gt_dead_segments=len(gt_dead),
        missed_segment_rate=frag['completely_missed_dead_segments']/len(gt_dead) if gt_dead else None,
        majority_missed_segments=sum(np.mean(pred[a:b]==pm.D)<.5 for a,b,_ in gt_dead),
        false_dead_frames=int(np.sum((pred==pm.D)&(gt!=pm.D))),
        missed_dead_frames=int(np.sum((pred!=pm.D)&(gt==pm.D))),
        gt_dead_frames=int(np.sum(gt==pm.D)), gt_alive_frames=int(np.sum(gt!=pm.D)))
    score['boundaries_pair_1s'] = {}
    kinds = {'enter_dead': lambda t:t[2]==pm.D, 'exit_dead': lambda t:t[1]==pm.D,
             'ha_switch': lambda t:t[1] in (pm.H,pm.A) and t[2] in (pm.H,pm.A)}
    for name, select in kinds.items():
        offsets=[];ng=npred=missed=spurious=0
        for a,b in blocks:
            g=[t for t in pm.transitions(gt,[(a,b)]) if select(t)]
            p=[t for t in pm.transitions(pred,[(a,b)]) if select(t)]
            pairs, m, s=pm.match_transitions(g,p,int(round(fps)),typed='pair')
            offsets.extend((p[0]-g[0])/fps for g,p in pairs)
            ng+=len(g);npred+=len(p);missed+=len(m);spurious+=len(s)
        score['boundaries_pair_1s'][name]=dict(n_gt=ng,n_pred=npred,n_matched=len(offsets),n_missed=missed,n_spurious=spurious,
            mean_signed_s=float(np.mean(offsets)) if offsets else None,median_signed_s=float(np.median(offsets)) if offsets else None,
            mean_abs_s=float(np.mean(np.abs(offsets))) if offsets else None,n_late=sum(x>0 for x in offsets),n_early=sum(x<0 for x in offsets))
    for name, mask in visibility_strata(batch).items():
        cm=pm.scored_confusion(pm.confusion(gt[mask],pred[mask]))
        score['strata'][name].update(confusion=cm.tolist(), n_gt_dead=int(np.sum(gt[mask]==pm.D)),
            missed_dead=int(np.sum((gt[mask]==pm.D)&(pred[mask]!=pm.D))),
            home_head_wrong=int(np.sum(np.isin(gt[mask],[pm.H,pm.A]) & ((home[mask]>=.5)!=(gt[mask]==pm.H)))))
    return score
