"""Auditable spell-level effect of a proposed change on OTHER-match labels.

Only research diagnostics. This does NOT tune, select or run on the competition's
public five-minute ground truth, private GT, or restricted target assets.

The official scorer is the authoritative specification. This adapter reuses official matching and scoring. Compare
this adapter's score() with evaluation.possession_metrics.spell_scores using real
OTHER-match, complete labels. The official source must remain byte-identical.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from numbers import Integral
from evaluation import possession_metrics as pm

IOUS = pm.SPELL_IOUS
VALID = {"H", "A", "D", "X"}

@dataclass(frozen=True)
class Score:
    spell_f1_mean: float
    f1_by_iou: tuple[float, ...]
    n_gt: int
    n_pred: int
    matched_by_iou: tuple[int, ...]

    def as_dict(self):
        return {
            "spell_f1_mean": self.spell_f1_mean,
            "f1_by_iou": dict(zip(IOUS, self.f1_by_iou)),
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "matched_by_iou": dict(zip(IOUS, self.matched_by_iou)),
        }


def _validate(truth: Iterable[str], pred: Iterable[str]):
    a, b = list(truth), list(pred)
    if len(a) != len(b):
        raise ValueError("truth and prediction length differ")
    if any(x not in VALID for x in a+b):
        raise ValueError("invalid H/A/D/X label")
    return a,b


def _scored_runs(truth: list[str]):
    starts = []
    start = None
    for i in range(len(truth) + 1):
        alive = i < len(truth) and truth[i] != "X"
        if alive and start is None:
            start=i
        if not alive and start is not None:
            starts.append((start,i));start=None
    return starts


def _prepared(truth, prediction, *, blocks=None):
    gt,pred=_validate(truth,prediction)
    scored=_scored_runs(gt)
    if blocks is None:
        blocks=scored
    else:
        supplied=list(blocks);previous=0
        for a,b in supplied:
            if (not isinstance(a,Integral) or not isinstance(b,Integral) or
                    not 0<=a<b<=len(gt) or a<previous):
                raise ValueError("sorted disjoint half-open blocks required")
            previous=b
        blocks=[(max(a,c),min(b,d)) for a,b in supplied for c,d in scored if max(a,c)<min(b,d)]
    return pm.label_codes(gt), pm.fill_abstentions(pm.label_codes(pred), blocks), blocks


def score(truth, prediction, *, blocks=None) -> Score:
    gt, pred, runs = _prepared(truth, prediction, blocks=blocks)
    result = pm.spell_scores(gt, pred, runs)
    return Score(result['mean_f1'], tuple(result['per_iou'][t]['f1'] for t in IOUS),
                 result['n_gt'], result['n_pred'],
                 tuple(result['per_iou'][t]['n_matched'] for t in IOUS))


def _matched(truth, prediction, blocks):
    gt, pred, runs = _prepared(truth, prediction, blocks=blocks)
    spells = pm.possession_spells(gt, runs)
    overlaps = pm.spell_overlaps(spells, pm.possession_spells(pred, runs))
    return {t: {tuple(spells[i]) for i, flag in enumerate(
        pm.matched_gt_flags(overlaps, len(spells), t)) if flag} for t in IOUS}


def audit_edit(truth:Iterable[str], baseline:Iterable[str],
               start:int, end:int, replacement:str, *, blocks=None)->dict:
    """EXTERNAL LABELS ONLY. Counterfactual effect; never a deployable model.

    Assesses a single interval change and its exact five-IoU spell consequence.
    Candidate interval/times must be selected by label-free physical evidence.
    The provided GT MUST NOT be used to generate/edit proposals or choose a
    target model version. This evaluates a previously frozen proposal only.
    """
    gt,pred=_validate(truth,baseline)
    if replacement not in ("H","A","D","X"):
        raise ValueError("bad replacement")
    if not (isinstance(start, Integral) and isinstance(end, Integral) and 0<=start<end<=len(pred)):
        raise ValueError("edit interval out of range")
    blocks=list(blocks) if blocks is not None else None
    before=score(gt,pred,blocks=blocks)
    after_pred=pred.copy();after_pred[start:end]=[replacement]*(end-start)
    after=score(gt,after_pred,blocks=blocks)
    old, new = _matched(gt, pred, blocks), _matched(gt, after_pred, blocks)
    def describe(items):
        return [dict(start=int(a), end=int(b), team=pm.LABELS[k]) for a,b,k in sorted(items)]
    changes = {t: dict(gained=describe(new[t]-old[t]), lost=describe(old[t]-new[t])) for t in IOUS}
    return {"matches_by_iou": changes, "candidate":dict(start=start,end=end,replacement=replacement),
            "baseline":before.as_dict(),"edited":after.as_dict(),
            "delta_spellF1":after.spell_f1_mean-before.spell_f1_mean,
            "delta_matched_iou_05":after.matched_by_iou[2]-before.matched_by_iou[2],
            "delta_pred_spells":after.n_pred-before.n_pred}
