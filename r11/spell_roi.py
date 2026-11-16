"""Auditable spell-level effect of a proposed change on OTHER-match labels.

Only research diagnostics. This does NOT tune, select or run on the competition's
public five-minute ground truth, private GT, or restricted target assets.

The official scorer is the authoritative specification. For integration, compare
this scorer's score() with evaluation.possession_metrics.spell_scores using real
OTHER-match, complete labels. The official source must remain byte-identical.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from numbers import Integral

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
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


def _fill_abstentions(pred: list[str], blocks: list[tuple[int,int]]):
    result=pred.copy()
    for a,b in blocks:
        known=[i for i in range(a,b) if result[i] != "X"]
        if not known:
            continue
        last=result[known[0]]
        for i in range(a,b):
            if result[i]=="X":
                result[i]=last
            else:
                last=result[i]
    return result


def _spells(labels:list[str], blocks:list[tuple[int,int]]):
    spells=[]
    for a,b in blocks:
        last=a
        for i in range(a+1,b+1):
            if i==b or labels[i]!=labels[last]:
                if labels[last] in ("H","A"):
                    spells.append((last,i,labels[last]))
                last=i
    return spells


def score(truth: Iterable[str], prediction: Iterable[str], *, blocks=None) -> Score:
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
    # The official temporal evaluator fills X inside each graded run.
    filled=_fill_abstentions(pred,blocks)
    g=_spells(gt,blocks); p=_spells(filled,blocks)
    overlaps=[]
    for gi,(gs,ge,team) in enumerate(g):
        for pi,(ps,pe,pt) in enumerate(p):
            if team!=pt or pe<=gs or ge<=ps:
                continue
            i=min(ge,pe)-max(gs,ps)
            union=max(ge,pe)-min(gs,ps)
            overlaps.append((i/union,gi,pi))
    overlaps.sort(key=lambda row:(-row[0],row[1],row[2]))
    f1=[]; match=[]
    for thr in IOUS:
        used_g=set();used_p=set()
        for iou,gi,pi in overlaps:
            if iou<thr:break
            if gi not in used_g and pi not in used_p:
                used_g.add(gi);used_p.add(pi)
        n=len(used_g)
        match.append(n)
        precision=n/float(len(p)) if p else 0.
        recall=n/float(len(g)) if g else 0.
        f1.append(2.*precision*recall/(precision+recall) if precision+recall else 0.)
    return Score(sum(f1)/len(f1),tuple(f1),len(g),len(p),tuple(match))


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
    if not (0<=start<end<=len(pred)):
        raise ValueError("edit interval out of range")
    blocks=list(blocks) if blocks is not None else None
    before=score(gt,pred,blocks=blocks)
    after_pred=pred.copy();after_pred[start:end]=[replacement]*(end-start)
    after=score(gt,after_pred,blocks=blocks)
    return {"candidate":dict(start=start,end=end,replacement=replacement),
            "baseline":before.as_dict(),"edited":after.as_dict(),
            "delta_spellF1":after.spell_f1_mean-before.spell_f1_mean,
            "delta_matched_iou_05":after.matched_by_iou[2]-before.matched_by_iou[2],
            "delta_pred_spells":after.n_pred-before.n_pred}
