"""Typed observations and bounded residual edit proposals. All channels default off.

There is no fitted risk calibration in R14: a caller must supply a separately
validated risk model. This package does not accept OracleTruth or train on GT.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EvidencePacket:
    frame: int
    period: int
    modality: str
    observed_vs_inferred: str
    probability: float
    uncertainty: float
    provenance: str
    alignment_confidence: float
    model_revision: str
    def __post_init__(self):
        if self.frame<0 or self.period<=0 or self.observed_vs_inferred not in {'observed','inferred'} or not self.provenance or not self.model_revision:raise ValueError('evidence contract')
        if any(not np.isfinite(x) or not 0<=x<=1 for x in [self.probability,self.uncertainty,self.alignment_confidence]):raise ValueError('probability contract')

@dataclass(frozen=True)
class ContactEvidence:
    t: float
    player_tracklet: str
    touch_prob: float
    sustain_prob: float | None
    uncertainty: float
    source: str

@dataclass(frozen=True)
class WhistleEvidence:
    time: float
    confidence: float
    source: str
    alignment_confidence: float
    audio_quality: float

@dataclass(frozen=True)
class EditProposal:
    start: int
    end: int
    label: str
    evidence: EvidencePacket
    independent_dead_evidence: bool=False

def residual_candidates(packet,*,label,radius_frames=5):
    if label not in 'HAD' or radius_frames<0:raise ValueError('candidate')
    return EditProposal(max(0,packet.frame-radius_frames),packet.frame+radius_frames+1,label,packet)

def apply_edits(baseline,period,proposals,*,new_feature=False,trust=0.,model_off=True,calibrated_gate=None,max_edit_frames=25,max_changed_fraction=.01):
    if not new_feature or trust<=0 or model_off or calibrated_gate is None:return baseline
    p=np.asarray(baseline);period=np.asarray(period)
    if p.shape!=period.shape or p.ndim!=1 or not 0<trust<=1 or max_edit_frames<1 or not 0<=max_changed_fraction<=1:raise ValueError('edit budget')
    out=p.copy();used=np.zeros(len(p),bool);budget=int(len(p)*max_changed_fraction)
    for edit in proposals:
        a,b=edit.start,edit.end
        if not 0<=a<b<=len(p) or b-a>max_edit_frames or edit.label not in 'HAD' or not a<=edit.evidence.frame<b:continue
        if np.any(period[a:b]!=edit.evidence.period) or used[a:b].any():continue
        # Both inserting D and removing old D require separately sourced Dead evidence.
        if (edit.label=='D' or np.any(p[a:b]=='D')) and not edit.independent_dead_evidence:continue
        if not calibrated_gate(edit.evidence):continue
        change=p[a:b]!=edit.label
        if change.sum()>budget:continue
        out[a:b]=edit.label;used[a:b]=True;budget-=int(change.sum())
    return out
