"""Sparse mutual shape matches, shot-local affine clock fit and refusal.

Prototype constants are synthetic engineering tolerances, NOT calibrated trust.
Two anchor families are mandatory: shape and independent clock or visible ball.
No DTW path is forced through a replay, gap or ambiguous repeated formation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .fingerprint import fingerprint

@dataclass(frozen=True)
class AlignmentConfig:
    max_shape_distance: float=.025
    ambiguity_margin: float=.0001
    max_reprojection_px: float=2.
    min_pairs: int=4
    max_clock_error_s: float=.08
    max_ball_error_m: float=2.
    max_fit_residual_s: float=.12
    drift_range: tuple=(.8,1.25)

def align(reference,other,config=AlignmentConfig()):
    if reference.pitch_m!=other.pitch_m:raise ValueError('pitch mismatch')
    a=fingerprint(reference.players,reference.teams);b=fingerprint(other.players,other.teams)
    def eligible(v,f):return np.isfinite(f).all(1)&v.homography_valid&np.isfinite(v.reprojection_error)&(v.reprojection_error>=0)&(v.reprojection_error<=config.max_reprojection_px)&~v.replay&(v.period>0)
    va=eligible(reference,a);vb=eligible(other,b)
    mapping=np.full(len(a),-1,int);confidence=np.zeros(len(a));records=[];last={}
    # Split on explicit shot, period AND frame discontinuity; do not conflate reused IDs.
    cut=np.r_[0,np.flatnonzero((other.shot[1:]!=other.shot[:-1])|(other.period[1:]!=other.period[:-1])|(np.diff(other.frame)!=1))+1,len(b)]
    for lo,hi in zip(cut[:-1],cut[1:]):
        record=dict(start=int(lo),end=int(hi),shot=int(other.shot[lo]),period=int(other.period[lo]),accepted=False,reason='insufficient_shape')
        records.append(record);ia=np.flatnonzero(va&(reference.period==other.period[lo]));ib=np.arange(lo,hi)[vb[lo:hi]]
        if len(ia)<2 or len(ib)<config.min_pairs:continue
        dist,idx=cKDTree(a[ia]).query(b[ib],k=2);rd,ri=cKDTree(b[ib]).query(a[ia],k=1)
        keep=(dist[:,0]<=config.max_shape_distance)&((dist[:,1]-dist[:,0])>=config.ambiguity_margin)&(ri[idx[:,0]]==np.arange(len(ib)))
        j=ib[keep];i=ia[idx[keep,0]];ds=dist[keep,0]
        if len(i)<config.min_pairs:continue
        order=np.argsort(j);i,j,ds=i[order],j[order],ds[order]
        if np.any(np.diff(i)<=0):record['reason']='nonmonotonic_or_replay';continue
        period=int(other.period[lo])
        if len(i) and i[0]<=last.get(period,-1):record['reason']='replay_or_reordered_shot';continue
        anchors=np.zeros(len(i),bool);clock_n=0;ball_n=0
        if reference.clock_s is not None and other.clock_s is not None:
            c1=reference.clock_s[i];c2=other.clock_s[j];known=np.isfinite(c1)&np.isfinite(c2)
            conflict=known&(np.abs(c1-c2)>config.max_clock_error_s)
            if conflict.sum()>max(0,int(.1*known.sum())):
                record['reason']='independent_clock_conflict';continue
            ok=known&~conflict;anchors|=ok;clock_n=int(ok.sum())
        common=reference.ball_observed[i]&other.ball_observed[j]
        if common.any():
            e=np.linalg.norm((reference.ball[i]-other.ball[j])*reference.pitch_m,axis=1);ok=common&np.isfinite(e)&(e<=config.max_ball_error_m);anchors|=ok;ball_n=int(ok.sum())
        record.update(clock_anchors=clock_n,ball_anchors=ball_n)
        if anchors.sum()<config.min_pairs:record['reason']='no_second_anchor';continue
        x=other.time_s[j];y=reference.time_s[i];slope,offset=np.polyfit(x,y,1)
        residual=np.abs(y-(slope*x+offset));good=residual<=config.max_fit_residual_s
        if not config.drift_range[0]<=slope<=config.drift_range[1] or good.sum()<config.min_pairs or np.mean(good)<.9:
            record['reason']='clock_fit_failed';continue
        # Only cycle-consistent matches inside independent-anchor time support.
        good&=(j>=j[anchors].min())&(j<=j[anchors].max())
        if good.sum()<config.min_pairs:record['reason']='insufficient_anchor_support';continue
        i,j,ds=i[good],j[good],ds[good];q=np.exp(-ds/config.max_shape_distance)
        free=mapping[i]<0;i,j,q=i[free],j[free],q[free]
        mapping[i]=j;confidence[i]=q
        if len(i):last[period]=int(i[-1])
        record.update(accepted=True,reason='synthetic_tolerance_pass_not_calibrated',pairs=int(len(i)),shift_s=float(offset),drift=float(slope),max_residual_s=float(residual[good].max()))
    return dict(index=mapping,confidence=confidence,shots=records,calibrated=False)
