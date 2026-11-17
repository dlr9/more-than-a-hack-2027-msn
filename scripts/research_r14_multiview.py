"""Deterministic synthetic matching/negative controls; no real-view claims."""
import argparse,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from multiview.contracts import View
from multiview.align import align,AlignmentConfig
from multiview.fingerprint import fingerprint
from multiview.fuse import fuse
from multiview.camera import camera_proxies
from solution.idsse import digest
from scripts.research_r7 import write_new
P=Path('docs/R14_MULTIVIEW_PROTOCOL.json')

def scene(t):
 rng=np.random.default_rng(1401);base=rng.uniform(.2,.8,(12,2));freq=rng.uniform(.12,.5,(12,2));phase=rng.uniform(0,6,(12,2))
 return base[None]+.04*np.sin(t[:,None,None]*freq[None]+phase[None])+np.stack([.05*np.sin(.17*t),.03*np.cos(.19*t)],1)[:,None]

def make_view(n=400,start=0.,drift=1.,missing=False,source='synthetic'):
 time=np.arange(n)/25;actual=start+drift*time;p=scene(actual);ball=np.stack([.5+.15*np.sin(.43*actual),.5+.1*np.cos(.37*actual)],1);observed=np.ones(n,bool)
 if missing:observed[100:180]=False;ball[~observed]=np.nan
 return View(source,25,np.arange(n),np.ones(n,int),time,p,np.broadcast_to(np.repeat([0,1],6),(n,12)).copy(),ball,observed,np.zeros(n,int),np.zeros(n,bool),np.ones(n,bool),np.zeros(n),actual,np.ones(n,bool))

def freeze():
 paths=['multiview/'+x for x in ['contracts.py','fingerprint.py','align.py','fuse.py','camera.py']]+['scripts/research_r14_multiview.py']
 write_new(P,dict(id='R14_MULTIVIEW_V1',seed=1401,source='synthetic scene only, not external video',config=AlignmentConfig().__dict__,scenarios=['self','offset_and_drift','team_swap','slot_permutation','replay','clock_wrong','shape_wrong_time','missing_balls_with_clock','missing_balls_without_clock','bad_homography','coordinate_conflict'],gate='correct accepted mapping within1 reference frame; no second anchor/replay/conflict must reject; observed inputs preserved; default off exact. Passing is engineering only, NEVER production calibration.',camera='fixed H, smoothly changing H and cut; center/pan dynamics and null fixed-camera signal; no real information gain claim',hashes={p:digest(p) for p in paths}))

def run(out='docs/R14_MULTIVIEW_RESULTS.json'):
 protocol=json.loads(P.read_text())
 for p,h in protocol['hashes'].items():assert digest(p)==h,p
 ref=make_view(600,missing=True,source='synthetic_reference');other=make_view(400,2,1.02,source='synthetic_other');rng=np.random.default_rng(1401)
 perm=rng.permutation(12);wrong=replace(other,players=np.roll(other.players,150,axis=0));missing=replace(other,ball=np.full_like(other.ball,np.nan),ball_observed=np.zeros(len(other.ball),bool))
 cases={'self':(ref,ref),'offset_and_drift':(ref,other),'team_swap':(ref,replace(other,teams=1-other.teams)),'slot_permutation':(ref,replace(other,players=other.players[:,perm],teams=other.teams[:,perm])),'replay':(ref,replace(other,replay=np.ones(len(other.frame),bool))),'clock_wrong':(ref,replace(other,clock_s=other.clock_s+3)),'shape_wrong_time':(ref,wrong),'missing_balls_with_clock':(replace(ref,ball=np.full_like(ref.ball,np.nan),ball_observed=np.zeros(len(ref.ball),bool)),missing),'missing_balls_without_clock':(replace(ref,ball=np.full_like(ref.ball,np.nan),ball_observed=np.zeros(len(ref.ball),bool),clock_s=None),replace(missing,clock_s=None)),'bad_homography':(ref,replace(other,reprojection_error=np.full(len(other.frame),100.))),'coordinate_conflict':(ref,replace(other,ball=other.ball+.1))}
 rows=[]
 for name,(a,b) in cases.items():
  began=time.monotonic();alignment=align(a,b);idx=alignment['index'];selected=np.flatnonzero(idx>=0);error=np.abs(a.clock_s[selected]-b.clock_s[idx[selected]])*25 if len(selected) and a.clock_s is not None and b.clock_s is not None else np.array([])
  fused=fuse(a,b,alignment,enabled=True,validated=True);off=fuse(a,b,alignment);assert off['ball'] is a.ball
  np.testing.assert_array_equal(fused['ball'][a.ball_observed],a.ball[a.ball_observed])
  if name in ['replay','clock_wrong','missing_balls_without_clock','bad_homography']:assert len(selected)==0,name
  rows.append(dict(case=name,accepted=int(len(selected)),mapping_coverage=float(len(selected)/len(a.frame)),false_alignment_rate=float(np.mean(error>1)) if len(error) else None,max_time_error_frames=float(error.max()) if len(error) else None,coverage_gain=fused['coverage_gain'],conflict_frames=int(fused['conflict'].sum()),intersection_ball_error_median=float(np.median(fused.get('intersection_ball_error_m',[]))) if len(fused.get('intersection_ball_error_m',[])) else None,shots=alignment['shots'],wall_s=time.monotonic()-began))
  print(name,rows[-1]['accepted'],rows[-1]['false_alignment_rate'],rows[-1]['coverage_gain'],flush=True)
 n=200;H=np.broadcast_to(np.diag([1920.,1080.,1.]),(n,3,3)).copy();common=(np.ones(n,bool),np.zeros(n),np.arange(n),np.ones(n,int),np.zeros(n,int),25,1920,1080)
 fixed=camera_proxies(H,*common);H[:,0,2]=np.arange(n)*2.;pan=camera_proxies(H,*common);H[100:,0,2]+=700;cut=camera_proxies(H,*common)
 assert np.nanmax(abs(fixed['camera_pan_proxy_pxps']))==0;assert np.allclose(pan['camera_pan_proxy_pxps'][1:,0],50);assert cut['cut_suspected'][100]
 write_new(out,dict(source='SYNTHETIC_ONLY',rows=rows,camera=dict(fixed_pan_max=0.,known_pan_pxps=50.,cut_detected=bool(cut['cut_suspected'][100]),real_ball_prediction_information='NOT_TESTED'),decision='NO_EVIDENCE_TO_PROMOTE_REAL_MULTIVIEW',protocol_sha256=digest(P)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--out',default='docs/R14_MULTIVIEW_RESULTS.json');a=p.parse_args();freeze() if a.action=='freeze' else run(a.out)
