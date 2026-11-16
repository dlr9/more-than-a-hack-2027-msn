"""One pinned public PathCRF CPU compatibility attempt, never target inference."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import pickletools
import platform
import shutil
import time
import zipfile
import numpy as np
from r11.pathcrf_adapter import adapt_window, ContractError
from r11.pathcrf_runtime import ORIGINAL_COMMIT, WEIGHT_SHA256, load_original, forward


def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def synthetic():
    n=250;xy=np.zeros((n,22,2));xy[:,:11,0]=np.linspace(-48,10,11)
    xy[:,11:,0]=np.linspace(48,-10,11);xy[:,:,1]=np.tile(np.linspace(-25,25,11),2)
    xy[:,:,0]+=.2*np.arange(n)[:,None]/25
    return dict(frame=np.arange(n),period=np.ones(n),players_xy=xy,
        player_team=np.repeat([0,1],11),player_track_id=np.array([f'p{i:02}' for i in range(22)]),
        fps=25.,pitch_m=[105.,68.],goalkeeper_ids={0:'p00',1:'p11'})


def public_window(cache):
    # Fixed consumed public-other-game only; no source labels or ball loaded.
    path=pathlib.Path(cache)/'r6/prepared/J03WPY/raw.npz'
    with np.load(path,allow_pickle=False) as z:
        frame,period,xy,team,ids=[z[k] for k in ('frame','period','players_xy','player_team','player_track_id')]
    for start in range(0,min(len(frame)-250,25000),250):
        stop=start+250;visible=np.isfinite(xy[start:stop]).all(axis=2)
        active=visible.all(axis=0)
        if any(np.sum(active&(team==s))!=11 for s in (0,1)):continue
        if np.any(visible[:,~active]):continue
        gk={s:str(ids[np.flatnonzero(active&(team==s))[np.argmax(np.abs(xy[start:stop,active&(team==s),0].mean(axis=0)))]] ) for s in (0,1)}
        args=dict(frame=frame[start:stop],period=period[start:stop],players_xy=xy[start:stop],
            player_team=team,player_track_id=ids,fps=25.,pitch_m=[105.,68.],goalkeeper_ids=gk)
        try:window=adapt_window(**args)
        except ContractError:continue
        return args,window,dict(match='J03WPY',source_sha256=digest(path),
            frame_start=int(frame[start]),frame_end=int(frame[stop-1]),
            keeper_source='most_laterally_extreme_mean_x_per_team_10s_unverified_roles',
            exposure='AUTHOR_TRAIN_MATCH_AND_TEAM_CONSUMED_DIAGNOSTIC_ONLY')
    raise ValueError('no complete compatible public window in bounded first 25000 frames')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='local_artifacts/r11/pathcrf')
    ap.add_argument('--cache',default='local_artifacts')
    ap.add_argument('--receipt',required=True)
    args=ap.parse_args();root=pathlib.Path(args.root);out=pathlib.Path(args.receipt)
    if out.exists():raise FileExistsError('append-only receipt required')
    began=time.time();receipt=dict(model=ORIGINAL_COMMIT+'/saved/140/model/state_dict_best.pt',
        selected_versions=1,new_fits=0,official_score=None,full_spell_f1=None,
        target_access=False,synthetic_forward='NOT_RUN',public_forward='NOT_RUN',
        runtime_python=platform.python_version(),architecture=platform.machine())
    weight=root/'original140.pt';receipt['weight_sha256']=digest(weight)
    if receipt['weight_sha256']!=WEIGHT_SHA256:raise ValueError('weight digest mismatch')
    with zipfile.ZipFile(weight) as z:
        p=next(n for n in z.namelist() if n.endswith('data.pkl'))
        receipt['checkpoint_inspection']=dict(zip_members=len(z.namelist()),
            expanded_bytes=sum(x.file_size for x in z.infolist()),
            globals=sorted(set(arg for op,arg,_ in pickletools.genops(z.read(p)) if op.name=='GLOBAL')),
            pickle_executed=False)
    wheels=list(root.glob('torch-*.whl'))
    if wheels:
        with zipfile.ZipFile(wheels[0]) as z:expanded=sum(x.file_size for x in z.infolist())
        receipt['disk_guard']=dict(free_bytes=shutil.disk_usage(root).free,torch_expanded_bytes=expanded,
            required_minimum_bytes=2*expanded,installed=False)
    fixture=synthetic();syn=adapt_window(**fixture)
    receipt['synthetic_adapter']=dict(shape=list(syn['nodes'].shape),sha256=hashlib.sha256(syn['nodes'].tobytes()).hexdigest())
    data,pub,meta=public_window(args.cache);receipt['public_adapter']=dict(**meta,shape=list(pub['nodes'].shape),
        tensor_sha256=hashlib.sha256(pub['nodes'].tobytes()).hexdigest())
    perturb={}
    for name in ['slot_permutation','complete_outage','one_second_outage','unknown_goalkeeper','team_swap','frame_gap']:
        a={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()}
        if name=='slot_permutation':
            n=len(a['frame']);rng=np.random.default_rng(1101)
            a['player_team']=np.broadcast_to(a['player_team'],a['players_xy'].shape[:2]).copy()
            a['player_track_id']=np.broadcast_to(a['player_track_id'],a['players_xy'].shape[:2]).copy()
            for i in range(n):
                order=rng.permutation(a['players_xy'].shape[1])
                for k in ('players_xy','player_team','player_track_id'):a[k][i]=a[k][i,order]
        elif name=='complete_outage':a['players_xy'][:]=np.nan
        elif name=='one_second_outage':a['players_xy'][50:75]=np.nan
        elif name=='unknown_goalkeeper':a['goalkeeper_ids']=None
        elif name=='team_swap':
            a['player_team']=1-a['player_team'];a['goalkeeper_ids']={0:data['goalkeeper_ids'][1],1:data['goalkeeper_ids'][0]}
        elif name=='frame_gap':a['frame'][100:]+=1
        try:
            changed=adapt_window(**a);perturb[name]=dict(status='ACCEPT',tensor_equal=bool(np.array_equal(changed['nodes'],pub['nodes'])),left_team=changed['left_team'])
        except ContractError as e:perturb[name]=dict(status='REJECT',reason=str(e))
    receipt['public_contract_perturbations']=perturb
    # This is a real import/load/forward attempt. An unavailable torch runtime
    # is reported as such, without claiming load or neural forward occurred.
    try:
        model,torch=load_original(root)
        receipt['synthetic_forward']=forward(model,torch,syn)
        receipt['public_forward']=forward(model,torch,pub)
        receipt['decision']='RUNNABLE_DIAGNOSTIC_ONLY_NO_PROMOTION'
    except (ImportError,ModuleNotFoundError) as e:
        receipt['runtime_failure']=dict(type=type(e).__name__,message=str(e))
        receipt['decision']='NO_GO_PATHCRF_ENVIRONMENT_DISK_RUNTIME'
    receipt['elapsed_seconds']=time.time()-began
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
