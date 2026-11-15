"""Read-only diagnosis of the frozen public SkillCorner transfer, never target data."""
import argparse,json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution.evidence import _runs, visibility_strata
from solution.quality import period_blocks
from solution.decoder import emission_probabilities
from solution.r7_skillcorner import read_positions
from solution.idsse import digest
from scripts.research_r7 import write_new

def summary(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
                quantiles=np.quantile(x,[0,.1,.5,.9,1]).tolist() if len(x) else [])

def main(output='docs/R7B_MISSING_DIAGNOSIS.json'):
    root=Path('local_artifacts/r7/transfer-v2')
    b=dict(np.load(root/'prepared.npz',allow_pickle=False))
    old=dict(np.load('local_artifacts/v2/1874553-observed.npz',allow_pickle=False))
    raw,_=read_positions('local_artifacts/external/skillcorner/1874553')
    names=b['feature_names'].tolist();visible=b['X'][:,names.index('ball_visible')]==1
    for k in ['frame','period']:
        assert np.array_equal(raw[k],b[k]) and np.array_equal(old[k],b[k]),k
    for k in ['y_alive','y_home']:assert np.array_equal(old[k],b[k]),k
    assert np.array_equal(visible,np.isfinite(raw['ball_xy']).all(axis=1))
    known=(b['y_alive']==1)&(b['y_home']>=0)
    assert np.array_equal(old['raw_y_home'][known],b['y_home'][known])
    assert np.array_equal(old['phase_y_home'][known],b['y_home'][known])
    target=known&visibility_strata(b)['missing_long'];fps=float(b['fps'])
    predictions={v:dict(np.load(root/f'{v}.npz'))for v in ['M1','M2','M3','M4','M5','M6']}
    episodes=[]
    for lo,hi in period_blocks(b['period']):
        for start,stop in _runs(~visible[lo:hi+1]):
            start+=lo;stop+=lo;k=target[start:stop]
            if not k.any():continue
            idx=np.arange(start,stop)[k];duration=(stop-start)/fps
            row=dict(start_frame=int(b['frame'][start]),stop_frame_exclusive=int(b['frame'][stop-1])+1,
                     period=int(b['period'][start]),duration_s=duration,n_known_ha=int(k.sum()),
                     known_runs=[(int(start+a),int(start+c))for a,c in _runs(k)],
                     duration_bin='0.5-2' if duration<=2 else '2-5' if duration<=5 else '5-10' if duration<=10 else '>10',
                     home_labels=int(np.sum(b['y_home'][idx]==1)),away_labels=int(np.sum(b['y_home'][idx]==0)),
                     detected_players=summary(np.isfinite(raw['players_xy'][idx]).all(axis=2).sum(axis=1)),models={})
            for v,p in predictions.items():
                probs=emission_probabilities(p['p_alive'],p['p_home']);arg=np.argmax(probs,axis=1)
                context={}
                for side,j in [('previous',start-1),('next',stop)]:
                    context[side]=dict(p_alive=float(p['p_alive'][j]),p_home=float(p['p_home'][j]),decoded=str(p['decoded'][j])) if lo<=j<=hi else None
                row['models'][v]=dict(p_alive=summary(p['p_alive'][idx]),p_home=summary(p['p_home'][idx]),
                    argmax_dead_rate=float(np.mean(arg[idx]==2)),viterbi_dead_rate=float(np.mean(p['decoded'][idx]=='D')),
                    home_wrong_rate=float(np.mean((p['p_home'][idx]>=.5)!=b['y_home'][idx])),context=context)
            episodes.append(row)
    features={name:dict(missing_rate=float(np.mean(~np.isfinite(b['X'][target,i]))),values=summary(b['X'][target,i]))for i,name in enumerate(names)}
    aggregate={}
    for v,p in predictions.items():
        probs=emission_probabilities(p['p_alive'],p['p_home'])
        aggregate[v]=dict(p_alive=summary(p['p_alive'][target]),p_home=summary(p['p_home'][target]),
            argmax_dead=int(np.sum(np.argmax(probs[target],axis=1)==2)),viterbi_dead=int(np.sum(p['decoded'][target]=='D')),
            joint_mean=probs[target].mean(axis=0).tolist())
    result=dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=digest(__file__),
        inputs={str(p):digest(p)for p in [root/'prepared.npz',Path('local_artifacts/v2/1874553-observed.npz')]},
        n_frames=int(target.sum()),n_episodes=len(episodes),n_matches=1,n_known_ha=int(known.sum()),
        alignment='exact frame/period/labels/ball-finiteness equality; strict raw possession and phase agreement checked',
        limitation='Partial source proxy labels; one previously examined match; selection is not independent of source possession availability. No true D evaluation.',
        models=aggregate,episodes=episodes,features=features)
    write_new(output,result)
    print(json.dumps({k:result[k]for k in ['n_frames','n_episodes','models']},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='docs/R7B_MISSING_DIAGNOSIS.json');args=parser.parse_args();main(args.output)
