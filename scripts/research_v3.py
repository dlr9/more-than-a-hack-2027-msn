"""Frozen E1/E2/E3 diagnostics on public OTHER matches; no target or sealed-test access."""
import hashlib,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared,_weights
from solution.calibration import BinaryCalibration,probability_metrics
from solution.candidate import Candidate,unweight_probability
from solution.decoder import emission_probabilities,viterbi_decode
from solution.quality import period_blocks
from solution.evidence import labels_from_batch,score_sequence,visibility_strata,error_attribution

ROOT=Path('local_artifacts/v2');OUT=Path('local_artifacts/v3')
IDS=['1874553','1886347','1899585','1927964']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def predict(model,b):return model.predict_proba(pd.DataFrame(b['X'],columns=b['feature_names'].astype(str)))[:,1]

def fit(batches,target,*,missing_only=False):
    xs,ys=[],[]
    names=batches[0]['feature_names'].astype(str).tolist()
    for b in batches:
        assert b['feature_names'].astype(str).tolist()==names
        y=b[target];mask=(y>=0)&(b['period']>0)&(b['frame']%5==0)
        if missing_only:mask &= b['X'][:,names.index('ball_visible')]!=1
        xs.append(b['X'][mask]);ys.append(y[mask])
    x,y=np.vstack(xs),np.concatenate(ys)
    if set(np.unique(y))!={0,1}:raise ValueError('insufficient classes')
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=2027)
    model.fit(pd.DataFrame(x,columns=names),y,sample_weight=_weights(y))
    return model,dict(n=len(y),class_counts=np.bincount(y,minlength=2).tolist(),positive_prior=float(y.mean()))

def diagnostics(b,pa,ph,pred):
    truth=labels_from_batch(b);yh=b['y_home'];ya=b.get('raw_y_alive',b['y_alive'])
    hm=(yh>=0)&(b['period']>0);am=(ya>=0)&(b['period']>0)
    semantics='metrica_partial_event_proxy' if str(b['match_id'])=='metrica-game-3' else 'skillcorner_strict_weak'
    result=dict(primary=score_sequence(truth,pred,b['period'],float(b['fps']),semantics=semantics),home=probability_metrics(yh[hm],ph[hm]),alive_proxy=probability_metrics(ya[am],pa[am]),attribution=error_attribution(truth,pa,ph,pred),strata={})
    result['ha_macro_f1']=float(f1_score(yh[hm],ph[hm]>=.5,labels=[0,1],average='macro',zero_division=0))
    result['probability_quantiles']=dict(alive=np.quantile(pa[am],[0,.1,.5,.9,1]).tolist(),home=np.quantile(ph[hm],[0,.1,.5,.9,1]).tolist())
    for name,mask in visibility_strata(b).items():
        m=mask&hm
        result['strata'][name]=dict(home=probability_metrics(yh[m],ph[m]),ha_error=int(np.sum((ph[m]>=.5)!=yh[m])),decoded_ha_error=int(np.sum(pred[m]!=np.where(yh[m]==1,'H','A'))),n=int(m.sum()))
    if 'raw_y_alive' in b:
        result['phase_proxy_sensitivity']=score_sequence(labels_from_batch(b,'phase_proxy'),pred,b['period'],float(b['fps']),semantics='skillcorner_phase_complement_unverified')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report_path=OUT/'research_results.json'
    if report_path.exists():raise RuntimeError('run directory already has results; preserve history')
    protocol=json.loads(Path('docs/evaluation_protocol_v3.json').read_text())
    assert protocol['status']=='frozen_before_new_experiments'
    cache={(m,v):load_prepared(ROOT/f'{m}-{v}.npz')for m in IDS for v in ['observed','dropout']}
    metrica=load_prepared(ROOT/'metrica.npz')
    inputs=[ROOT/f'{m}-{v}.npz'for m in IDS for v in ['observed','dropout']]+[ROOT/'metrica.npz']
    manifest=dict(protocol_sha256=digest('docs/evaluation_protocol_v3.json'),inputs=[dict(path=str(p),sha256=digest(p))for p in inputs],code={str(p):digest(p)for p in [Path(__file__),Path('solution/evidence.py'),Path('solution/candidate.py')]},models=[])
    rows=[]
    history=json.loads(Path('docs/external_validation_v2.json').read_text())['rows']
    for i,heldout in enumerate(IDS):
        cal=IDS[(i+1)%4];train=[m for m in IDS if m not in [heldout,cal]]
        alive,alive_training=fit([cache[m,'observed']for m in train],'raw_y_alive')
        home,home_training=fit([cache[m,'observed']for m in train],'y_home')
        expert,expert_training=fit([cache[m,v]for m in train for v in ['observed','dropout']],'y_home',missing_only=True)
        cb=cache[cal,'observed'];mask=(cb['raw_y_alive']>=0)&(cb['period']>0)
        calibrator=BinaryCalibration().fit(predict(alive,cb)[mask],cb['raw_y_alive'][mask],training_ids=train,calibration_ids=[cal])
        bundle=Candidate(alive,home,cb['feature_names'].astype(str).tolist(),dict(schema_version=2,observation_view='observed',train=train,calibration=cal,heldout=heldout,promoted=False,alive_label_policy='phase_complement_unverified',protocol=protocol['version'],decoder_time_normalized=False))
        folder=OUT/heldout;folder.mkdir(exist_ok=True)
        joblib.dump(bundle,folder/'baseline.joblib');bundle.missing_expert=expert;joblib.dump(bundle,folder/'missing_expert.joblib');bundle.missing_expert=None
        joblib.dump(calibrator,folder/'alive_calibration.joblib')
        manifest['models'].append(dict(heldout=heldout,train=train,calibration=cal,alive_training=alive_training,home_training=home_training,expert_training=expert_training,baseline_sha256=digest(folder/'baseline.joblib'),expert_sha256=digest(folder/'missing_expert.joblib')))
        for domain,b in [('natural',cache[heldout,'observed']),('dropout_stress',cache[heldout,'dropout']),('metrica',metrica)]:
            pa,ph=predict(alive,b),predict(home,b)
            missing=b['X'][:,b['feature_names'].tolist().index('ball_visible')]!=1
            pe=ph.copy()
            if missing.any():pe[missing]=predict(expert,b)[missing]
            assert np.array_equal(ph[~missing],pe[~missing])
            pc=unweight_probability(pa,alive_training['positive_prior']);pl=calibrator.predict(pa)
            blocks=period_blocks(b['period']);fps=float(b['fps'])
            variants=[('baseline',pa,ph,1.),('E1_prior',pc,ph,1.),('E1_platt',pl,ph,1.),('E2_missing_expert',pa,pe,1.),('E3_time_normalized',pa,ph,fps/10)]
            probabilities={}
            for variant,a,h,scale in variants:
                p=emission_probabilities(a,h);pred=viterbi_decode(p,blocks,3.*scale,1.*scale)
                row=dict(heldout=heldout,calibration=cal,train=train,domain=domain,variant=variant,diagnostics=diagnostics(b,a,h,pred))
                if variant=='baseline' and domain=='natural':
                    old=next(r for r in history if r['heldout']==heldout and r['variant']=='observed' and r['domain']=='skillcorner' and not r['calibrated'])
                    assert abs(old['ha_macro_f1']-row['diagnostics']['ha_macro_f1'])<1e-12,'baseline replay drift'
                    row['v2_home_macro_replay_exact']=True
                rows.append(row);probabilities[variant+'_labels']=pred
            np.savez_compressed(folder/f'{domain}-probabilities.npz',alive=pa,home=ph,home_expert=pe,alive_prior=pc,alive_platt=pl,**probabilities)
        result=dict(protocol=protocol['version'],promoted=False,manifest=manifest,rows=rows)
        report_path.write_text(json.dumps(result,separators=(',',':'))+'\n')
        print('completed fixed fold',heldout,flush=True)
    print('completed',len(rows),'diagnostic groups; no automatic promotion',flush=True)

if __name__=='__main__':main()
