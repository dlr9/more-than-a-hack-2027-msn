"""One-match fit / one-match calibration / third-match evaluation, fixed rotation.

This is a separate data-budget comparison, not a replacement of the two-match
LOMO baseline. Both experts refit only on fit_match; Home remains frozen R6-M2.
"""
import json,sys,time
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.calibration import BinaryCalibration
from solution.quality_experts import gate_mask
from solution.r7_metrics import extended
from solution.decoder import emission_probabilities
from solution.idsse import digest
from scripts.research_r7 import IDS,PARAMS,load,write_new
from scripts.research_r7b import prepared,baseline,decode,identity,record,ROOT

def sample(b):
    bins=np.floor(b['frame']*2/float(b['fps'])).astype(int)
    return (b['period']>0)&np.r_[True,np.diff(bins)>0]
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def mixture(b,players,ba,pa):
    # A fixed .5 logit blend once quality gate qualifies; no fitted gate.
    weight=.5*gate_mask(b,players)
    return 1/(1+np.exp(-((1-weight)*logit(ba)+weight*logit(pa))))
def main():
    out=ROOT/'calibration-v1';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,test in enumerate(IDS):
        fit=IDS[(i+1)%3];cal=IDS[(i+2)%3];path=out/f'{test}.joblib';started=time.monotonic()
        if path.with_suffix('.json').exists():
            meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['model_sha256'];bundle=joblib.load(path)
        else:
            if path.exists():raise FileExistsError('incomplete calibration receipt')
            x=[];y=[]
            for domain in ['natural','dropout']:
                b=load(fit,domain);s=sample(b);x.append(b['X'][s]);y.append(b['y_alive'][s])
            ba=HistGradientBoostingClassifier(**PARAMS).fit(np.vstack(x),np.concatenate(y))
            b=load(fit,'natural');s=sample(b);p=prepared(fit)
            pa=HistGradientBoostingClassifier(**PARAMS).fit(p['X'][s],b['y_alive'][s])
            cp=prepared(cal);ap=pa.predict_proba(cp['X'])[:,1];bs=[];ps=[];ys=[]
            for domain in ['natural','dropout']:
                b=load(cal,domain);s=sample(b);bs.append(ba.predict_proba(b['X'][s])[:,1]);ps.append(ap[s]);ys.append(b['y_alive'][s])
            cb=BinaryCalibration().fit(np.concatenate(bs),np.concatenate(ys),training_ids=[fit],calibration_ids=[cal])
            cpcal=BinaryCalibration().fit(np.concatenate(ps),np.concatenate(ys),training_ids=[fit],calibration_ids=[cal])
            bundle=dict(ball=ba,player=pa,ball_calibration=cb,player_calibration=cpcal,fit=[fit],calibration=[cal],test=test)
            joblib.dump(bundle,path)
            meta=dict(stage='calibrated_expert_fit',fit=[fit],calibration=[cal],test=test,model_sha256=digest(path),wall_s=time.monotonic()-started,**identity())
            write_new(path.with_suffix('.json'),meta);record(meta);print('CALIBRATION FIT',test,fit,cal,flush=True)
        players=prepared(test);pa=bundle['player'].predict_proba(players['X'])[:,1]
        for domain in ['natural','dropout','long']:
            b=load(test,domain);ba=bundle['ball'].predict_proba(b['X'])[:,1];h=baseline(test,domain)['p_home']
            candidates=[('G2_uncal',mixture(b,players,ba,pa)),('G2_cal',mixture(b,players,bundle['ball_calibration'].predict(ba),bundle['player_calibration'].predict(pa)))]
            for variant,a in candidates:
                resultpath=out/f'{test}-{variant}-{domain}.json'
                if resultpath.exists():rows.append(json.loads(resultpath.read_text()));continue
                pred=decode(a,h,b);arg=np.array(list('HAD'))[emission_probabilities(a,h).argmax(axis=1)]
                row=dict(match=test,fit=[fit],calibration=[cal],variant=variant,domain=domain,
                    metrics=dict(decoded=extended(b,a,h,pred),argmax=extended(b,a,h,arg)),model_sha256=digest(path))
                np.savez_compressed(resultpath.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred,argmax=arg)
                write_new(resultpath,row);rows.append(row);print('CALIBRATION SCORE',test,variant,domain,row['metrics']['decoded']['full_spell_f1'],flush=True)
    path=Path('docs/R7B_CALIBRATION_RESULTS.json')
    if not path.exists():write_new(path,dict(rows=rows,**identity(),budget='one fit match vs two in P1; same-training calibrated/uncalibrated contrast only'))
if __name__=='__main__':main()
