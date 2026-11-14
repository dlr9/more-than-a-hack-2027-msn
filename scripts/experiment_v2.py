"""Fixed four-match grouped development ablation; never a final test or promotion."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from solution.external import prepare_skillcorner_match, prepare_metrica_game3
from solution.modeling import load_prepared, _weights, _truth_and_blocks
from solution.calibration import BinaryCalibration, probability_metrics
from solution.quality import dropout_windows, phase_coverage
from solution.decoder import emission_probabilities, viterbi_decode, observed_control

ROOT=Path('local_artifacts/external')
OUT=Path('local_artifacts/v2')
IDS=['1874553','1886347','1899585','1927964']
RADII=[1.,2.,3.,5.,10.]; WINDOWS=[.2,.5,1.]

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    audits=[]
    for match in IDS:
        folder=ROOT/'skillcorner'/match
        for view in ['all','observed','dropout']:
            output=OUT/f'{match}-{view}.npz'
            mask=None
            if view=='dropout':
                batch=load_prepared(OUT/f'{match}-observed.npz')
                mask=dropout_windows(batch['period'],float(batch['fps']))
            prepare_skillcorner_match(folder,output,RADII,WINDOWS,player_view='all' if view=='all' else 'observed',ball_dropout_mask=mask)
            print('prepared',match,view,flush=True)
        b=load_prepared(OUT/f'{match}-observed.npz')
        audits.append(phase_coverage(b,folder/f'{match}_dynamic_events.csv'))
    (OUT/'phase_audit.json').write_text(json.dumps(audits,indent=2))
    m=ROOT/'metrica'
    prepare_metrica_game3(m/'Sample_Game_3_metadata.xml',m/'Sample_Game_3_tracking.txt',m/'Sample_Game_3_events.json',OUT/'metrica.npz',RADII,WINDOWS)

def fit(batches,target,weak=False):
    xs,ys=[],[]
    for b in batches:
        y=b['raw_y_alive'] if weak else b[target]
        mask=(y>=0)&(b['period']>0)
        # Deterministic 2 Hz sampling reduces training cost, keeps match grouping.
        mask &= b['frame']%5==0
        xs.append(b['X'][mask]);ys.append(y[mask])
    x,y=np.vstack(xs),np.concatenate(ys)
    if set(np.unique(y))!={0,1}: raise ValueError('missing class')
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1,early_stopping=False,random_state=2027)
    model.fit(x,y,sample_weight=_weights(y))
    return model

def evaluate(variants=None, result_name="experiment_results.json"):
    variants = variants or ["all", "observed", "dropout"]
    cache={(m,v):load_prepared(OUT/f'{m}-{v}.npz') for m in IDS for v in ['all','observed','dropout']}
    metrica=load_prepared(OUT/'metrica.npz')
    rows=[]
    for index,test in enumerate(IDS):
        calibration=IDS[(index+1)%4]; train=[m for m in IDS if m not in [test,calibration]]
        for view in variants:
            training=[cache[m,'all' if view=='all' else 'observed'] for m in train]
            alive_training = list(training)
            if view in {'dropout', 'home_dropout_only'}: training += [cache[m,'dropout'] for m in train]
            if view == 'dropout': alive_training = training
            alive=fit(alive_training,'y_alive',weak=True) # Explicit legacy weak-label diagnostic only.
            home=fit(training,'y_home')
            cb=cache[calibration,'all' if view=='all' else 'observed']
            calibrators=[]
            for model,target in [(alive,'raw_y_alive'),(home,'y_home')]:
                mask=(cb[target]>=0)&(cb['period']>0)
                calibrators.append(BinaryCalibration().fit(model.predict_proba(cb['X'][mask])[:,1],cb[target][mask],training_ids=train,calibration_ids=[calibration]))
            for domain,b in [('skillcorner',cache[test,'all' if view=='all' else 'observed']),('dropout_stress',cache[test,'dropout']),('metrica_partial',metrica)]:
                pa,ph=alive.predict_proba(b['X'])[:,1],home.predict_proba(b['X'])[:,1]
                yh=b['y_home']; ya=b.get('raw_y_alive',b['y_alive'])
                hm=(yh>=0)&(b['period']>0); am=(ya>=0)&(b['period']>0)
                visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
                for calibrated in [False,True]:
                    a=calibrators[0].predict(pa) if calibrated else pa
                    h=calibrators[1].predict(ph) if calibrated else ph
                    probabilities=emission_probabilities(a,h)
                    _,blocks=_truth_and_blocks(b)
                    truth=np.full(len(ya),'X');truth[ya==0]='D';truth[(ya==1)&(yh==1)]='H';truth[(ya==1)&(yh==0)]='A'
                    valid=(truth!='X')&(b['period']>0)
                    row=dict(heldout=test,calibration=calibration,train=train,variant=view,domain=domain,calibrated=calibrated,alive_label_semantics='phase_complement_unverified' if domain!='metrica_partial' else 'partial_event_interiors',home=probability_metrics(yh[hm],h[hm]),alive=probability_metrics(ya[am],a[am]))
                    row['ha_missing_ball']=probability_metrics(yh[hm&~visible],h[hm&~visible])
                    row['ha_macro_f1']=float(f1_score(yh[hm],h[hm]>=.5,labels=[0,1],average='macro',zero_division=0))
                    for decoder in ['fixed','control_proxy']:
                        control=observed_control(b) if decoder=='control_proxy' else None
                        pred=viterbi_decode(probabilities,blocks,3.,1.,control=control)
                        row[decoder]=dict(partial_had_macro_f1=float(f1_score(truth[valid],pred[valid],labels=['H','A','D'],average='macro',zero_division=0)),per_class_f1=dict(zip(['H','A','D'],f1_score(truth[valid],pred[valid],labels=['H','A','D'],average=None,zero_division=0).tolist())),full_spell_status='not_tested_no_complete_independent_gt')
                    rows.append(row)
            print('evaluated',test,view,flush=True)
            (OUT/result_name).write_text(json.dumps(dict(protocol='fixed_4match_development_crossfit',promoted=False,rows=rows),indent=2))
    # CI resamples independent SkillCorner matches; repeated Metrica scores are NOT independent.
    rng=np.random.default_rng(2027);summary=[]
    for domain in ['skillcorner','dropout_stress']:
        for calibrated in [False,True]:
            for view in variants:
                group=[r for r in rows if r['domain']==domain and r['calibrated']==calibrated and r['variant']==view]
                vals=np.array([r['ha_macro_f1'] for r in group])
                ci=np.quantile(rng.choice(vals,(2000,len(vals)),replace=True).mean(axis=1),[.025,.975]).tolist()
                summary.append(dict(domain=domain,variant=view,calibrated=calibrated,ha_macro_f1_mean=float(vals.mean()),match_bootstrap_95_ci=ci,small_n=4))
    (OUT/('summary.json' if result_name=='experiment_results.json' else 'followup_summary.json')).write_text(json.dumps(summary,indent=2))

if __name__=='__main__':
    if '--followup' in sys.argv:
        evaluate(['home_dropout_only'], 'followup_results.json')
    else:
        if '--evaluate-only' not in sys.argv:prepare()
        evaluate()
