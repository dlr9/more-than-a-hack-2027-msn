"""Predeclared single-match SkillCorner partial H/A diagnostic, no training or full-spell."""
import json,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.r7_skillcorner import read_positions
from solution.provider_neutral import FeatureContract,assert_compatible,provider_neutral_features_v3
from solution.r7_features import project
from solution.modeling import load_prepared
from solution.evidence import labels_from_batch,score_sequence,period_reference,visibility_strata
from solution.calibration import probability_metrics
from solution.decoder import emission_probabilities,viterbi_decode
from solution.quality import period_blocks
from scripts.research_r7 import model_path,VARIANTS,write_new,record,source_identity
from solution.idsse import digest

def main():
    out=Path('local_artifacts/r7/transfer-v2')
    if out.exists():raise FileExistsError('preserve cross-provider result')
    protocol=json.loads(Path('docs/R7_TRANSFER_PROTOCOL.json').read_text())
    for p,sha in protocol['inputs'].items():
        if digest(p)!=sha:raise ValueError('transfer input drift')
    out.mkdir(parents=True);started=time.monotonic()
    raw,pitch=read_positions('local_artifacts/external/skillcorner/1874553')
    contract=FeatureContract('SkillCorner','observed',10.,pitch=pitch)
    base=provider_neutral_features_v3(raw,contract)
    old=load_prepared('local_artifacts/v2/1874553-observed.npz')
    assert np.array_equal(old['period'],base['period'])
    base.update(y_alive=old['y_alive'],y_home=old['y_home'],match_id=np.asarray('1874553'),label_semantics=np.asarray('strict_partial_SC_phase_team_agreement'))
    np.savez_compressed(out/'prepared.npz',**base);truth=labels_from_batch(base);rows=[]
    for v in ['M1','M2','M3','M4','M5','M6']:
        model=joblib.load(model_path('J03WPY',v));b=project(base,VARIANTS[v][0])if v in VARIANTS else base
        waiver=assert_compatible([FeatureContract(**c)for c in model.metadata['training_contracts']],contract,purpose='evaluate',allow_cross_provider=True,allow_cross_view=True,allow_cross_fps=True,allow_cross_pitch=True)
        x=pd.DataFrame(b['X'],columns=b['feature_names'].tolist())
        if list(x.columns)!=model.feature_names:raise ValueError('transfer schema names mismatch')
        # Explicit waiver is diagnostic-only. Production Candidate gates remain unchanged.
        a,h=model.alive.predict_proba(x)[:,1],model.home.predict_proba(x)[:,1]
        visible=base['X'][:,base['feature_names'].tolist().index('ball_visible')]==1
        scale=np.where(visible&np.r_[True,visible[:-1]],1.,4.)if v=='M3'else None
        pred=viterbi_decode(emission_probabilities(a,h),period_blocks(base['period']),3.,1.,dead_scale=scale)
        score=score_sequence(truth,pred,base['period'],10.,semantics='partial_SC');strata={}
        for name,mask in visibility_strata(base).items():
            keep=mask&(base['y_home']>=0)&(base['y_alive']==1);n=int(keep.sum())
            strata[name]=dict(n_known_ha=n,false_dead=int(np.sum(pred[keep]=='D')),false_dead_rate=float(np.mean(pred[keep]=='D'))if n else None,
                decoded_ha_accuracy=float(np.mean(pred[keep]==truth[keep]))if n else None,home=probability_metrics(base['y_home'][keep],h[keep]))
        rows.append(dict(variant=v,partial_metrics=score,strata=strata,waiver=waiver,full_spell_f1=None))
        np.savez_compressed(out/f'{v}.npz',p_alive=a,p_home=h,decoded=pred)
        print('partial transfer',v,score['partial_macro_f1'],flush=True)
    result=dict(experiment_id='E07_cross_provider_partial',**source_identity(),transfer_script_sha256=digest(__file__),adapter_sha256=digest('solution/r7_skillcorner.py'),transfer_protocol_sha256=digest('docs/R7_TRANSFER_PROTOCOL.json'),rows=rows,reference=score_sequence(truth,period_reference(base),base['period'],10.,semantics='partial_SC'),full_spell_f1=None,wall_s=time.monotonic()-started,decision='diagnostic only; no selection, no D ground truth',competition_promoted=False)
    write_new(out/'results.json',result);write_new('docs/R7_TRANSFER_RESULTS.json',result);record({k:v for k,v in result.items()if k not in{'rows','reference'}})
if __name__=='__main__':main()
