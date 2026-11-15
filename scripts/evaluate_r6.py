"""Shared full-period DFL-source evaluation. Never reads competition ground truth."""
import json
import numpy as np
from solution.evidence import labels_from_batch,score_sequence,error_attribution,visibility_strata
from solution.calibration import probability_metrics

def evaluate(batch,p_alive,p_home,prediction,*,crosschecked=False):
    truth=labels_from_batch(batch)
    complete=bool((truth[batch['period']>0]!='X').all())
    score=score_sequence(truth,prediction,batch['period'],float(batch['fps']),semantics='independent_complete'if complete and crosschecked else'DFL_unverified_or_partial')
    score['full_spell_scope']='DFL BallStatus/BallPossession source definition; not competition target truth'
    score['attribution']=error_attribution(truth,p_alive,p_home,prediction)
    score['probabilities']={}
    for name,y,p in [('alive',batch['y_alive'],p_alive),('home',batch['y_home'],p_home)]:
        m=(y>=0)&(batch['period']>0);score['probabilities'][name]=probability_metrics(y[m],p[m])
    score['strata']={}
    for name,m in visibility_strata(batch).items():
        m=m&(truth!='X')&(batch['period']>0)
        score['strata'][name]=dict(n=int(m.sum()),label_errors=int(np.sum(prediction[m]!=truth[m])),false_dead=int(np.sum((prediction[m]=='D')&np.isin(truth[m],['H','A']))))
    return score
