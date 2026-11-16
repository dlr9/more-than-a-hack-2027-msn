"""Research-only full timeline inference. No supplied events or source labels."""
import numpy as np
from .timeline import features,decode

def predict(batch,baseline,hazard_model,direction_model):
    hazard_x,direction_x=features(batch,True)
    hazard=hazard_model.predict_proba(hazard_x)[:,1]
    away=.5*(direction_model.predict_proba(direction_x)[:,1]+1-direction_model.predict_proba(-direction_x)[:,1])
    baseline=np.asarray(baseline)
    if baseline.shape!=(len(hazard),) or not np.isin(baseline,list('HAD')).all():
        raise ValueError('aligned complete independently computed H/A/D baseline required')
    output,accepted=decode(baseline,hazard,away,batch['frame'],batch['period'],float(batch['fps']))
    return dict(decoded=output,handover_score=hazard,p_away=away,accepted=accepted,
                status='RESEARCH_ONLY_REVERTED_FULL_SPELL_GATE_FAILED')
