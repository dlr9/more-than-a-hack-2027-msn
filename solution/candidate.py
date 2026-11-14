"""Serializable development candidate. No embedded target data or automatic promotion."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .decoder import emission_probabilities, viterbi_decode
from .quality import period_blocks

@dataclass
class Candidate:
    alive: object
    home: object
    feature_names: list[str]
    metadata: dict
    missing_expert: object = None

    def probabilities(self, batch):
        if int(batch.get('schema_version',1)) != self.metadata['schema_version']:
            raise ValueError('candidate feature schema mismatch')
        if batch['feature_names'].astype(str).tolist() != self.feature_names:
            raise ValueError('candidate feature names/order mismatch')
        x=pd.DataFrame(batch['X'],columns=self.feature_names)
        alive=self.alive.predict_proba(x)[:,1]
        home=self.home.predict_proba(x)[:,1]
        if self.missing_expert is not None:
            missing=x['ball_visible'].to_numpy()!=1
            if missing.any():home[missing]=self.missing_expert.predict_proba(x.loc[missing])[:,1]
        return alive,home

    def predict(self,batch):
        a,h=self.probabilities(batch)
        scale=float(batch['fps'])/10 if self.metadata.get('decoder_time_normalized',False) else 1.
        return viterbi_decode(emission_probabilities(a,h),period_blocks(batch['period']),3.*scale,1.*scale)


def unweight_probability(probability, positive_prior):
    """Analytic inverse of balanced class reweighting; a hypothesis, not calibration proof."""
    q=np.asarray(probability,dtype=float)
    if not 0<positive_prior<1 or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError('invalid prior or probability')
    numerator=q*positive_prior
    return numerator/(numerator+(1-q)*(1-positive_prior))
