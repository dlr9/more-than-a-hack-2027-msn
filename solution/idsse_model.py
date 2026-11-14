"""Research-only schema3 candidate with explicit provider/observation compatibility."""
import json
from .candidate import Candidate
from .provider_neutral import FeatureContract,assert_compatible

class IDSSECandidate(Candidate):
    def probabilities(self,batch):
        actual=FeatureContract(**json.loads(str(batch['feature_contract_json'])))
        sources=[FeatureContract(**item)for item in self.metadata['training_contracts']]
        assert_compatible(sources,actual,purpose='evaluate',allow_cross_pitch=True)
        return super().probabilities(batch)


class MissingStableCandidate(IDSSECandidate):
    def predict(self,batch):
        import numpy as np
        from .decoder import emission_probabilities,viterbi_decode
        from .quality import period_blocks
        a,h=self.probabilities(batch)
        visible=batch['X'][:,batch['feature_names'].tolist().index('ball_visible')]==1
        scale=np.where(visible & np.r_[True,visible[:-1]],1.,4.)
        fps_scale=float(batch['fps'])/10
        return viterbi_decode(emission_probabilities(a,h),period_blocks(batch['period']),
                              3*fps_scale,1*fps_scale,dead_scale=scale)
