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
