"""Verify the four saved new-task estimators without any refit."""
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.train_r11_events import ROOT, PROTOCOL, HELD, load
from r11.event_learning import temporal_inputs
from solution.idsse import digest


def replay():
    result=json.loads((ROOT/'results.json').read_text())
    protocol=json.loads(PROTOCOL.read_text())
    assert digest(PROTOCOL)==result['protocol_sha']
    for p,h in {**protocol['inputs'],**protocol['code_hashes']}.items():
        assert digest(p)==h,p
    batch,_,_=load(HELD)
    verified=[]
    for row in result['models']:
        assert digest(row['path'])==row['sha256']
        model=joblib.load(row['path']) # Locally generated, hashed estimator only.
        x=temporal_inputs(batch,protocol['training']['views'][row['view']])
        got=model.predict_proba(x)[:,1]
        with np.load(ROOT/f"{row['view']}-probabilities.npz",allow_pickle=False) as z:
            expected=z['p_acquisition' if row['target']=='acquisition' else 'p_away']
        np.testing.assert_array_equal(got,expected)
        verified.append(dict(model_sha256=row['sha256'],frames=len(got),exact=True))
    for row in result['rows']:
        assert digest(ROOT/f"{row['view']}-probabilities.npz")==row['prediction_sha']
    return dict(models=verified,new_fits=0,prediction_hashes_verified=5)

if __name__=='__main__':
    receipt=replay()
    path=ROOT/'replay.json'
    with path.open('x') as f: json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
