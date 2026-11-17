"""Read-only verification of public-external R14 artifacts; never opens target data."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from solution.idsse import digest
from scripts.research_r7 import load
from solution.evidence import labels_from_batch
from solution.hierarchical_decoder import continuous_blocks
from r11.spell_roi import score
from scripts.evaluate_r11_roi import check


def verify(out):
    if Path(out).exists():
        raise FileExistsError('Use a new verification receipt; preserve the old one')
    oracle=json.loads(Path('docs/R14_ORACLE_BOTTLENECK.json').read_text())
    dynamic=json.loads(Path('docs/R14_FEATURE_INCREMENTAL_VALUE.json').read_text())
    verified=set()
    def protocol(path, sha, commit):
        assert digest(path)==sha, path
        data=json.loads(Path(path).read_text())
        for name,h in data['hashes'].items():
            key=(name,h,commit if not name.startswith('local_artifacts/') else '')
            if key in verified:continue
            if name.startswith('local_artifacts/'):
                assert digest(name)==h,name
            else:
                blob=subprocess.check_output(['git','show',f'{commit}:{name}'])
                assert hashlib.sha256(blob).hexdigest()==h,(commit,name)
            verified.add(key)
    n=0
    for r in oracle['rows']:
        m,d=r['match'],r['domain']
        protocol('docs/R14_ORACLE_PROTOCOL_V2.json',r['protocol_sha256'],r['git_sha'])
        path=f'local_artifacts/r14/oracle-v1/{m}-{d}.npz'
        assert digest(path)==r['prediction_sha256'],path
        b=load(m,d);truth=labels_from_batch(b);blocks=list(continuous_blocks(b['frame'],b['period']))
        with np.load(path,allow_pickle=False) as z:
            for name,result in r['results'].items():
                saved=dict(result['metrics']['spell']);saved['per_iou']={float(k):v for k,v in saved['per_iou'].items()}
                check(score(truth.tolist(),z[name].tolist(),blocks=blocks),saved);n+=1
    for r in dynamic['rows']:
        paths=['docs/R14_DYNAMIC_PROTOCOL.json','docs/R14_DYNAMIC_PROTOCOL_V2.json']
        p=next(p for p in paths if digest(p)==r['protocol_sha256'])
        protocol(p,r['protocol_sha256'],r['git_sha'])
        assert digest(f"local_artifacts/r14/dynamic-v1/{r['match']}-timing.npz")==r['proposal_sha256']
    result=dict(oracle_saved_score_rechecks=n,artifact_and_protocol_hashes=True,historical_source_versions_verified=True,unique_hash_checks=len(verified),timing_gate=dynamic['timing_gate'],decision='NO_EVIDENCE_TO_PROMOTE',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    with Path(out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);verify(p.parse_args().out)
