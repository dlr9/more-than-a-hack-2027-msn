"""Verify hashes and exactly replay saved R7B predictions; no fit or reselection."""
import json,sys
from pathlib import Path
import numpy as np,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_r7b import ROOT,prepared,baseline,decode
from scripts.research_r7 import IDS,load
from scripts.calibrate_r7b import mixture
from solution.quality_experts import fuse
from solution.idsse import digest

def check(path,a,h,b):
    with np.load(path)as saved:
        assert np.array_equal(a,saved['p_alive']),str(path)+' alive'
        assert np.array_equal(h,saved['p_home']),str(path)+' home'
        assert np.array_equal(decode(a,h,b),saved['decoded']),str(path)+' decoded'
def main():
    registry=json.loads(Path('docs/R7B_MODEL_REGISTRY.json').read_text())
    for r in registry['models']:
        path=Path(r['path']);assert digest(path)==r['sha256']and path.stat().st_size==r['bytes']
    for m in IDS:
        players=prepared(m);p1=joblib.load(ROOT/'p1-v1'/f'{m}.joblib');pa=p1['alive'].predict_proba(players['X'])[:,1]
        cal=joblib.load(ROOT/'calibration-v1'/f'{m}.joblib');assert not set(cal['fit'])&set(cal['calibration']);assert m not in cal['fit']+cal['calibration']
        cp=cal['player'].predict_proba(players['X'])[:,1];cf=joblib.load(ROOT/'counterfactual-v1'/f'{m}.joblib')
        for domain in ['natural','dropout']:
            b=load(m,domain);old=baseline(m,domain);h=old['p_home'];g,_=fuse(b,players,old['p_alive'],pa)
            check(ROOT/'p1-eval-v1'/f'{m}-P1-{domain}.npz',pa,h,b)
            check(ROOT/'p1-eval-v1'/f'{m}-G1-{domain}.npz',g,h,b)
            check(ROOT/'counterfactual-v1'/f'{m}-CF1-{domain}.npz',cf.predict_proba(b['X'])[:,1],h,b)
            ba=cal['ball'].predict_proba(b['X'])[:,1]
            for variant,a in [('G2_uncal',mixture(b,players,ba,cp)),('G2_cal',mixture(b,players,cal['ball_calibration'].predict(ba),cal['player_calibration'].predict(cp)))]:
                check(ROOT/'calibration-v1'/f'{m}-{variant}-{domain}.npz',a,h,b)
        print('Exact saved replay',m,'P1/G1/CF1/G2uncal/G2cal natural+dropout',flush=True)
    print('Verified9newbundles and30complete prediction replays; no training or new blind evaluation')
if __name__=='__main__':main()
