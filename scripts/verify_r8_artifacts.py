"""Read-only R8 hashes, exact candidate replay, sample budgets and legacy scores."""
import json,sys
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest
from solution.hierarchical_decoder import decode,anchored
from scripts.research_r7 import IDS,load
from scripts.evaluate_r8_decoder import original,prediction_path

def main():
    root=Path('local_artifacts/r8');count=0
    for file in ['docs/R8_PROTOCOL.json','docs/R8_E08B_PROTOCOL.json','docs/R8_E08C_PROTOCOL.json']:
        p=json.loads(Path(file).read_text())
        for path,sha in p['inputs'].items():
            assert digest(path)==sha,(file,path)
    old=json.loads((root/'e08a/results.json').read_text())
    fixed=json.loads((root/'e08a-anchor-fix/results.json').read_text())
    for row in old['rows']+fixed['rows']:
        directory='e08a-anchor-fix'if row in fixed['rows']else'e08a'
        m,v,d=row['match'],row['variant'],row['domain'];path=root/directory/f'{m}-{v}-{d}.npz'
        assert digest(path)==row['prediction_sha256'];z=dict(np.load(path));b=load(m,d)
        if v in ['M2','M3','G1']:
            expected=original(z['p_alive'],z['p_home'],b,v)
        else:
            expected=decode(z['p_alive'],z['p_home'],b['frame'],b['period'],float(b['fps']))
        np.testing.assert_array_equal(expected,z['decoded']);count+=1
        if directory=='e08a-anchor-fix':
            base=dict(np.load(prediction_path(m,'G1',d)));a,h,_=anchored(base['p_alive'],base['p_home'],b)
            np.testing.assert_array_equal(a,z['p_alive']);np.testing.assert_array_equal(h,z['p_home'])
    # Both R8 baseline scores and decodes must match the frozen historical reports.
    historical=[]
    for file in ['docs/R7_CORE_RESULTS.json','docs/R7_STRESS_RESULTS.json','docs/R7B_P1_RESULTS.json']:
        historical+=json.loads(Path(file).read_text())['rows']
    for row in old['rows']:
        if row['variant']not in ['M2','M3','G1']:continue
        ref=next(x for x in historical if all(x.get(k)==row[k]for k in ['match','variant','domain']))
        assert row['metrics']['decoded']['full_spell_f1']==ref['metrics']['decoded']['full_spell_f1']
    c=json.loads((root/'e08c/results.json').read_text());fits=0
    old_fits=json.loads(Path('docs/R6_IDSSE_RESULTS.json').read_text())['models']
    for m in IDS:
        path=root/'e08c'/f'{m}.joblib';receipt=json.loads(path.with_suffix('.json').read_text())
        assert digest(path)==receipt['model_sha256'];assert m not in receipt['train_matches'];fits+=receipt['fits']
        for head in receipt['heads']:
            assert all(s['n']==s['baseline_n']for s in head['selection'])
            original_fit=next(r for r in old_fits if r['heldout']==m and r['variant']=='M2_dropout')
            assert head['n']==original_fit[head['target'].removeprefix('y_')]['n']
        model=joblib.load(path)
        for d in ['natural','dropout','short','long']:
            row=next(r for r in c['rows']if r['match']==m and r['domain']==d)
            p=root/'e08c'/f'{m}-{d}.npz';assert digest(p)==row['prediction_sha256'];z=dict(np.load(p));b=load(m,d)
            a,h=model.probabilities(b)
            np.testing.assert_array_equal(a,z['p_alive']);np.testing.assert_array_equal(h,z['p_home'])
            np.testing.assert_array_equal(original(a,h,b,'M2'),z['decoded']);count+=1
    bresult=json.loads((root/'e08b/results.json').read_text())
    for row in bresult['rows']:
        p=root/'e08b'/f"{row['match']}-{row['domain']}-score.npz"
        assert digest(p)==row['prediction_sha256'];z=dict(np.load(p));b=load(row['match'],row['domain'])
        np.testing.assert_array_equal(original(z['p_alive'],z['p_home'],b,'M2'),z['decoded']);count+=1
    assert fits==6
    assert digest('evaluation/possession_metrics.py')=='341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d'
    print(f'R8 verified: {count} full saved decodes, 12 exact C1 head replays, 3newbundles/6fits, equal sample budgets, 36 historical baseline spell scores; scorer unchanged')

if __name__=='__main__':main()
