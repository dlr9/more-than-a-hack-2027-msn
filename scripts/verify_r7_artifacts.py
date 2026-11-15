"""Hash check all preserved models, optionally replay the nine new R7 bundles."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_r6_artifacts import verify_hashes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',action='store_true');a=p.parse_args()
    for name in ['R6','R7']:verify_hashes(json.loads(Path(f'docs/{name}_MODEL_REGISTRY.json').read_text()))
    if not a.replay:return
    import joblib,numpy as np
    from scripts.research_r7 import IDS,VARIANTS,load,model_path
    from solution.r7_features import project
    for m in IDS:
        for v in VARIANTS:
            model=joblib.load(model_path(m,v))
            for domain in ['natural','dropout']:
                b=project(load(m,domain),VARIANTS[v][0]);pa,ph=model.probabilities(b);pred=model.predict(b)
                with np.load(Path('local_artifacts/r7/core-v1')/f'{m}-{v}-{domain}.npz')as saved:
                    if not(np.array_equal(pa,saved['p_alive'])and np.array_equal(ph,saved['p_home'])and np.array_equal(pred,saved['decoded'])):raise ValueError('saved R7 replay mismatch')
            print('exact saved R7 replay',m,v,flush=True)
if __name__=='__main__':main()
