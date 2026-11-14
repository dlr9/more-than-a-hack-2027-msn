"""Run a trusted local candidate on an explicitly prepared external match (not target data)."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if Path(args.out).exists(): raise FileExistsError('preserve existing prediction artifacts')
    batch=load_prepared(args.prepared)
    model=joblib.load(args.bundle)  # Only locally produced, trusted artifacts; never arbitrary downloads.
    a,h=model.probabilities(batch)
    with open(args.out,'xb') as stream:
        np.savez_compressed(stream,frame=batch['frame'],labels=model.predict(batch),p_alive=a,p_home=h)
    print('external development predictions saved; not a competition submission')

if __name__=='__main__': main()
