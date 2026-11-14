"""Inspect tracked PATHS only. Never open a suspected private artifact."""
import subprocess
from pathlib import PurePosixPath

FORBIDDEN_PARTS = {'local_artifacts', 'pie_private', 'models', 'tactical', 'broadcast'}
FORBIDDEN_NAMES = {'hack2027_solution_technical_report.html', 'model_training_validation.md'}

def violations(paths):
    bad=[]
    for path in paths:
        p=PurePosixPath(path)
        if set(p.parts)&FORBIDDEN_PARTS or p.name in FORBIDDEN_NAMES or p.suffix in {'.npz','.joblib','.mp4','.jsonl','.pl3container','.gzip'}:
            bad.append(path)
        elif p.parts[:2] == ('evaluation','results') and p.suffix=='.csv' and not p.name.startswith('baseline_'):
            bad.append(path)
        elif p.parts and p.parts[0]=='data' and path!='data/README.md':
            bad.append(path)
    return bad

def main():
    paths=subprocess.check_output(['git','ls-files','-z'],text=True).split('\0')
    bad=violations([p for p in paths if p])
    if bad:
        print('STOP: human isolation/review required for tracked paths (contents not read):')
        print('\n'.join(bad));raise SystemExit(1)
    print('Public path guard passed; this is not a content or history audit.')

if __name__=='__main__':main()
