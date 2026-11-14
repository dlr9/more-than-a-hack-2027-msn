"""Stream public IDSSE files to frozen natural/dropout feature caches."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions',required=True);p.add_argument('--matchinfo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.positions,a.matchinfo,a.out),indent=2))
