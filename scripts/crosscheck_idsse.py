"""Optional independent floodlight parser check on public development XML only."""
import argparse,hashlib,importlib.metadata,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.idsse import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--match',choices=['J03WPY','J03WOH','J03WQQ'],required=True);a=p.parse_args()
    from floodlight.io.dfl import read_position_data_xml
    import inspect
    root=Path('local_artifacts/r6/idsse-hf');m=a.match
    pos=root/f'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-{m}.xml';meta=root/f'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-{m}.xml';rawpath=Path(f'local_artifacts/r6/prepared/{m}/raw.npz')
    out=Path(f'local_artifacts/r6/floodlight_crosscheck_{m}.json')
    if out.exists():raise FileExistsError('preserve existing independent check')
    xy,poss,status,_,_=read_position_data_xml(pos,meta);halves=[]
    with np.load(rawpath)as raw:
        for i,half in enumerate(['firstHalf','secondHalf'],1):
            mask=raw['period']==i;s=raw['ball_status'][mask];q=raw['ball_possession'][mask]
            labels_equal=bool(np.array_equal(s,status[half].code)and np.array_equal(q,poss[half].code))
            ball_equal=bool(np.allclose(raw['ball_xy'][mask],xy[half]['Ball'].xy,equal_nan=True))
            if not labels_equal or not ball_equal:raise ValueError('independent parser mismatch')
            halves.append(dict(period=half,frames=int(mask.sum()),labels_equal=labels_equal,ball_xy_equal=ball_equal,transition_positions_equal=labels_equal,status_sha256=hashlib.sha256(s.tobytes()).hexdigest(),possession_sha256=hashlib.sha256(q.tobytes()).hexdigest()))
    result=dict(match=m,dependencies={k:importlib.metadata.version(k)for k in ['floodlight','lxml','numpy','iso8601']},files={str(p):digest(p)for p in [pos,meta,rawpath]},floodlight_parser_sha256=digest(inspect.getfile(read_position_data_xml)),halves=halves)
    out.write_text(json.dumps(result,indent=2)+'\n');print(m,'independent full-period check passed')
if __name__=='__main__':main()
