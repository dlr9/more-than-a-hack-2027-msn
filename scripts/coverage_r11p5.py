"""Label-free whole-match ten-second PathCRF coverage, no torch."""
import json,time
from pathlib import Path
import numpy as np
from solution.idsse import digest
from solution.hierarchical_decoder import continuous_blocks
from scripts.research_r7 import IDS,write_new

def main():
 rows=[]
 for m in IDS:
  path=Path(f'local_artifacts/r6/prepared/{m}/raw.npz')
  with np.load(path,allow_pickle=False) as z:f,p,xy,team,ids=[z[k] for k in ('frame','period','players_xy','player_team','player_track_id')]
  n=valid=0
  for a,b in continuous_blocks(f,p):
   for start in range(a,b-249,250):
    n+=1;v=np.isfinite(xy[start:start+250]).all(axis=2);active=v.all(axis=0)
    valid+=int(all(np.sum(active&(team==s))==11 for s in (0,1)) and not np.any(v[:,~active]) and len(set(ids[active]))==22)
  rows.append(dict(match=m,source_sha=digest(path),windows=n,complete_stable_identity_windows=valid,fraction=valid/n,verified_keeper_windows=None,strict_usable_fraction=None,reason='GK metadata not verified; complete identity coverage is upper bound only, source optical tracking not automatic target tracking'))
 path=Path('local_artifacts/external/skillcorner/1874553/1874553_tracking_extrapolated.jsonl')
 meta=json.loads(path.with_name('1874553_match.json').read_text());teams={str(meta['home_team']['id']):0,str(meta['away_team']['id']):1};players={str(p['id']):teams[str(p['team_id'])] for p in meta['players']}
 n=valid=0;window=[];previous=None
 with path.open() as f:
  for line in f:
   r=json.loads(line);frame=r['frame'];period=r['period']
   if period not in (1,2):window=[];previous=None;continue
   if previous is not None and (frame!=previous[0]+1 or period!=previous[1]):window=[]
   observed=[]
   for player in r['player_data']:
    if player.get('is_detected') is True:
     key=str(player.get('player_id'));team=players.get(key)
     if team is not None and all(np.isfinite(player.get(k,np.nan)) for k in ('x','y')):observed.append((key,team))
   window.append(set(observed));previous=(frame,period)
   if len(window)==100:
    n+=1;s=window[0];valid+=int(len(s)==22 and sum(t==0 for _,t in s)==11 and all(v==s for v in window));window=[]
 rows.append(dict(match='SC1874553',source_sha=digest(path),windows=n,complete_stable_identity_windows=valid,fraction=valid/n,verified_keeper_windows=None,strict_usable_fraction=None,reason='Detected players only, no extrapolation; metadata GK role unverified; strict coverage remains unresolved'))
 write_new('docs/R11P5_PATHCRF_COVERAGE.json',dict(rows=rows,decision='NO_INSTALL; unknown GK and detected coverage gap; no partial-input forward claim',labels_read=False,torch_installed=False))
 print(json.dumps(rows))
if __name__=='__main__':main()
