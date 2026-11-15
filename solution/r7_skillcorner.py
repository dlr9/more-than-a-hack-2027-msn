"""Observed-player raw coordinates for explicit cross-provider research only."""
import json
from pathlib import Path
import numpy as np
from .external import _jsonl_objects

def read_positions(folder):
    folder=Path(folder);m=folder.name
    meta=json.loads((folder/f'{m}_match.json').read_text())
    home,away=meta['home_team']['id'],meta['away_team']['id']
    players=[p for p in meta['players']if p['team_id']in{home,away}]
    slots={};teams=[];identifiers=[]
    for i,p in enumerate(players):
        teams.append(int(p['team_id']==away));identifiers.append(str(p['id']))
        for key in ['id','trackable_object']:
            if p.get(key)is not None:slots[str(p[key])]=i
    periods=meta['match_periods'];n=max(int(x['end_frame'])for x in periods)+1
    raw=dict(frame=np.arange(n),period=np.zeros(n,dtype=np.int16),ball_xy=np.full((n,2),np.nan),
        players_xy=np.full((n,len(players),2),np.nan),player_team=np.asarray(teams),player_track_id=np.asarray(identifiers))
    for p in periods:raw['period'][int(p['start_frame']):int(p['end_frame'])+1]=int(p['period'])
    count=0
    for record in _jsonl_objects(folder/f'{m}_tracking_extrapolated.jsonl'):
        frame=int(record['frame'])
        if frame!=count or frame>=n:raise ValueError('unaligned SC raw timeline')
        count+=1;ball=record.get('ball_data')or{}
        if ball.get('x')is not None and ball.get('y')is not None:raw['ball_xy'][frame]=[ball['x'],ball['y']]
        seen=set()
        for p in record.get('player_data')or[]:
            if p.get('is_detected')is not True:continue
            key=str(p.get('player_id',p.get('trackable_object')))
            if key not in slots:raise ValueError('unknown detected player')
            slot=slots[key]
            if slot in seen:raise ValueError('duplicate detected player')
            seen.add(slot)
            if p.get('x')is not None and p.get('y')is not None:raw['players_xy'][frame,slot]=[p['x'],p['y']]
    if count!=n:raise ValueError('incomplete SC raw timeline')
    return raw,(float(meta['pitch_length']),float(meta['pitch_width']))
