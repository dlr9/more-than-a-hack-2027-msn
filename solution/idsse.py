"""IDSSE public DFL XML parser. Labels are isolated from coordinate-only inputs."""
from __future__ import annotations
import datetime,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
from .provider_neutral import FeatureContract,provider_neutral_features_v3
from .quality import dropout_windows

SECTIONS={'firstHalf':1,'secondHalf':2}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def metadata(path):
    root=ET.parse(path).getroot();general=root.find('.//General');env=root.find('.//Environment')
    teams={};players={}
    for t in root.findall('.//Teams/Team'):
        role=t.get('Role')
        if role not in {'home','guest'}:raise ValueError('unrecognized team role')
        team=int(role=='guest');teams[t.get('TeamId')]=team
        for p in t.findall('./Players/Player'):
            key=p.get('PersonId')
            if key in players:raise ValueError('duplicate metadata player')
            players[key]=team
    if len(teams)!=2 or teams.get(general.get('HomeTeamId'))!=0 or teams.get(general.get('GuestTeamId'))!=1:raise ValueError('metadata teams disagree')
    return dict(match_id=general.get('MatchId').removeprefix('DFL-MAT-'),pitch=(float(env.get('PitchX')),float(env.get('PitchY'))),teams=teams,players=players)

def iter_frames(path):
    context={}
    for event,element in ET.iterparse(path,events=('start','end')):
        if event=='start'and element.tag=='FrameSet':context=dict(element.attrib)
        elif event=='end'and element.tag=='Frame':
            yield context,element.attrib.copy()
            element.clear()
        elif event=='end'and element.tag=='FrameSet':element.clear()

def label(status,possession):
    if status=='0':return 'D'
    if status=='1'and possession in {'1','2'}:return 'H'if possession=='1'else'A'
    return 'X'

def number(frame,name):
    try:v=float(frame.get(name,'nan'))
    except (TypeError,ValueError):return np.nan
    return v if np.isfinite(v)else np.nan

def read_raw(positions,matchinfo):
    meta=metadata(matchinfo);balls={k:[]for k in SECTIONS};inventory=set()
    for context,frame in iter_frames(positions):
        section=context.get('GameSection')
        if section not in SECTIONS:raise ValueError('unsupported period')
        if context.get('MatchId')!='DFL-MAT-'+meta['match_id']:raise ValueError('match ID conflict')
        team=context.get('TeamId')
        if team.lower()=='ball':balls[section].append(frame)
        else:
            person=context.get('PersonId')
            if team not in meta['teams']or meta['players'].get(person)!=meta['teams'][team]:raise ValueError('unknown or conflicting player team')
            inventory.add(person)
    if any(not v for v in balls.values()):raise ValueError('both complete regulation periods required')
    ids=sorted(inventory);slots={p:i for i,p in enumerate(ids)}
    bounds={k:(min(int(f['N'])for f in fs),max(int(f['N'])for f in fs))for k,fs in balls.items()}
    total=sum(b-a+1 for a,b in bounds.values());offset={};cursor=0
    for k,(a,b)in bounds.items():offset[k]=cursor;cursor+=b-a+1
    raw=dict(frame=np.arange(total),period=np.empty(total,dtype=np.int16),ball_xy=np.full((total,2),np.nan),players_xy=np.full((total,len(ids),2),np.nan),player_team=np.array([meta['players'][p]for p in ids]),player_track_id=np.array(ids))
    reported_speed=np.full((total,len(ids)),np.nan,dtype=np.float32)
    truth=np.full(total,'X');source_frame=np.zeros(total,dtype=np.int64);time=np.full(total,np.nan);ball_seen=np.zeros(total,dtype=np.int8)
    status=np.full(total,-1,dtype=np.int8);possession=np.full(total,-1,dtype=np.int8)
    duplicates=0;ball_examples=[]
    for section,fs in balls.items():
        lo,hi=bounds[section];a=offset[section];b=a+hi-lo+1
        raw['period'][a:b]=SECTIONS[section];source_frame[a:b]=np.arange(lo,hi+1)
        for f in fs:
            j=a+int(f['N'])-lo
            if ball_seen[j]:
                duplicates+=1;truth[j]='X';raw['ball_xy'][j]=np.nan;status[j]=-1;possession[j]=-1;continue
            ball_seen[j]=1;truth[j]=label(f.get('BallStatus'),f.get('BallPossession'))
            status[j]=int(f['BallStatus'])if f.get('BallStatus')in {'0','1'}else -1
            possession[j]=int(f['BallPossession'])if f.get('BallPossession')in {'0','1','2'}else -1
            raw['ball_xy'][j]=[number(f,'X'),number(f,'Y')]
            time[j]=datetime.datetime.fromisoformat(f['T']).timestamp()
            if len(ball_examples)<3:ball_examples.append({key:f.get(key)for key in ['N','T','X','Y','BallStatus','BallPossession']})
    del balls
    seen=np.zeros((total,len(ids)),dtype=bool);outside=0
    for context,f in iter_frames(positions):
        if context['TeamId'].lower()=='ball':continue
        section=context['GameSection'];lo,hi=bounds[section];n=int(f['N'])
        if not lo<=n<=hi:outside+=1;continue
        j=offset[section]+n-lo;s=slots[context['PersonId']]
        if seen[j,s]:raise ValueError('duplicate player frame')
        seen[j,s]=True;raw['players_xy'][j,s]=[number(f,'X'),number(f,'Y')]
        reported_speed[j,s]=number(f,'S')/3.6 # Source km/h, never fed to neutral features.
    same=raw['period'][1:]==raw['period'][:-1];valid=same&np.isfinite(time[1:])&np.isfinite(time[:-1])
    delta=np.diff(time)[valid];bad=int(np.sum(np.abs(delta-.04)>1e-5))
    if bad:raise ValueError(f'{bad} timestamp/frame intervals disagree with 25 FPS')
    audit=dict(match_id=meta['match_id'],n_frames=total,bounds=bounds,period_lengths={k:b-a+1 for k,(a,b)in bounds.items()},label_counts=dict(Counter(truth.tolist())),known_fraction=float(np.mean(truth!='X')),missing_ball_records=int(np.sum(ball_seen==0)),duplicate_ball_records=duplicates,player_records_outside_ball_period=outside,timestamp_bad_intervals=bad,pitch=meta['pitch'],position_sha256=digest(positions),metadata_sha256=digest(matchinfo),source='DFL IDSSE CC BY 4.0; ballstatus/possession source semantics, not official competition truth',ball_samples=ball_examples)
    for section,p in SECTIONS.items():
        m=raw['period']==p;y=truth[m];edges=np.diff(np.r_[False,y=='X',False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        audit[section]=dict(counts=dict(Counter(y.tolist())),longest_unknown_s=float(max(ends-starts,default=0)/25),transitions=int(np.sum(y[1:]!=y[:-1])),dead_segments=int(np.sum(np.diff(np.r_[False,y=='D',False].astype(int))==1)))
    return raw,truth,meta,audit,dict(source_frame=source_frame,ball_status=status,ball_possession=possession,reported_speed_mps=reported_speed)

def prepare(positions,matchinfo,output):
    out=Path(output)
    if out.exists():raise FileExistsError('never overwrite prepared run')
    raw,truth,meta,audit,extra=read_raw(positions,matchinfo)
    contract=FeatureContract(provider='IDSSE_DFL',observation_view='available_unknown',fps=25.,pitch=meta['pitch'])
    out.mkdir(parents=True)
    np.savez_compressed(out/'raw.npz',**raw,truth=truth,**extra)
    for name,mask in [('natural',None),('dropout',dropout_windows(raw['period'],25.,2027))]:
        batch=provider_neutral_features_v3(raw,contract,ball_dropout_mask=mask)
        batch.update(y_alive=np.where(truth=='X',-1,(truth!='D').astype(np.int8)),y_home=np.where(np.isin(truth,['H','A']),(truth=='H').astype(np.int8),-1),match_id=np.asarray(meta['match_id']),label_semantics=np.asarray('DFL_source_complete'if(truth!='X').all()else'DFL_source_partial'))
        np.savez_compressed(out/f'{name}.npz',**batch)
        audit[name]=dict(prepared_sha256=digest(out/f'{name}.npz'),features=len(batch['feature_names']),missing_feature_fraction=float(np.mean(np.isnan(batch['X']))),ball_missing_fraction=float(np.mean(batch['X'][:,batch['feature_names'].tolist().index('ball_visible')]==0)))
        print(meta['match_id'],name,'features saved',flush=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit
