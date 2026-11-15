"""Explicit weak-label research on pinned PUBLIC Sample Game 3, never target ATD."""
import json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,joblib,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.atd import _triple
from solution.provider_neutral import FeatureContract,provider_neutral_features_v3
from solution.player_only_features import player_inputs,build_player_features
from solution.quality_experts import fuse
from solution.evidence import score_sequence,labels_from_batch,visibility_strata
from solution.calibration import probability_metrics
from solution.idsse import digest
from scripts.research_r7 import model_path,write_new
from scripts.research_r7b import ROOT,identity,model,decode

def read_public_metrica():
    folder=Path('local_artifacts/external/metrica');manifest=json.loads(Path('local_artifacts/external/manifest.json').read_text())
    files=[folder/f'Sample_Game_3_{s}'for s in ['metadata.xml','tracking.txt','events.json']]
    for path in files:
        entry=next(r for r in manifest['files']if r['path']==str(path))
        if digest(path)!=entry['sha256']:raise ValueError('pinned public sample drift')
        if '/metrica-sports/sample-data/'not in entry['url']or '/Sample_Game_3/'not in entry['url']:raise ValueError('unapproved source')
    root=ET.parse(files[0]).getroot();old=dict(np.load('local_artifacts/v2/metrica.npz',allow_pickle=False));n=len(old['frame'])
    teams=[node.get('id')for node in root.iter('Team')];pt={node.get('id'):node.get('teamId')for node in root.iter('Player')}
    channels={node.get('id'):node.get('playerId')for node in root.iter('PlayerChannel')};layouts=[]
    for spec in root.iter('DataFormatSpecification'):
        registers=[r for r in spec.findall('SplitRegister')if r.get('separator')==';']
        if len(registers)!=1:raise ValueError('ambiguous public slot register')
        ids=[];mapping=[]
        for slot in registers[0].findall('SplitRegister'):
            found={channels.get(ref.get('playerChannelId'))for ref in slot.iter('PlayerChannelRef')}
            if len(found)!=1 or None in found:raise ValueError('ambiguous public player identity')
            player=next(iter(found));ids.append(player)
            if pt[player]not in teams:raise ValueError('non-team player must be excluded explicitly')
            mapping.append(teams.index(pt[player]))
        layouts.append((int(spec.get('startFrame'))-1,int(spec.get('endFrame')),ids,mapping))
    slots=max(len(x[2])for x in layouts)
    raw=dict(frame=old['frame'].copy(),period=old['period'].copy(),ball_xy=np.full((n,2),np.nan),players_xy=np.full((n,slots,2),np.nan),
             player_team=np.full((n,slots),-1,dtype=np.int8),player_track_id=np.full((n,slots),'',dtype='<U64'))
    for a,b,ids,mapping in layouts:raw['player_team'][a:b,:len(ids)]=mapping;raw['player_track_id'][a:b,:len(ids)]=ids
    with files[1].open()as f:
        for count,line in enumerate(f):
            parts=line.rstrip('\r\n').split(':')
            if len(parts)!=3 or int(parts[0])-1!=count or count>=n:raise ValueError('public sample timeline mismatch')
            layout=[x for x in layouts if x[0]<=count<x[1]]
            if len(layout)!=1:raise ValueError('ambiguous layout at frame')
            groups=parts[1].split(';')
            if len(groups)!=len(layout[0][2]):raise ValueError('slot mismatch')
            for slot,item in enumerate(groups):
                parsed=_triple(item,True)
                if parsed is not None:raw['players_xy'][count,slot]=(np.array(parsed[:2])-.5)*[105,68]
            parsed=_triple(parts[2],True)
            if parsed is not None:raw['ball_xy'][count]=(np.array(parsed[:2])-.5)*[105,68]
    if count+1!=n:raise ValueError('incomplete sample')
    return raw,old,{str(p):digest(p)for p in files}

def main():
    out=ROOT/'metrica-v1';out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if (out/'prepared.json').exists():
        meta=json.loads((out/'prepared.json').read_text())
        for filename,sha in meta['prepared_hashes'].items():assert digest(out/filename)==sha
        b=dict(np.load(out/'base.npz'));players=dict(np.load(out/'players.npz'))
    else:
        raw,old,inputs=read_public_metrica();contract=FeatureContract('Metrica_public_sample','available_unknown',25.)
        b=provider_neutral_features_v3(raw,contract);players=build_player_features(player_inputs(raw),fps=25.)
        b.update(y_alive=old['y_alive'],y_home=old['y_home'],match_id=np.asarray('metrica-game-3'))
        np.savez_compressed(out/'base.npz',**b);np.savez_compressed(out/'players.npz',**players)
        write_new(out/'prepared.json',dict(inputs=inputs,prepared_hashes={p:digest(out/p)for p in ['base.npz','players.npz']},wall_s=time.monotonic()-started,**identity()))
    ball=joblib.load(model_path('J03WPY','M2'));x=pd.DataFrame(b['X'],columns=b['feature_names'].tolist())
    ba=ball.alive.predict_proba(x)[:,1];h=ball.home.predict_proba(x)[:,1];pa=model('J03WPY')['alive'].predict_proba(players['X'])[:,1];mixed,use=fuse(b,players,ba,pa);rows=[]
    for v,a in [('M2',ba),('P1',pa),('G1',mixed)]:
        path=out/f'{v}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        pred=decode(a,h,b);truth=labels_from_batch(b);score=score_sequence(truth,pred,b['period'],25.,semantics='partial_event_proxy')
        k=(b['y_alive']>=0)&(b['period']>0);score['alive']=probability_metrics(b['y_alive'][k],a[k])
        row=dict(variant=v,metrics=score,full_spell_f1=None,waiver='Explicit research-only provider transfer; weak event interiors, no complete state truth')
        np.savez_compressed(path.with_suffix('.npz'),p_alive=a,p_home=h,decoded=pred);write_new(path,row);rows.append(row)
        print('METRICA WEAK',v,score['partial_macro_f1'],score['live_to_dead_proxy_errors'],score['dead_to_live_proxy_errors'],flush=True)
    path=Path('docs/R7B_METRICA_RESULTS.json')
    if not path.exists():write_new(path,dict(rows=rows,**identity(),adapter_sha256=digest(__file__),selection='diagnostic only; fixed first DFL fold'))
if __name__=='__main__':main()
