"""Human event supervision coverage, NOT inference or model selection.

Install duckdb==1.4.1 in an isolated target; set PYTHONPATH accordingly.
Only public J03WPY benchmark labels and public DFL source-frame IDs are read.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.research_r7 import load,write_new
from solution.evidence import labels_from_batch

def run():
 import duckdb
 path=Path('local_artifacts/r12/source-audit/J03WPY.parquet');blob=path.read_bytes()
 assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()=='1125bb08c765f0570814adb07fdf01b4622c2b0a'
 con=duckdb.connect();data=con.execute('SELECT period_id, player_id, spadl_type, frame_id, receiver_id, receive_frame_id FROM read_parquet(?)',[str(path)]).fetchall()
 b=load('J03WPY','natural');g=labels_from_batch(b)
 with np.load('local_artifacts/r6/prepared/J03WPY/raw.npz',allow_pickle=False) as z:source=z['source_frame']
 index={(int(p),int(f)):i for i,(p,f) in enumerate(zip(b['period'],source))}
 ev=json.loads(Path('local_artifacts/r12/global-v1/J03WPY-natural-anchors.json').read_text());signal=np.full(len(g),-1,np.int8)
 for e in ev:signal[e['start']:e['end']]=int(e['team']=='A')
 counts={};groups={k:dict(n=0,aligned=0,correct_anchor_at_time=0,correct_anchor_within1s=0,source_HAD_agrees_at_time=0) for k in ['all_receipts','cross_team_receipts','ball_recovery','interception','keeper_pick_up','keeper_claim']}
 for period,player,kind,frame,receiver,receive in data:
  counts[kind]=counts.get(kind,0)+1;items=[]
  if receiver and receive is not None and np.isfinite(receive):
   items.append(('all_receipts',receive,receiver))
   if player.split('_')[0]!=receiver.split('_')[0]:items.append(('cross_team_receipts',receive,receiver))
  if kind in groups:items.append((kind,frame,player))
  for name,f,who in items:
   q=groups[name];q['n']+=1;i=index.get((int(period),int(f)))
   if i is None:continue
   q['aligned']+=1;team=0 if who.startswith('home_') else 1
   q['correct_anchor_at_time']+=int(signal[i]==team)
   lo=max(0,i-25);hi=min(len(g),i+26);same=b['period'][lo:hi]==period
   q['correct_anchor_within1s']+=int(np.any((signal[lo:hi]==team)&same))
   q['source_HAD_agrees_at_time']+=int(g[i]==('H' if team==0 else 'A'))
 result=dict(match='J03WPY',source='https://github.com/hyunsungkim-ds/elastic/tree/cikm2026/benchmark',license='CC BY 4.0 benchmark; MPL2 source not imported',git_blob_sha='1125bb08c765f0570814adb07fdf01b4622c2b0a',sha256=hashlib.sha256(blob).hexdigest(),event_count=len(data),types=counts,groups=groups,notes=['Human revised event times are new supervision, not an autonomous inference input.','Same consumed DFL match, not new independent test match.','Receipt is not necessarily durable controlled acquisition; at-time source-HAD disagreement does not establish either annotation wrong.','Unknown/uncovered event times are retained in denominators. No threshold or model changed.'],download_command='Download the linked J03WPY.parquet only; validate git blob SHA before reading.')
 write_new('docs/R12_EXTERNAL_EVENT_AUDIT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
