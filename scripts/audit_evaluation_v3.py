"""E0: no fitting; expose partial-label censoring and time-scale differences."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solution.modeling import load_prepared
from solution.evidence import labels_from_batch,period_reference,score_sequence
from solution.heuristic import majority_smooth
from evaluation import possession_metrics as pm

IDS=['1874553','1886347','1899585','1927964']
def main():
    rows=[];identities={}
    for match in IDS+['metrica']:
        path=Path('local_artifacts/v2')/('metrica.npz' if match=='metrica' else f'{match}-observed.npz')
        batch=load_prepared(path)
        for policy in (['strict'] if match=='metrica' else ['strict','phase_proxy']):
            truth=labels_from_batch(batch,policy)
            for mode in ['historical_50_global','seconds_2_period']:
                pred=majority_smooth(batch['nearest_codes'],50) if mode=='historical_50_global' else period_reference(batch)
                score=score_sequence(truth,pred,batch['period'],float(batch['fps']),semantics='metrica_event_proxy' if match=='metrica' else 'skillcorner_'+policy)
                # Exact historical duration convention, explicitly kept separate.
                legacy=pm.evaluate(pm.label_codes(truth),pm.label_codes(pred),bootstrap_n=0)
                rows.append(dict(match=match,policy=policy,baseline=mode,fps=float(batch['fps']),score=score,legacy_reported_mean_spell_s=legacy['mean_spell_duration_s'],legacy_censored_spell_f1=legacy['spell_f1_mean']))
        if match!='metrica':
            meta=json.loads((Path('local_artifacts/external/skillcorner')/match/f'{match}_match.json').read_text())
            identities[match]=[meta['home_team']['id'],meta['away_team']['id']]
    result=dict(protocol_sha256=hashlib.sha256(Path('docs/evaluation_protocol_v3.json').read_bytes()).hexdigest(),fit_performed=False,rows=rows,team_ids=identities,team_overlap_pairs=[dict(matches=[a,b],teams=sorted(set(identities[a])&set(identities[b])))for i,a in enumerate(IDS)for b in IDS[i+1:]if set(identities[a])&set(identities[b])])
    Path('docs/evaluation_audit_v3.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(dict(rows=len(rows),team_overlap=result['team_overlap_pairs']),indent=2))
if __name__=='__main__':main()
