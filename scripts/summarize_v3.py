"""Aggregate frozen external diagnostics; never rank by partial spell F1."""
import hashlib,json
from pathlib import Path
import numpy as np

PATH=Path('local_artifacts/v3/research_results.json')

def missing(d):
    metrics=[d['strata'][k]['home']for k in ['missing_short','missing_medium','missing_long']]
    n=sum(m['n']for m in metrics)
    return None if not n else sum(m['n']*m.get('brier',0)for m in metrics)/n

def main():
    result=json.loads(PATH.read_text());rows=result['rows']
    ids=['1874553','1886347','1899585','1927964']
    def get(match,domain,variant):
        return next(r['diagnostics']for r in rows if r['heldout']==match and r['domain']==domain and r['variant']==variant)
    comparisons=[]
    for domain in ['natural','dropout_stress','metrica']:
        for variant in ['baseline','E1_prior','E1_platt','E2_missing_expert','E3_time_normalized']:
            ds=[get(m,domain,variant)for m in ids]
            comparisons.append(dict(domain=domain,variant=variant,partial_macro_f1=[d['primary']['partial_macro_f1']for d in ds],missing_home_brier=[missing(d)for d in ds],alive_proxy_brier=[d['alive_proxy']['brier']for d in ds],alive_proxy_log_loss=[d['alive_proxy']['log_loss']for d in ds],alive_proxy_ece=[d['alive_proxy']['ece']for d in ds],home_brier=[d['home']['brier']for d in ds]))
    delta=np.array([missing(get(m,'natural','E2_missing_expert'))-missing(get(m,'natural','baseline'))for m in ids])
    rng=np.random.default_rng(2027);ci=np.quantile(rng.choice(delta,size=(2000,4),replace=True).mean(axis=1),[.025,.975])
    macro_delta=np.array([get(m,'natural','E2_missing_expert')['primary']['partial_macro_f1']-get(m,'natural','baseline')['primary']['partial_macro_f1']for m in ids])
    visible_equal=True
    for m in ids:
        with np.load(f'local_artifacts/v2/{m}-observed.npz')as b, np.load(f'local_artifacts/v3/{m}/natural-probabilities.npz')as p:
            visible=b['X'][:,b['feature_names'].tolist().index('ball_visible')]==1
            visible_equal &= np.array_equal(p['home'][visible],p['home_expert'][visible])
    retained=bool((delta<0).sum()>=3 and ci[1]<0 and visible_equal and (macro_delta>=-.01).all())
    summary=dict(protocol=result['protocol'],source_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),heldout_order=ids,comparisons=comparisons,missing_expert=dict(delta_per_match=delta.tolist(),mean_delta=float(delta.mean()),paired_match_bootstrap_95_interval=ci.tolist(),improved_matches=int((delta<0).sum()),visible_home_exactly_unchanged=bool(visible_equal),primary_macro_delta=macro_delta.tolist(),experimental_retention=retained,promoted=False),limitations=['four previously used development matches','team identities overlap across matches','Metrica is one previously viewed match evaluated by four models, not four independent matches','natural missing labels are sparse and selected','complete independent HAD truth unavailable'])
    Path('docs/external_validation_v3.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    Path('docs/external_summary_v3.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['missing_expert'],indent=2))
if __name__=='__main__':main()
