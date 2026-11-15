"""One-shot evaluation of frozen R6 reference models on two public sealed matches."""
import datetime,hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_idsse import REVISION,REPO,verify_xml,safe_error
from scripts.research_r7 import js,write_new,record,source_identity,model_path,eval_model
from solution.idsse import prepare,digest
from solution.modeling import load_prepared

ROOT=Path('local_artifacts/r7/holdout-v1');IDS=['J03WMX','J03WN1']
def name(m,kind):return ('DFL_02_01_matchinformation'if kind=='matchinformation'else'DFL_04_03_positions_raw_observed')+f'_DFL-COM-000001_DFL-MAT-{m}.xml'
def main():
    protocol=json.loads(Path('docs/R7_HOLDOUT_PROTOCOL.json').read_text())
    for p,sha in protocol['frozen_models'].items():
        if digest(p)!=sha:raise ValueError('frozen holdout model changed')
    if (ROOT/'evaluation_started.json').exists():raise FileExistsError('sealed evaluation already started; no automatic rescoring')
    ROOT.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}'
    with urllib.request.urlopen(url,timeout=30)as r:entries=json.load(r)
    files={item['path']:item for item in entries};manifest={};checks={}
    for m in IDS:
        for kind in ['matchinformation','positions']:
            filename=name(m,kind);meta=files[filename];dest=ROOT/'raw'/filename
            sha=meta['lfs']['oid'] if 'lfs' in meta else None;size=meta['size']
            if not dest.exists():
                code='from huggingface_hub import hf_hub_download;hf_hub_download(repo_id='+repr(REPO)+',repo_type="dataset",revision='+repr(REVISION)+',filename='+repr(filename)+',local_dir='+repr(str(ROOT/'raw'))+')'
                subprocess.run([sys.executable,'-c',code],check=True,timeout=900)
            info=verify_xml(dest,size,sha256=sha,kind=kind)
            if sha is None:
                blob=b'blob '+str(size).encode()+b'\0'+dest.read_bytes()
                if hashlib.sha1(blob).hexdigest()!=meta['oid']:raise ValueError('Git blob digest mismatch')
            manifest[str(dest)]={**info,'pinned_git_or_lfs_oid':meta.get('lfs',{}).get('oid',meta['oid'])}
            print('verified public sealed file',m,kind,flush=True)
        out=ROOT/'prepared'/m
        if not out.exists():prepare(ROOT/'raw'/name(m,'positions'),ROOT/'raw'/name(m,'matchinformation'),out)
        from floodlight.io.dfl import read_position_data_xml
        xy,poss,status,_,_=read_position_data_xml(ROOT/'raw'/name(m,'positions'),ROOT/'raw'/name(m,'matchinformation'))
        checks[m]=[]
        with np.load(out/'raw.npz',allow_pickle=False)as raw:
            for index,half in enumerate(['firstHalf','secondHalf'],1):
                mask=raw['period']==index
                assert np.array_equal(raw['ball_status'][mask],status[half].code)
                assert np.array_equal(raw['ball_possession'][mask],poss[half].code)
                assert np.allclose(raw['ball_xy'][mask],xy[half]['Ball'].xy,equal_nan=True)
                checks[m].append(dict(period=half,frames=int(mask.sum()),labels_and_xy_equal=True))
        del xy,poss,status
    write_new(ROOT/'acquisition.json',dict(revision=REVISION,files=manifest,independent_checks=checks))
    identity=dict(**source_identity(),holdout_protocol_sha256=digest('docs/R7_HOLDOUT_PROTOCOL.json'),holdout_script_sha256=digest(__file__))
    write_new(ROOT/'evaluation_started.json',dict(**identity,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    started=time.monotonic();rows=[]
    for m in IDS:
        for variant in ['M1','M2','M3']:
            model=joblib.load(model_path('J03WPY',variant))
            for domain in ['natural','dropout']:
                b=load_prepared(ROOT/'prepared'/m/f'{domain}.npz')
                if str(b['match_id'])!=m or str(b['label_semantics'])!='DFL_source_complete':raise ValueError('holdout source mismatch')
                score,arrays=eval_model(model,b,variant)
                np.savez_compressed(ROOT/f'{m}-{variant}-{domain}.npz',**arrays)
                row=dict(match=m,variant=variant,domain=domain,metrics=score)
                write_new(ROOT/f'{m}-{variant}-{domain}.json',row);rows.append(row)
                print('one-shot',m,variant,domain,score['decoded']['full_spell_f1'],flush=True)
    result=dict(experiment_id='E06_one_shot_sealed',**identity,rows=rows,files=manifest,checks=checks,wall_s=time.monotonic()-started,competition_promoted=False,decision='one-shot diagnostic; no changes based on these results')
    write_new(ROOT/'results.json',result);write_new('docs/R7_HOLDOUT_RESULTS.json',result)
    record({k:v for k,v in result.items()if k not in {'rows','files','checks'}}|dict(next='preserve sealed evidence; any further method development requires new data'))
if __name__=='__main__':main()
