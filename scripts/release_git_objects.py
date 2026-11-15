"""Publish audited local Git objects without rewriting commit identities or force-pushing.

Fallback for blocked Git transport. Requires authenticated gh; no credentials are read
or printed by this script. Object POSTs are immutable; the branch is created last.
"""
import argparse,base64,concurrent.futures,datetime,json,os,re,subprocess
from pathlib import Path
from scripts.audit_public_release import BASE,audit

REPO='dlr9/more-than-a-hack-2027-msn'
def git(*args):return subprocess.check_output(['git',*args],env={**os.environ,'GIT_NO_LAZY_FETCH':'1'})
def api(endpoint,payload=None,method='POST'):
    args=['gh','api',f'repos/{REPO}/'+endpoint]
    if payload is not None:args+=['--method',method,'--input','-']
    result=subprocess.run(args,input=None if payload is None else json.dumps(payload),capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr[:800])
    return json.loads(result.stdout)
def identity(line):
    name,email,stamp,zone=re.fullmatch(r'(.*) <(.*)> (\d+) ([+-]\d{4})',line).groups()
    minutes=(int(zone[1:3])*60+int(zone[3:]))*(1 if zone[0]=='+'else -1)
    date=datetime.datetime.fromtimestamp(int(stamp),datetime.timezone(datetime.timedelta(minutes=minutes))).isoformat()
    return dict(name=name,email=email,date=date)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--uploaded-base',default=BASE,help='Previously verified remote ancestor; full public history audit still applies')
    p.add_argument('--update-existing',action='store_true',help='Only fast-forward an existing branch after local ancestry verification')
    p.add_argument('--workers',type=int,default=4,choices=range(1,5),help='Bounded blob upload concurrency; use1 to diagnose transient concurrent API failures')
    a=p.parse_args();uploaded_base=a.uploaded_base
    if not a.branch.startswith('codex/'):raise ValueError('new codex branch required')
    if git('status','--porcelain').strip():raise ValueError('commit all reviewed work first')
    checked=audit();head=checked['head']
    subprocess.run(['git','merge-base','--is-ancestor',uploaded_base,head],check=True)
    if api(f'git/commits/{uploaded_base}')['sha']!=uploaded_base:raise ValueError('remote source ancestor mismatch')
    # Existing branch moves require explicit opt-in and both local/server ancestry checks.
    branches=api('branches?per_page=100')
    current=next((b for b in branches if b['name']==a.branch),None)
    if current is not None:
        if not a.update_existing:raise FileExistsError('branch already exists; inspect before any update')
        subprocess.run(['git','merge-base','--is-ancestor',current['commit']['sha'],head],check=True)
    elif a.update_existing:raise ValueError('requested existing branch is absent')
    raw=git('rev-list','--objects',head,'^'+uploaded_base).decode().splitlines()
    ids=[line.split()[0]for line in raw]
    kinds={sha:git('cat-file','-t',sha).decode().strip()for sha in ids}
    blobs=[sha for sha in ids if kinds[sha]=='blob'];uploaded=set()
    def blob(sha):
        data=git('cat-file','blob',sha)
        actual=api('git/blobs',dict(content=base64.b64encode(data).decode(),encoding='base64'))['sha']
        if actual!=sha:raise ValueError('blob SHA mismatch')
        return sha
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:
        for sha in pool.map(blob,blobs):
            uploaded.add(sha)
            print('Verified blob',sha,flush=True)
    print('Verified uploaded blobs',len(blobs),flush=True)
    trees=set(sha for sha in ids if kinds[sha]=='tree')
    def tree(sha):
        if sha not in trees or sha in uploaded:return
        entries=[]
        for entry in git('ls-tree','-z',sha).split(b'\0'):
            if not entry:continue
            header,path=entry.split(b'\t',1);mode,kind,child=header.decode().split()
            if kind=='tree':tree(child)
            entries.append(dict(path=path.decode(),mode=mode,type=kind,sha=child))
        if api('git/trees',dict(tree=entries))['sha']!=sha:raise ValueError('tree SHA mismatch')
        uploaded.add(sha)
    commits=git('rev-list','--reverse',f'{uploaded_base}..{head}').decode().splitlines()
    for sha in commits:
        content=git('cat-file','commit',sha).decode();header,message=content.split('\n\n',1)
        fields={};parents=[]
        for line in header.splitlines():
            key,value=line.split(' ',1)
            if key=='parent':parents.append(value)
            elif key in {'tree','author','committer'}:fields[key]=value
            else:raise ValueError('unsupported signed/extra commit headers; preserve source and stop')
        tree(fields['tree'])
        payload=dict(tree=fields['tree'],parents=parents,message=message,author=identity(fields['author']),committer=identity(fields['committer']))
        actual=api('git/commits',payload)['sha']
        if actual!=sha:raise ValueError('commit SHA mismatch; no branch publication')
        print('Verified commit',sha,flush=True)
    ref=(api('git/refs/heads/'+a.branch,dict(sha=head,force=False),method='PATCH')if current is not None
         else api('git/refs',dict(ref='refs/heads/'+a.branch,sha=head)))
    if ref['object']['sha']!=head:raise ValueError('published head mismatch')
    receipt=dict(repo=REPO,branch=a.branch,head=head,previous_head=current['commit']['sha']if current else None,base_source=BASE,uploaded_base=uploaded_base,objects=len(uploaded),commits=commits,force=False,merge=False)
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print('Published new branch',a.branch,head,flush=True)
if __name__=='__main__':main()
