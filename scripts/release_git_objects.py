"""Publish audited local Git objects without rewriting commit identities or force-pushing.

Fallback for blocked Git transport. Requires authenticated gh; no credentials are read
or printed by this script. Object POSTs are immutable; the branch is created last.
"""
import argparse,base64,concurrent.futures,datetime,json,os,re,subprocess
from pathlib import Path
from scripts.audit_public_release import BASE,audit

REPO='dlr9/more-than-a-hack-2027-msn'
def git(*args):return subprocess.check_output(['git',*args],env={**os.environ,'GIT_NO_LAZY_FETCH':'1'})
def api(endpoint,payload=None):
    args=['gh','api',f'repos/{REPO}/'+endpoint]
    if payload is not None:args+=['--method','POST','--input','-']
    result=subprocess.run(args,input=None if payload is None else json.dumps(payload),capture_output=True,text=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr[:800])
    return json.loads(result.stdout)
def identity(line):
    name,email,stamp,zone=re.fullmatch(r'(.*) <(.*)> (\d+) ([+-]\d{4})',line).groups()
    minutes=(int(zone[1:3])*60+int(zone[3:]))*(1 if zone[0]=='+'else -1)
    date=datetime.datetime.fromtimestamp(int(stamp),datetime.timezone(datetime.timedelta(minutes=minutes))).isoformat()
    return dict(name=name,email=email,date=date)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    if not a.branch.startswith('codex/'):raise ValueError('new codex branch required')
    if git('status','--porcelain').strip():raise ValueError('commit all reviewed work first')
    checked=audit();head=checked['head']
    if api(f'git/commits/{BASE}')['sha']!=BASE:raise ValueError('remote source ancestor mismatch')
    # Existing branch is never moved by this publisher; use non-force git/API separately.
    branches=api('branches?per_page=100')
    if any(b['name']==a.branch for b in branches):raise FileExistsError('branch already exists; inspect before any update')
    raw=git('rev-list','--objects',head,'^'+BASE).decode().splitlines()
    ids=[line.split()[0]for line in raw]
    kinds={sha:git('cat-file','-t',sha).decode().strip()for sha in ids}
    blobs=[sha for sha in ids if kinds[sha]=='blob'];uploaded=set()
    def blob(sha):
        data=git('cat-file','blob',sha)
        actual=api('git/blobs',dict(content=base64.b64encode(data).decode(),encoding='base64'))['sha']
        if actual!=sha:raise ValueError('blob SHA mismatch')
        return sha
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for sha in pool.map(blob,blobs):uploaded.add(sha)
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
    commits=git('rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()
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
    ref=api('git/refs',dict(ref='refs/heads/'+a.branch,sha=head))
    if ref['object']['sha']!=head:raise ValueError('published head mismatch')
    receipt=dict(repo=REPO,branch=a.branch,head=head,base_source=BASE,objects=len(uploaded),commits=commits,force=False,merge=False)
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print('Published new branch',a.branch,head,flush=True)
if __name__=='__main__':main()
