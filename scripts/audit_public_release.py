"""Read only allowlisted public text in the proposed Git increment; reject other paths first."""
import argparse, hashlib, json, re, subprocess
from pathlib import PurePosixPath

BASE='1d62c7219868dfcbf98921fd0c7464f10dbed55b'
EXACT={'.gitignore','docs/R10_V3_DECISIONS.jsonl','.github/workflows/public-solution.yml','requirements-idsse-tools.txt','docs/FEATURE_PROVENANCE.csv','docs/R6_EXPERIMENT_LEDGER.jsonl','docs/R8_EXPERIMENT_LEDGER.jsonl','docs/R9_V2_EXPERIMENT_LEDGER.jsonl'}
SECRETS=[re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
         re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
         re.compile(rb'(?:AKIA|ASIA)[A-Z0-9]{16}'),re.compile(rb'https?://[^\s/:]+:[^\s/@]+@')]

def allowed(path):
    p=PurePosixPath(path)
    if '..'in p.parts or p.is_absolute():return False
    if path in EXACT:return True
    if len(p.parts)==2 and p.parts[0]in{'solution','scripts','tests'} and p.suffix in{'.py','.command'}:return True
    if len(p.parts)==2 and p.parts[0]=='docs' and p.name.startswith(('R6_','R7_','R7B_','R8_','R9_V2_','R10_V3_','R11_','R11P5_','R12_','R13_')) and p.suffix in{'.json','.md'}:return True
    if p.parts and p.parts[0]=='r10' and len(p.parts) in (2,3) and p.suffix in {'.py','.md'} and (len(p.parts)==2 or p.parts[1]=='adapters'):return True
    if len(p.parts)==2 and p.parts[0] in {'r11','r11p5','r12'} and p.suffix=='.py':return True
    return False

def inspect_blob(path,content):
    if not allowed(path):raise ValueError('non-allowlisted public path: '+path)
    if len(content)>5_000_000 or b'\0'in content:raise ValueError('unexpected large/binary public file: '+path)
    content.decode('utf-8')
    if any(r.search(content)for r in SECRETS):raise ValueError('credential-like content: '+path)
    # These serialization signatures are raw tracking/target outputs, not program references.
    if content.lstrip().startswith((b'<FrameSet',b'<Tracab',b'frame,label\n',b'frame,label\r\n')):
        raise ValueError('raw/derived data content: '+path)

def git(*args):return subprocess.check_output(['git',*args])
def audit(base=BASE,head='HEAD'):
    commits=git('rev-list','--reverse',f'{base}..{head}').decode().splitlines()
    inspected={};changes=[]
    for commit in commits:
        paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines()
        for path in paths:
            if not allowed(path):raise ValueError('stop before reading non-allowlisted history path: '+path)
            # Resolve entries without opening files; deletion has no new blob.
            entry=git('ls-tree',commit,'--',path).decode().strip()
            if not entry:continue
            mode,kind,sha=entry.split('\t')[0].split()
            if kind!='blob'or mode not in{'100644','100755'}:raise ValueError('nonregular Git entry')
            if sha not in inspected:
                data=git('cat-file','blob',sha);inspect_blob(path,data)
                inspected[sha]=dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            changes.append(dict(commit=commit,path=path,blob=sha))
    # Official scorer must be byte-identical to the preserved PR2 source baseline.
    scorer='evaluation/possession_metrics.py'
    if git('rev-parse',f'{base}:{scorer}')!=git('rev-parse',f'{head}:{scorer}'):raise ValueError('official scorer changed')
    return dict(base=base,head=git('rev-parse',head).decode().strip(),commits=len(commits),blobs=len(inspected),changes=changes,inspected=inspected,official_scorer_unchanged=True,
                limits='Allowlisted source and aggregate files were inspected; pattern checks supplement human review, not proof of all possible secrets.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default=BASE);p.add_argument('--head',default='HEAD');p.add_argument('--out',required=True);a=p.parse_args()
    result=audit(a.base,a.head)
    with open(a.out,'x')as f:json.dump(result,f,indent=2)
    print('Public incremental history audit passed:',result['commits'],'commits,',result['blobs'],'text blobs')
