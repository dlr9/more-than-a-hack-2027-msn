#!/usr/bin/env python3
"""Bounded public IDSSE acquisition. Never downloads sealed research matches.

Run with the optional isolated huggingface_hub dependency on PYTHONPATH.
The pinned manifest was verified against actual downloaded bytes and HF LFS.
Figshare, HF, and floodlight are respectively two origins and three clients;
floodlight is a Figshare diagnostic, not an independent data source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

REVISION = 'a715a38dfbaf5f58e431727c2b78d174101a703c'
REPO = 'pysport/idsse-data'
MANIFEST = {
 'J03WPY': {
  'matchinformation': (51643487,11922,'65a4568e4f374af94d3c1015cbc7c4f5','566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420'),
  'positions': (51643526,371688499,'e56ff1336a334f8da51110fbe7cab764','845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1')},
 'J03WOH': {
  'matchinformation': (51643478,11745,'b9710781d400dd812f62b0a6e3d7bce8','f23b3ee13b988b1ff9a0bce438c416dac47e71355bbc7ee1dab21aaf3e33c18c'),
  'positions': (51643520,348945715,'3efff25a86a61768e7d253b4bba137a3','ec4dcf30200aad694dea263d5289485fcd0ca42badddaeb7c81f5cd5a4229d4f')},
 'J03WQQ': {
  'matchinformation': (51643484,11932,'5979209af316a914ce57347d0633df6c','e976aa0c5dba862cfa898ea57ec226efce9d1bc04b23b8d775908db86aa6fa41'),
  'positions': (51643529,356219423,'9105bad987191d3c3c362c1fce9a4fcd','cbefcb0bd7f39f3b94ee69a7bc97aac0f936425062c1cc1b144733df36804d6f')}}


def filename(match, kind):
    prefix = 'DFL_02_01_matchinformation' if kind == 'matchinformation' else 'DFL_04_03_positions_raw_observed'
    return f'{prefix}_DFL-COM-000002_DFL-MAT-{match}.xml'


def safe_error(error):
    # Do not record signed redirect query strings, access tokens or cookies.
    text = re.sub(r'https?://[^\s\"\'<>]+', lambda m: urlsplit(m.group()).scheme+'://'+urlsplit(m.group()).netloc+urlsplit(m.group()).path, str(error))
    return text[:1600]


def verify_xml(path, size, md5=None, sha256=None, kind=None):
    path = Path(path)
    if path.stat().st_size != size:
        raise ValueError(f'size mismatch: {path.stat().st_size} != {size}')
    hashes = {'md5':hashlib.md5(), 'sha256':hashlib.sha256()}
    with path.open('rb') as source:
        prefix = source.read(256).lstrip()
        if not prefix.startswith(b'<') or b'<html' in prefix.lower() or b'<!doctype html' in prefix.lower():
            raise ValueError('not XML; rejects HTML and LFS/Xet pointer files')
        source.seek(0)
        for block in iter(lambda:source.read(1024*1024), b''):
            for digest in hashes.values(): digest.update(block)
    result = {key:value.hexdigest() for key,value in hashes.items()}
    for name, expected in [('md5',md5),('sha256',sha256)]:
        if expected is not None and result[name] != expected:
            raise ValueError(f'{name} mismatch')
    required = {'positions':'FrameSet','matchinformation':'MatchInformation'}.get(kind)
    found = required is None
    # Clear each completed element: validation is streaming, not a full DOM.
    for _, elem in ET.iterparse(path, events=('end',)):
        found |= elem.tag.rsplit('}',1)[-1] == required
        elem.clear()
    if not found: raise ValueError(f'XML missing required {required}')
    return dict(size=size, **result)


def download_http(url, destination, *, size, md5=None, sha256=None, kind=None,
                  opener=urlopen, timeout=30):
    """One bounded stable-URL attempt; caller chooses a different fallback on failure.

    A 200 response to Range replaces the partial file, never appends. A 206
    response must start exactly at the local byte offset and declare total size.
    Existing final caches are verified and never overwritten.
    """
    destination = Path(destination)
    if destination.exists():
        return verify_xml(destination,size,md5,sha256,kind)
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial = destination.with_suffix(destination.suffix+'.part')
    start = partial.stat().st_size if partial.exists() else 0
    if start == size:
        result = verify_xml(partial,size,md5,sha256,kind)
        partial.replace(destination)
        return result
    if start > size: raise ValueError('partial cache larger than expected; preserve for investigation')
    request = Request(url,headers={'Range':f'bytes={start}-'} if start else {})
    with opener(request,timeout=timeout) as response:
        status = response.status
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or tuple(map(int,match.groups())) != (start,size-1,size):
                raise ValueError('invalid or misaligned Content-Range')
            mode = 'ab' if start else 'wb'
        elif status == 200:
            mode = 'wb'  # Server ignored Range: start over safely.
        else:
            raise ValueError(f'unexpected HTTP status {status}')
        written = start if mode == 'ab' else 0
        with partial.open(mode) as out:
            while block := response.read(1024*1024):
                written += len(block)
                if written > size: raise ValueError('response exceeds declared expected file size')
                out.write(block)
    result = verify_xml(partial,size,md5,sha256,kind)
    partial.replace(destination)
    return dict(http_status=status,**result)


def append_log(path, **record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f: f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**record),sort_keys=True)+'\n')


def figshare_metadata(opener=urlopen):
    with opener('https://api.figshare.com/v2/articles/28196177',timeout=25) as r:
        metadata=json.load(r)
    title=metadata.get('title','').lower()
    license_text=json.dumps(metadata.get('license',{})).lower()
    if 'soccer' not in title or not ('creativecommons.org/licenses/by/4.0' in license_text or 'cc by 4.0' in license_text):
        raise ValueError('unexpected article title or license')
    return metadata


def acquire(match, directory, log):
    if match not in MANIFEST: raise ValueError('only preregistered development matches allowed')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    # Offline recovery of fully verified existing cache performs no network call.
    if all((directory/filename(match,kind)).exists() for kind in MANIFEST[match]):
        for kind,(_,size,md5,sha) in MANIFEST[match].items():
            info=verify_xml(directory/filename(match,kind),size,md5,sha,kind)
            append_log(log,match=match,kind=kind,source='verified_cache',status='OK',**info)
        return
    metadata=None
    try:
        metadata=figshare_metadata()
        append_log(log,source='figshare_metadata',status='OK',title=metadata['title'],license=metadata['license'])
    except ValueError:
        raise  # A reachable source reporting conflicting identity/license fails closed.
    except Exception as error:
        append_log(log,source='figshare_metadata',status='FAILED',error_class=type(error).__name__,error=safe_error(error),next_fallback='pinned verified manifest then Hugging Face')
    for kind,(file_id,size,md5,sha) in MANIFEST[match].items():
        name=filename(match,kind); destination=directory/name
        if destination.exists():
            info=verify_xml(destination,size,md5,sha,kind)
            append_log(log,match=match,kind=kind,source='verified_cache',status='OK',**info)
            continue
        if metadata is not None:
            entry=next(f for f in metadata['files'] if f['id']==file_id)
            if entry['name'] != name or entry['size'] != size or entry['computed_md5'] != md5:
                raise ValueError('Figshare metadata conflicts with pinned file manifest')
        try:
            info=download_http(f'https://ndownloader.figshare.com/files/{file_id}',destination,size=size,md5=md5,sha256=sha,kind=kind)
            append_log(log,match=match,kind=kind,source='figshare',status='OK',**info)
            continue
        except Exception as error:
            append_log(log,match=match,kind=kind,source='figshare',status='FAILED',error_class=type(error).__name__,error=safe_error(error),next_fallback='huggingface')
        # Separate stage prevents an unverified library download replacing final cache.
        # Hugging Face manages resumable .incomplete files internally.
        script = ('from huggingface_hub import hf_hub_download; '
                  'hf_hub_download(repo_id='+repr(REPO)+',repo_type="dataset",revision='+repr(REVISION)+
                  ',filename='+repr(name)+',local_dir='+repr(str(directory/'.hf-stage'))+')')
        try:
            completed=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=900)
        except subprocess.TimeoutExpired as error:
            append_log(log,match=match,kind=kind,source='huggingface',status='FAILED',
                       error_class='DownloadTimeout',error='900 second process budget reached',
                       next_fallback='existing external training; retain incomplete cache')
            raise RuntimeError('HF download exceeded bounded process budget') from error
        if completed.returncode:
            error=safe_error(completed.stderr)
            append_log(log,match=match,kind=kind,source='huggingface',status='FAILED',error_class='HFClientError',error=error,next_fallback='floodlight same-origin diagnostic or existing external training')
            raise RuntimeError(error)
        staged=directory/'.hf-stage'/name
        info=verify_xml(staged,size,md5,sha,kind)
        # Write final .part only after validation; existing cache was checked above.
        partial=destination.with_suffix('.xml.part')
        shutil.copyfile(staged,partial); partial.replace(destination)
        append_log(log,match=match,kind=kind,source='huggingface',revision=REVISION,status='OK',**info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',choices=tuple(MANIFEST),default='J03WPY')
    parser.add_argument('--directory',type=Path,default=Path('local_artifacts/r6/idsse-hf'))
    parser.add_argument('--log',type=Path,default=Path('local_artifacts/r6/download_attempts.jsonl'))
    args=parser.parse_args(); acquire(args.match,args.directory,args.log)

if __name__ == '__main__': main()
