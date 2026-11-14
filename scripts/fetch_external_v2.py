"""Download only pinned, public OTHER matches; never the sealed test matches."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('local_artifacts/external')
REF = '4340d274572876239c154c90bc507a9b3250a656'
MATCHES = ('1874553', '1886347', '1899585', '1927964')

def fetch(item):
    url, path = item
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    pointer = None
    if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
        pointer = dict(line.split(' ', 1) for line in content.decode().splitlines()[1:])
        url = url.replace('raw.githubusercontent.com/', 'media.githubusercontent.com/media/')
        with urllib.request.urlopen(url, timeout=180) as response:
            content = response.read()
        assert len(content) == int(pointer['size'])
        assert hashlib.sha256(content).hexdigest() == pointer['oid'].split(':')[1]
    path.write_bytes(content)
    row = dict(url=url, path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), lfs_verified=pointer is not None)
    print(json.dumps(row), flush=True)
    return row

def main():
    jobs = []
    for match in MATCHES:
        for suffix in ('match.json', 'phases_of_play.csv', 'dynamic_events.csv', 'tracking_extrapolated.jsonl'):
            name = f'{match}_{suffix}'
            jobs.append((f'https://raw.githubusercontent.com/SkillCorner/opendata/{REF}/data/matches/{match}/{name}', ROOT / 'skillcorner' / match / name))
    mref = 'e706dd506b360d69d9d123d5b8026e7294b13996'
    for suffix in ('events.json', 'metadata.xml', 'tracking.txt'):
        name = f'Sample_Game_3_{suffix}'
        jobs.append((f'https://raw.githubusercontent.com/metrica-sports/sample-data/{mref}/data/Sample_Game_3/{name}', ROOT / 'metrica' / name))
    rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, item): item[0] for item in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(url=futures[future], error=str(error)))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'manifest.json').write_text(json.dumps(dict(files=rows, failures=failures), indent=2))
    if failures:
        print(json.dumps(failures), flush=True)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
