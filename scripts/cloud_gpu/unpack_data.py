#!/usr/bin/env python3
"""Install only manifest-listed dataset members; validate bytes before rename."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def install(archive=None,root=ROOT,datasets=None):
    root=Path(root);manifest=json.loads((ROOT/'datasets/manifest.json').read_text())['datasets'];selected=datasets or list(manifest)
    tar=tarfile.open(archive,'r:*') if archive else None
    try:
        for name in selected:
            for split,row in manifest[name]['files'].items():
                target=root/row['target']
                if target.exists():
                    if target.stat().st_size!=row['size'] or sha(target)!=row['sha256']:raise ValueError('Existing data differs; refusing overwrite: '+str(target))
                elif tar:
                    member=tar.getmember(row['target'])
                    if not member.isfile() or member.size!=row['size']:raise ValueError('Invalid dataset member '+row['target'])
                    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp')
                    try:
                        with tar.extractfile(member) as src,temp.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                        if sha(temp)!=row['sha256']:raise ValueError('Dataset SHA256 mismatch '+row['target'])
                        temp.replace(target)
                    finally:
                        if temp.exists():temp.unlink()
                else:raise FileNotFoundError('Dataset missing; run unpack_data.py --archive <migration.tar>: '+str(target))
                print(f'OK {name}/{split} {row["sha256"]}',flush=True)
    finally:
        if tar:tar.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--datasets',nargs='+',choices=['taxibj','bikenyc','chap']);a=p.parse_args();install(a.archive,a.data_root,a.datasets)
