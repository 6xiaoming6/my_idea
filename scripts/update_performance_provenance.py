"""Explicitly separate modified performance files from unchanged release sources."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/'docs/provenance.json';original_path=ROOT/'docs/provenance_original_i37.json'
 if not original_path.exists():original_path.write_bytes(path.read_bytes())
 original=json.loads(original_path.read_text());p=json.loads(original_path.read_text())
 unchanged={};modified={}
 for name,old in original['unchanged_source_sha256'].items():
  now=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
  if old==now:unchanged[name]=old
  else:modified[name]=dict(reference_sha256=old,candidate_sha256=now)
 p['unchanged_source_sha256']=unchanged;p['performance_modified_source_sha256']=modified
 p['performance_added_source_sha256']={'src/stmoe_imputer/performance.py':hashlib.sha256((ROOT/'src/stmoe_imputer/performance.py').read_bytes()).hexdigest()}
 p['performance_base_commit']='6caef6740bc00013a65d70dd0a98083648e1f88d';p['source_kind']='Original I37 release plus explicitly tracked opt-in execution optimizations; numerical equivalence is separately tested.'
 path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
