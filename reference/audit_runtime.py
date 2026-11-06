"""Read-only integrity and early-result audit of the active campaign."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
frozen=json.loads((ROOT/'frozen.json').read_text())
for name,expected in frozen['files'].items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, f'Frozen file changed: {name}'
status=json.loads((ROOT/'campaign-status.json').read_text())
rs=[]
for p in (ROOT/'runs/eval').glob('*/*.json'):
 r=json.loads(p.read_text());assert len(r['episodes'])==r['n']
 assert all(e['horizon']==frozen['horizons'][r['suite']] and e['steps']<=e['horizon'] for e in r['episodes'])
 assert len({e['state_index'] for e in r['episodes']})==50
 assert all(e['declared_state_sha256'] for e in r['episodes'])
 rs.append(r)
print(json.dumps(dict(frozen_files_verified=True,status=status,completed_groups=len(rs),completed_episodes=sum(r['n'] for r in rs),horizon_violations=0),indent=2))
