"""Read-only, matched-coverage statistics; no model inference."""
from pathlib import Path
import json,math,random
ROOT=Path(__file__).resolve().parent
rows=[json.loads(p.read_text()) for p in (ROOT/'runs/eval').glob('*/*.json')]
lookup={(r['arm'],r['suite'],r['task'],r['seed']):r for r in rows}

def aggregate(rs):
 n=sum(r['n'] for r in rs);s=sum(r['successes'] for r in rs)
 if n:
  z=1.95996398454;p=s/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;ci=[center-half,center+half]
 else:ci=None
 return dict(episodes=n,successes=s,success_rate=s/n if n else None,binomial_wilson_95=ci,wall_seconds=sum(r['wall_seconds'] for r in rs),policy_seconds=sum(r['metrics']['policy_seconds'] for r in rs),simulator_seconds=sum(e['simulator_seconds'] for r in rs for e in r['episodes']),vlm_prefix_calls=sum(r['metrics']['vlm_prefix_calls'] for r in rs),vlm_prefix_rows=sum(r['metrics']['vlm_prefix_rows'] for r in rs),vlm_input_valid_tokens=sum(r['metrics']['vlm_input_valid_tokens'] for r in rs),external_api_calls=sum(r['metrics']['external_api_calls'] for r in rs),external_api_tokens=sum(r['metrics']['external_api_tokens'] for r in rs),horizon_violations=sum(r['horizon_violations'] for r in rs))

pairs=[]
for r in rows:
 if r['arm']!='authors':continue
 q=lookup.get(('my_harness',r['suite'],r['task'],r['seed']))
 if q:
  x={e['seed']:e for e in r['episodes']};y={e['seed']:e for e in q['episodes']}
  assert x.keys()==y.keys()
  assert all(x[k]['initial_state_sha256']==y[k]['initial_state_sha256'] for k in x)
  assert all(x[k]['fixture_pose_sha256']==y[k]['fixture_pose_sha256'] for k in x)
  pairs.append((r,q))
matched={'authors':aggregate([r for r,q in pairs]),'my_harness':aggregate([q for r,q in pairs])}
only_mine=only_authors=0
for r,q in pairs:
 x={e['seed']:e for e in r['episodes']};y={e['seed']:e for e in q['episodes']}
 only_mine+=sum(y[k]['success'] and not x[k]['success'] for k in x)
 only_authors+=sum(x[k]['success'] and not y[k]['success'] for k in x)
matched['my_only_wins']=only_mine;matched['authors_only_wins']=only_authors
clusters={}
for r,q in pairs:
 key=(r['suite'],r['task'])
 d,n=clusters.get(key,(0,0));clusters[key]=(d+q['successes']-r['successes'],n+r['n'])
if len(clusters)>=5:
 by_suite={}
 for (suite,task),value in clusters.items():by_suite.setdefault(suite,[]).append(value)
 rng=random.Random(731);deltas=[]
 for _ in range(2000):
  chosen=[rng.choice(values) for values in by_suite.values() for _ in range(len(values))]
  deltas.append(sum(d for d,n in chosen)/sum(n for d,n in chosen))
 deltas.sort();matched['success_rate_difference_my_minus_authors']=sum(d for d,n in clusters.values())/sum(n for d,n in clusters.values())
 matched['task_cluster_bootstrap_95']=[deltas[49],deltas[1949]]
 matched['bootstrap_clusters']=len(clusters)
else:matched['task_cluster_bootstrap_95']=None
out=dict(coverage={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ['bare_policy','authors','my_harness']},matched=matched,per_suite={s:{arm:aggregate([r if arm=='authors' else q for r,q in pairs if r['suite']==s]) for arm in ['authors','my_harness']} for s in ['libero_spatial','libero_object','libero_goal','libero_10']},bare_policy_matched_seed0={arm:aggregate([r for r in rows if r['arm']==arm and r['seed']==0 and ('bare_policy',r['suite'],r['task'],0) in lookup and ('authors',r['suite'],r['task'],0) in lookup and ('my_harness',r['suite'],r['task'],0) in lookup]) for arm in ['bare_policy','authors','my_harness']},uncertainty_note='Wilson intervals treat episodes as independent; repeated tasks/states induce correlation. Do not use these alone as evidence of benchmark superiority.',timing_note='Shared GPU; compare matched coverage and alternate arm order. Wall time includes simulator/reset/processing; policy and simulator clocks reported separately.')
(ROOT/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
