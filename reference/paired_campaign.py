"""Matched VGAS evaluation with explicit initial states and native horizons."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,fcntl,math
ROOT=Path(__file__).resolve().parent
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(ROOT/'libero-config'),WANDB_MODE='disabled',OMP_NUM_THREADS='4',HF_HUB_DISABLE_TELEMETRY='1')
sys.path[:0]=[str(ROOT/'upstream'),str(ROOT/'LIBERO')]
import numpy as np
import torch
import gymnasium as gym
from scripts.eval_qc_bestofn import LiberoEnvCompat
from smolvla_qchunk.eval.bestofn_eval import build_policy_eval_bundle,evaluate_policy_with_best_of_n
from lerobot.envs.libero import LiberoEnv,get_libero_dummy_action,TASK_SUITE_MAX_STEPS
from my_harness import install
from reset_repair import install as install_reset_repair
install_reset_repair()
H={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}
# Enforce the declared benchmark protocol before any environment is built.
TASK_SUITE_MAX_STEPS.update(H)
CRITICS={'libero_spatial':'spatial','libero_object':'object','libero_goal':'goal','libero_10':'long'}
CONTEXT={}
original_reset=LiberoEnv.reset;original_step=LiberoEnv.step

def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,default=str)+'\n');tmp.replace(p)

def status(state,**kw):
 atomic(ROOT/'campaign-status.json',dict(state=state,time=time.time(),**kw));print(state,kw,flush=True)

def audited_reset(self,seed=None,**kw):
 if not CONTEXT.get('active') or seed is None:
  self._campaign_record=None
  return original_reset(self,seed=seed,**kw)
 assert self._max_episode_steps==H[CONTEXT['suite']], 'Native horizon differs from protocol'
 trial=int(seed)-CONTEXT['seed'];assert 0<=trial<CONTEXT['n']
 if CONTEXT['split']=='eval':
  # Shared repair: installed LeRobot applies set_init_state BEFORE reset, which
  # discards it. Reorder to ensure that declared evaluation states are loaded.
  gym.Env.reset(self,seed=seed);self._env.seed(seed);raw=self._env.reset()
  source=np.asarray(self._init_states[trial]);source_hash=hashlib.sha256(source.tobytes()).hexdigest()
  raw=self._env.set_init_state(source)
  for _ in range(self.num_steps_wait):raw,_,_,_=self._env.step(get_libero_dummy_action())
  obs,info=self._format_raw_obs(raw),{'is_success':False}
 else:
  obs,info=original_reset(self,seed=seed,**kw);source_hash=None
 actual=np.asarray(self._env.env.sim.get_state().flatten())
 rec=dict(seed=int(seed),state_index=trial if CONTEXT['split']=='eval' else None,initial_state_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),fixture_pose_sha256=hashlib.sha256(np.asarray(self._env.env.sim.model.body_pos).tobytes()+np.asarray(self._env.env.sim.model.body_quat).tobytes()).hexdigest(),declared_state_sha256=source_hash,steps=0,success=False,simulator_seconds=0.,reset_settling_actions=self.num_steps_wait,horizon=self._max_episode_steps)
 self._campaign_record=rec;CONTEXT['episodes'].append(rec)
 return obs,info

def audited_step(self,action):
 rec=getattr(self,'_campaign_record',None)
 if rec is not None and rec['steps']>=rec['horizon']:raise RuntimeError('Horizon overrun')
 t=time.perf_counter();result=original_step(self,action);elapsed=time.perf_counter()-t
 if rec is not None:
  rec['steps']+=1;rec['simulator_seconds']+=elapsed;rec['success']|=bool(result[4].get('is_success',False))
 return result
LiberoEnv.reset=audited_reset;LiberoEnv.step=audited_step

def build(suite):
 cfg=LiberoEnvCompat(task=suite,selected_task_ids=(0,),episode_length=H[suite])
 b=build_policy_eval_bundle(policy_path=ROOT/'checkpoints/smolvla/5_SHOT/pretrained_model',critic_state_path=ROOT/f'checkpoints/{CRITICS[suite]}/vgas_critic/last.ckpt',sample_env_cfg=cfg,sample_batch_size=1,device='cuda',best_of_n=8,n_action_steps=20,critic_q_agg_override='min',use_current_critic=True)
 for p in b.policy.parameters():p.requires_grad_(False)
 return b

def run(b,suite,task,seed,n,split,arm,batch_size):
 target=ROOT/'runs'/split/arm/f'{suite}-t{task}-seed{seed}.json'
 if target.exists():return json.loads(target.read_text())
 status('running',split=split,arm=arm,suite=suite,task=task,seed=seed,episodes=n)
 p=b.policy;orig_select=p.select_action;orig_predict=p.predict_action_chunk;orig_embed=p.model.embed_prefix
 metrics=dict(policy_forward_calls=0,policy_candidate_rows=0,policy_seconds=0.,vlm_prefix_calls=0,vlm_prefix_rows=0,vlm_input_padded_tokens=0,vlm_input_valid_tokens=0,external_api_calls=0,external_api_tokens=0)
 def predict(*a,**kw):
  batch=a[0];tensor=next(x for x in batch.values() if isinstance(x,torch.Tensor))
  metrics['policy_forward_calls']+=1;metrics['policy_candidate_rows']+=int(tensor.shape[0])
  torch.cuda.synchronize();t=time.perf_counter()
  try:return orig_predict(*a,**kw)
  finally:torch.cuda.synchronize();metrics['policy_seconds']+=time.perf_counter()-t
 def embed(*a,**kw):
  out=orig_embed(*a,**kw);pad=out[1]
  metrics['vlm_prefix_calls']+=1;metrics['vlm_prefix_rows']+=pad.shape[0];metrics['vlm_input_padded_tokens']+=pad.numel();metrics['vlm_input_valid_tokens']+=int(pad.sum().item())
  return out
 p.predict_action_chunk=predict;p.model.embed_prefix=embed
 if arm=='bare_policy':p.select_action=type(p).select_action.__get__(p,type(p))
 stats,undo=install(p,b.critic_trainer) if arm=='my_harness' else ({},lambda:None)
 CONTEXT.clear();CONTEXT.update(active=True,seed=seed,n=n,split=split,suite=suite,episodes=[])
 torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 try:
  cfg=LiberoEnvCompat(task=suite,selected_task_ids=(task,),episode_length=H[suite])
  info=evaluate_policy_with_best_of_n(policy_path=ROOT/'checkpoints/smolvla/5_SHOT/pretrained_model',critic_state_path=None if arm=='bare_policy' else ROOT/f'checkpoints/{CRITICS[suite]}/vgas_critic/last.ckpt',env_cfg=cfg,n_episodes=n,batch_size=batch_size,seed=seed,device='cuda',best_of_n=1 if arm=='bare_policy' else 8,n_action_steps=20,critic_q_agg_override='min',policy_bundle=b,max_render=0)
  elapsed=time.perf_counter()-t
  rows=CONTEXT['episodes'];assert len(rows)==n,(len(rows),n)
  assert {r['seed'] for r in rows}==set(range(seed,seed+n))
  assert all(r['steps']<=H[suite] and r['steps']>0 for r in rows)
  sr=sum(r['success'] for r in rows)/n*100
  assert abs(sr-info['overall']['pc_success'])<1e-5,(sr,info)
  result=dict(arm=arm,suite=suite,task=task,seed=seed,split=split,n=n,episodes=rows,successes=sum(r['success'] for r in rows),wall_seconds=elapsed,metrics=metrics,cache_stats=stats,peak_allocated_bytes=torch.cuda.max_memory_allocated(),horizon_violations=0,source='authors VGAS algorithm; shared explicit-state reset and spatial220 horizon repairs; batch size5 instead of released50')
  atomic(target,result);return result
 finally:
  CONTEXT['active']=False;undo();p.select_action=orig_select;p.predict_action_chunk=orig_predict;p.model.embed_prefix=orig_embed;p.reset()

def summarize():
 results=[json.loads(p.read_text()) for p in (ROOT/'runs/eval').glob('*/*.json')]
 out={'arms':{},'pairs':0,'paired_authors_successes':0,'paired_my_harness_successes':0,'my_only_wins':0,'authors_only_wins':0}
 for arm in ['bare_policy','authors','my_harness']:
  rs=[r for r in results if r['arm']==arm];n=sum(r['n'] for r in rs);s=sum(r['successes'] for r in rs)
  out['arms'][arm]=dict(episodes=n,successes=s,success_rate=s/n if n else None,wall_seconds=sum(r['wall_seconds'] for r in rs),horizon_violations=sum(r['horizon_violations'] for r in rs),vlm_prefix_rows=sum(r['metrics']['vlm_prefix_rows'] for r in rs))
 lookup={(r['arm'],r['suite'],r['task'],r['seed']):r for r in results}
 for r in results:
  if r['arm']!='authors':continue
  q=lookup.get(('my_harness',r['suite'],r['task'],r['seed']))
  if q is None:continue
  a={e['seed']:e for e in r['episodes']};b={e['seed']:e for e in q['episodes']}
  assert a.keys()==b.keys()
  for k in a:
   assert a[k]['initial_state_sha256']==b[k]['initial_state_sha256'],'Unmatched actual initial states'
   assert a[k]['declared_state_sha256']==b[k]['declared_state_sha256']
   assert a[k]['fixture_pose_sha256']==b[k]['fixture_pose_sha256'],'Unmatched fixed fixture poses'
   x,y=a[k]['success'],b[k]['success'];out['pairs']+=1;out['paired_authors_successes']+=x;out['paired_my_harness_successes']+=y;out['my_only_wins']+=int(y and not x);out['authors_only_wins']+=int(x and not y)
 atomic(ROOT/'comparison.json',out);return out

def main():
 lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Development only: fresh seeded resets, disjoint from final saved-state bank.
 b=build('libero_goal')
 dev_results={}
 for arm in ['authors','my_harness']:
  dev_results[arm]=run(b,'libero_goal',0,900200,5,'dev',arm,5)
 da={e['seed']:e for e in dev_results['authors']['episodes']};dm={e['seed']:e for e in dev_results['my_harness']['episodes']}
 assert da.keys()==dm.keys()
 assert all(da[k]['initial_state_sha256']==dm[k]['initial_state_sha256'] for k in da), 'Unmatched development states'
 assert all(e['horizon']==300 and e['steps']<=300 for r in dev_results.values() for e in r['episodes'])
 del b;torch.cuda.empty_cache()
 status('development_complete')
 assert json.loads((ROOT/'preflight.json').read_text())['api_inference'] is False
 files=['paired_campaign.py','my_harness.py','validate_equivalence.py','preflight.json','pip-freeze.txt','assets-metadata.json','horizon-validation.json','reset_repair.py','reset-bank-validation.json']
 validation=json.loads((ROOT/'reset-bank-validation.json').read_text());assert validation['passed']
 assert {r['state_index'] for r in validation['checks'] if r['suite']=='libero_spatial' and r['task']==4}==set(range(50))
 manifest=dict(protocol_version=2,upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip(),libero_commit=subprocess.check_output(['git','-C',str(ROOT/'LIBERO'),'rev-parse','HEAD'],text=True).strip(),files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},assets_revision=json.loads((ROOT/'assets-metadata.json').read_text())['revision'],eval_seeds=[0,42,1234,124,410,2000],episodes_per_task_seed=50,tasks_per_suite=10,task_order='Spatial4 first to validate the formerly affected condition, then the remaining task IDs; other suites ascending',suites=list(H),horizons=H,horizon_multiplier=1,settling_actions_excluded=10,action_candidates=8,execute_actions=20,batch_size=5,shared_repairs=['reset before set_init_state; explicitly choose all 50 released states','explicit native horizon table Spatial220/Object280/Goal300/Long520; installed Spatial280 overridden','rebuild BDDL-generated property samplers each reset; prevent accumulation across auto resets'],paper_parity_gaps=['paper main-table seed list unspecified; six seeds from release launcher','batch5 vs release50','shared state reset repair'],candidate='reuse identical prefix across candidates; BF16 batch kernels produce numerical differences, measured on development only',test_time_learning=False,api_inference=False)
 frozen=ROOT/'frozen.json'
 if frozen.exists():assert json.loads(frozen.read_text())==manifest,'Changed frozen experiment'
 else:atomic(frozen,manifest)
 # Only the authors and my harness run the full paired evaluation. Bare policy
 # remains separately labelled and is measured on the first seed for each task.
 for suite in H:
  b=build(suite)
  for seed in manifest['eval_seeds']:
   for task in ([4,0,1,2,3,5,6,7,8,9] if suite=='libero_spatial' else range(10)):
    arms=['authors','my_harness'] if (task+seed)%2==0 else ['my_harness','authors']
    if seed==0:arms+=['bare_policy']
    for arm in arms:
     run(b,suite,task,seed,50,'eval',arm,5);summarize()
  del b;torch.cuda.empty_cache()
 status('complete',comparison=summarize())
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e));raise
