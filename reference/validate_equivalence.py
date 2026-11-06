"""Real-weight equivalence and alternating latency checks on a dev observation."""
import os,sys,json,time,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(ROOT/'libero-config'),WANDB_MODE='disabled',OMP_NUM_THREADS='4')
sys.path[:0]=[str(ROOT/'upstream'),str(ROOT/'LIBERO')]
import torch
from scripts.eval_qc_bestofn import LiberoEnvCompat
from smolvla_qchunk.eval.bestofn_eval import build_policy_eval_bundle,_sample_policy_batch
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from my_harness import install
cfg=LiberoEnvCompat(task='libero_goal',selected_task_ids=(0,),episode_length=300)
bundle=build_policy_eval_bundle(policy_path=ROOT/'checkpoints/smolvla/5_SHOT/pretrained_model',critic_state_path=ROOT/'checkpoints/goal/vgas_critic/last.ckpt',sample_env_cfg=cfg,sample_batch_size=1,device='cuda',best_of_n=8,n_action_steps=20,critic_q_agg_override='min',use_current_critic=True)
p=bundle.policy;c=bundle.critic_trainer
for param in p.parameters():param.requires_grad_(False)
envs=make_env(cfg,n_envs=1,use_async_envs=False)
batch,_=_sample_policy_batch(envs,bundle.preprocessor,'cuda',p)
close_envs(envs)
records=[]
for trial in range(6):
 seed=900100+trial
 order=['authors','my_harness'] if trial%2==0 else ['my_harness','authors']
 outputs={}
 for arm in order:
  p.reset();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  stats,undo=install(p,c) if arm=='my_harness' else ({},lambda:None)
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.no_grad():
   encoding=bundle.encoder_fn(p,batch)
   out=c._best_of_n_actions_soon(p,batch,encoding,return_candidates=True)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  rng=torch.cuda.get_rng_state().clone()
  outputs[arm]=([x.detach().clone() for x in out],rng)
  undo();records.append(dict(trial=trial,arm=arm,seconds=elapsed,stats=stats))
 ref,rng0=outputs['authors'];new,rng1=outputs['my_harness']
 errors=[(a-b).abs().max().item() for a,b in zip(ref,new)]
 assert torch.equal(rng0,rng1),'Changed RNG consumption'
 assert all(torch.isfinite(x).all() for x in new),'Nonfinite actions'
 assert (ref[2]-new[2]).abs().max()<0.2,'Large numerical divergence'
 print('numerical_probe',trial,errors,flush=True)
 records[-1]['max_absolute_errors']=errors
summary=dict(passed=True,numerically_equivalent=False,acceptance='Finite outputs, identical RNG consumption, candidate difference below 0.2 on dev probe; success-rate comparison is required',checkpoint_repository_revision=json.loads((ROOT/'assets-metadata.json').read_text())['revision'],dev_seed_range=[900100,900105],checks=['all candidate actions finite','maximum candidate numerical difference recorded','identical CUDA RNG state'],records=records,median_seconds={arm:statistics.median(r['seconds'] for r in records if r['arm']==arm and r['trial']>0) for arm in ['authors','my_harness']},caveat='shared GPU; same single dev observation; timing is preliminary')
(ROOT/'equivalence.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['median_seconds']),flush=True)
