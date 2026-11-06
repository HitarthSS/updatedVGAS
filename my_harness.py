"""Reuse identical visual/language prefix computation across action candidates.

Policy and critic parameters, eight proposals, noise shape, flow steps, scoring,
execution chunk and environment budget stay unchanged. Cache lifetime is one
proposal batch. A failed identical-input check uses the authors' original path.
"""
from types import MethodType
import torch


def _repeat_cache(value, repeats):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_cache(v, repeats) for key,v in value.items()}
    if isinstance(value, list):
        return [_repeat_cache(v, repeats) for v in value]
    if isinstance(value, tuple):
        return tuple(_repeat_cache(v, repeats) for v in value)
    if value is None:return None
    raise TypeError(type(value))


def install(policy, critic):
    from models.smolvla.modeling_smolvla import make_att_2d_masks
    if getattr(policy,'_shared_prefix_installed',False):raise RuntimeError('Already installed')
    model=policy.model
    original_sample=model.sample_actions
    method_name="predict_best_of_n" if hasattr(critic,"predict_best_of_n") else "_best_of_n_actions_soon"
    original_predict=getattr(critic,method_name)
    stats=dict(optimized_batches=0,fallback_batches=0,prefix_rows_original=0,prefix_rows_computed=0)
    active_repeats=1

    def sample(self,images,img_masks,lang_tokens,lang_masks,state,noise=None):
        repeats=active_repeats
        tensors=[*images,*img_masks,lang_tokens,lang_masks,state]
        total=state.shape[0]
        valid=(not self.training and self.config.use_cache and repeats>1 and total%repeats==0)
        if valid:
            valid=all(x.shape[0]==total and torch.equal(x,x[::repeats].repeat_interleave(repeats,dim=0)) for x in tensors)
        stats['prefix_rows_original']+=total
        if not valid:
            stats['fallback_batches']+=1;stats['prefix_rows_computed']+=total
            return original_sample(images,img_masks,lang_tokens,lang_masks,state,noise=noise)
        stats['optimized_batches']+=1;stats['prefix_rows_computed']+=total//repeats
        # Consume exactly the same random numbers, in the same order, as upstream.
        if noise is None:
            noise=self.sample_noise((total,self.config.chunk_size,self.config.max_action_dim),state.device)
        prefix,pad,att=self.embed_prefix([x[::repeats] for x in images],[x[::repeats] for x in img_masks],lang_tokens[::repeats],lang_masks[::repeats],state=state[::repeats])
        _,cache=self.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad,att),position_ids=torch.cumsum(pad,dim=1)-1,past_key_values=None,inputs_embeds=[prefix,None],use_cache=self.config.use_cache,fill_kv_cache=True)
        pad=pad.repeat_interleave(repeats,dim=0);cache=_repeat_cache(cache,repeats)
        dt=torch.tensor(-1.0/self.config.num_steps,dtype=torch.float32,device=state.device)
        time=torch.tensor(1.0,dtype=torch.float32,device=state.device);x_t=noise
        while time>=-dt/2:
            v_t=self.denoise_step(pad,cache,x_t,time.expand(total))
            x_t+=dt*v_t;time+=dt
        return x_t

    def predict(self,*args,**kw):
        nonlocal active_repeats
        old=active_repeats
        active_repeats=kw.get('action_samples') or getattr(self.cfg,'action_samples',1)
        try:return original_predict(*args,**kw)
        finally:active_repeats=old

    model.sample_actions=MethodType(sample,model)
    setattr(critic,method_name,MethodType(predict,critic))
    policy._shared_prefix_installed=True
    def uninstall():
        model.sample_actions=original_sample;setattr(critic,method_name,original_predict)
        del policy._shared_prefix_installed
    return stats,uninstall
