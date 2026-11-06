"""Keep LIBERO's BDDL-generated property samplers independent of reset count.

The pinned upstream builds these on every hard reset but appends to a persistent
list. Clearing only this generated list before rebuilding restores the initial
sampler set. Applied in this process to both arms; installed code stays intact.
"""
def install():
 from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
 if getattr(BDDLBaseDomain,'_vgas_reset_repair',False):return
 original=BDDLBaseDomain._add_placement_initializer
 def rebuild(self):
  self.object_property_initializers=[]
  return original(self)
 BDDLBaseDomain._add_placement_initializer=rebuild
 BDDLBaseDomain._vgas_reset_repair=True
