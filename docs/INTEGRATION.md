# Integration and arm definitions

In the pinned upstream VGAS environment:

```python
from my_harness import install
stats = install(policy, critic)
# Run the original best-of-eight evaluation loop unchanged.
```

For the authors' arm, omit this installation. It reuses a common visual/language prefix only within one proposal batch, retains eight proposals and the trained critic, and falls back on heterogeneous inputs. It is not a trained replacement policy. The BF16 grouped kernels are not bitwise identical to ungrouped calls, so identical RNG draws do not prove identical actions. Reference reset repair and partial-campaign scripts are included for audit, not as a claim of four-suite replication.
