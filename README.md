# My harness for VGAS — shared visual prefix

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2602.07399). The evaluated behavior is in [`my_harness.py`](my_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | Spatial 56.2%; four-suite aggregate 49.0% |

| Our run of the authors' harness | 1,326/2,550 (52.00%) |

| **My harness** | **1,320/2,550 (51.76%)** |


Matched completed LIBERO-Spatial episodes from a user-stopped partial campaign; no Object/Goal/Long evaluation is included.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Compute an identical visual/language prefix once for a proposal batch, then expand its cache across the eight action candidates.

2. Preserve proposal count, RNG draws, denoising, pretrained critic selection and physical action limits.

3. Fall back to native computation when grouped inputs are not identical. Cache lifetime is one proposal batch.


Categories: `uncertainty_compute`.

## Models and inputs

- SemyonXu616/VGAS-5-shot at f23261d42938662851e438cee6e25b2ceaa05ceb; frozen SmolVLA proposal policy and released pretrained critic.

- No critic or policy training is performed by this repository.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/Jyugo-15/VGAS) before installing its model stack.
2. Controller dependency: Use the pinned upstream VGAS PyTorch/LeRobot environment.
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
python -m compileall -q my_harness.py
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

In the prepared upstream VGAS environment, call from my_harness import install; stats = install(policy, critic) before the existing best-of-N evaluation. All policy/critic operations remain upstream.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

220 controlled Spatial actions, 1.0x; reset settling recorded separately.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Slightly lower success, with 5,121.70 versus 2,483.55 recorded policy seconds. Shared-GPU timing is descriptive.

- The campaign is incomplete; these are not the full four-suite results.

- BF16 batch kernels cause numerical differences; this optimization is not bitwise equivalent.

- Common initial-state/reset repairs and changed batch size prevent exact source-protocol replication.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

Source syntax and recorded-core hashes checked. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
