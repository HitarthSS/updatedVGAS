# Evaluation: My harness for VGAS — shared visual prefix

Matched completed LIBERO-Spatial episodes from a user-stopped partial campaign; no Object/Goal/Long evaluation is included.

## Budget

220 controlled Spatial actions, 1.0x; reset settling recorded separately.

## Interpretation

- Slightly lower success, with 5,121.70 versus 2,483.55 recorded policy seconds. Shared-GPU timing is descriptive.

- The campaign is incomplete; these are not the full four-suite results.

- BF16 batch kernels cause numerical differences; this optimization is not bitwise equivalent.

- Common initial-state/reset repairs and changed batch size prevent exact source-protocol replication.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
